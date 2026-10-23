---
schema: mindie-graded-unit/1
id: unit-vllm-omni-diffusion-and-mindie-sd-npu-serving-sop
title: vLLM-Omni Diffusion and MindIE-SD NPU serving SOP (MiniMax-H3, Qwen-Image, FLUX, Wan2.2, RainFusion, Cache-DiT, DLO)
tier: L2
category: sop
evidence_strength: reproduced_empirical
domain_path: vllm-ascend/vllm-omni/models-diffusion
inherits_from:
  - unit-vllm-omni-multistage-pipeline-deploy-and-npu-platform-axioms
  - unit-vllm-ascend-npu-runtime-and-graph-invariants
conditions:
  framework: vllm-omni
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "Missing preprocess images"
    - "pixel_values and image_grid_thw must be provided together"
    - "IndexError: tuple index out of range"
    - "AI Core 507015"
    - "staying dense"
  keywords:
    - MiniMax-H3
    - Qwen-Image
    - QwenImagePipeline
    - QwenImageEditPlusPipeline
    - RAINFUSION_ATTN
    - cache_dit
    - enable-distributed-layerwise-offload
    - text-encoder-tp-size
    - mm_swiglu_mxquant
    - FlashGen
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:vllm_omni/platforms/npu/models/minimax_h3.py
  source_digest: 980b621e474ecdc1e2e2e35a6c8f21189b6af6009dafe2324a05f5abcb6542d3
  evidence_refs:
    - vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:vllm_omni/platforms/npu/models/minimax_h3.py
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/framework-integration/references/vllm-omni-enablement.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/framework-integration/references/framework-support-matrix.md
  supersedes: []
revision: 1
---
## Standard Operating Procedure

1. **Launch `vllm-omni` Diffusion Serving & Select the Correct Endpoint**:
   - Export required Ascend runtime variables before launching `vllm serve <model_dir> --omni`:
     ```bash
     export VLLM_WORKER_MULTIPROC_METHOD=spawn
     export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
     export HCCL_NPU_SOCKET_PORT_RANGE="auto"
     ```
   - **Endpoint Selection Trap**: Text-to-Image pipelines (`QwenImagePipeline`) use JSON POST to `/v1/images/generations`. Image-Edit / I2I pipelines (`QwenImageEditPlusPipeline`, e.g., `Qwen-Image-Edit-2511`) **must** use multipart form POST (`-F "image=@..."`) to `/v1/images/edits`; sending JSON returns HTTP 500 `Missing preprocess images`.
   - **Worker Process Cleanup**: `pkill -f "vllm serve"` only terminates the parent process while `vLLM-Omni::DiffusionWorker` child processes remain alive and hold NPU HBM. Always terminate the process tree of your own serve PID and verify HBM release before switching configurations.

2. **Configure Attention Routing (`FLASH_ATTN` vs `TORCH_SDPA` vs `RAINFUSION_ATTN`)**:
   - **Auto-Routing & Baseline Freeze**: When `mindiesd` is installed in the environment, `vllm-omni` automatically routes diffusion attention to `FLASH_ATTN` and replaces `RMSNorm`/`RoPE`/`AdaLN` with Ascend custom ops (plus `apply_minimax_h3_qwen3vl_patch` for fused BNSD RoPE, native GQA SDPA, and packed `npu_swiglu` in the MiniMax-H3 Qwen3-VL encoder). To establish an unaccelerated baseline, explicitly pass `--diffusion-attention-config '{"default":{"backend":"TORCH_SDPA"}}'` and verify the resolution log.
   - **Sparse Attention (`RAINFUSION_ATTN` / `rf_v2`) Preconditions**:
     - Requires 3D video token layout (`qkv_layout='BSND'` with `video_spans`). On 2D image models (`Qwen-Image-2512`), `RAINFUSION_ATTN` logs `staying dense` and produces byte-identical output to lossless dense FA—classify this fail-closed as **not engaged**, never as "low speedup".
     - **`end_step` Semantics**: `end_step` specifies the number of **final steps kept dense** (not the step index where sparsity stops). Setting `end_step` equal to `num_inference_steps` forces dense execution across all steps.
     - **CP $\times$ Sparse Compatibility**: `RAINFUSION_ATTN` is compatible with Ulysses (`--usp`) and composite AllGather-KV $\times$ Ulysses (`--usp k --allgather-degree m`), which gather full K/V before sparse block selection. It is **incompatible** with Ring attention (`--ring > 1` is rejected or bypasses the attention backend, silently disabling sparsity).

3. **Configure Parallelism (`--usp`, `--tensor-parallel-size`, `--text-encoder-tp-size`) & Offload (`DLO`)**:
   - **Align `--text-encoder-tp-size` under Sequence Parallelism**: When using `--usp > 1` on `MiniMax-H3` (e.g., `TP1xUSP4` or `TP2xUSP2`), set `--text-encoder-tp-size` equal to the DiT world size (`dit_world`), otherwise ranks outside `[0, tp)` fail with a `GroupCoordinator` `cpu_group is None` assertion in `_build_text_encoder_group`. Note that pure TP without USP (`usp=1`) on long-video `MiniMax-H3` T2VA/FL2VA fails at the vision/denoise input seam (`pixel_values and image_grid_thw must be provided together` or `IndexError` in `_build_denoise_inputs`) regardless of `text_encoder_tp_size`—use USP (`TP2xUSP2` or `TP1xUSP4`).
   - **CFG Parallel Guard**: Never set `--cfg-parallel-size 2` when `true_cfg_scale <= 1.0` or `negative_prompt` is omitted (single-branch CFG inference).
   - **Unlock High-USP BF16 Shapes with DLO**: When `TP1xUSP4` exceeds single-NPU HBM in BF16, enable `--enable-distributed-layerwise-offload` (**DLO**), which stores $1/\text{DP}$ weights on host and overlaps H2D with AllGather (always use the default AllGather path; the rank-local H2D no-AllGather path is several times slower). Avoid ordinary `--enable-layerwise-offload` where it triggers host OOM killers, and note that `FastH3` / 4-step runtime PEFT LoRA and online `mxfp4` are mutually exclusive with DLO.

4. **Apply Online Quantization (`int8` / `mxfp8`), Catlass FFN Fusion, and `cache_dit` / FlashGen LoRA**:
   - **Online Quantization**: Pass `--diffusion-quantization-config '{"transformer":{"method":"int8","activation_scheme":"dynamic"}}'` (or `"method":"mxfp8"`). Verify `DynamicQuantV2 -> QuantBatchMatmulV3` takes over the 5 GEMMs per DiT block (`qkv`, `out`, `fc1`, `down`, `adaln`).
   - **Marginal Gain Synergy on Long Video**: On long-sequence video (`15s 768P`), $O(S^2)$ attention dominates, so standalone linear quantization shows small initial gain—but its marginal speedup **increases monotonically** after `RAINFUSION_ATTN` and `cache_dit` compress attention cost. Never prune linear quantization based solely on its standalone short-circuit share.
   - **Compile $\times$ FFN-MX Fusion (`mm_swiglu_mxquant`)**: While whole-transformer `MindieSDBackend` compilation in `vllm-omni` is off by default for BF16/image paths, enabling `method:mxfp8` with the `mm_swiglu_mxquant` `GraphPatternEntry` fuses `QMatMul + SwiGLU + MXQuant` on `MiniMax-H3`, using C++ `AdaptLayoutCached` to transpose vLLM's `(K, N)` weights once and calling `aclrtSynchronizeStream` before kernel launch to prevent cross-stream race `AI Core 507015`.
   - **`cache_dit` vs 4-Step FlashGen Distillation LoRA**:
     - Enable step caching with `--cache-backend cache_dit --enable-cache-dit-summary` (`Fn_compute_blocks`, `Bn_compute_blocks`, `enable_taylorseer`).
     - When serving 4-step distilled LoRA (`--lora-backend peft --lora-path ...`, `num_inference_steps=4`), verify two contracts: (a) `model_index.json` must not pin `base_schedule` so `num_inference_steps=4` means 4 denoiser evaluations rather than 3 intervals; (b) `cache_dit` warmup (`max_warmup_steps`) consumes the entire 4-step budget, rendering cache a byte-identical no-op—disable `cache_dit` on 4-step distilled runs and stack `RAINFUSION_ATTN` + online quantization instead.
