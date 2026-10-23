---
schema: mindie-graded-unit/1
id: unit-vllm-omni-audio-tts-and-duplex-npu-model-adaptation-sop
title: vLLM-Omni Speech/Audio/TTS and Full-Duplex NPU model adaptation SOP (Qwen3-Omni, Qwen3-TTS, MiniCPM-o 4.5, CosyVoice3)
tier: L2
category: sop
evidence_strength: software_contract
domain_path: vllm-ascend/vllm-omni/models-audio-tts
inherits_from:
  - unit-vllm-omni-multistage-pipeline-deploy-and-npu-platform-axioms
  - unit-vllm-ascend-soc-generation-feature-matrix-and-abi-axioms
conditions:
  framework: vllm-omni
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
    - Ascend 310P
triggers:
  error_signatures:
    - "conv2d convolution operator in the code2wav module of Qwen3-TTS not being able to run on Aclnn"
    - "torch.stft is unsupported"
    - "ref_audio_max_seconds"
  keywords:
    - Qwen3-Omni
    - Qwen3-TTS
    - Qwen3TTSCode2Wav
    - MiniCPM-o-4.5
    - minicpmo_4_5_code2wav
    - cfm_fused_body
    - async_chunk
    - CosyVoice3
    - Step-Audio2
    - MOSS-TTS
    - VoxCPM2
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:vllm_omni/platforms/npu/models/qwen3_tts.py
  source_digest: 63eca4a4e1151f4a0ddb480139b765ff55b574ce15f1e6036aafe5227e316b5d
  evidence_refs:
    - vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:vllm_omni/platforms/npu/models/qwen3_tts.py
    - vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:docs/design/qwen3_omni_tts_performance_optimization.md
    - vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:docs/design/minicpm_o45_mrv2_performance.md
    - vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:docs/design/qwen3_omni_mrv2_performance.md
  supersedes: []
revision: 1
---
## Standard Operating Procedure

1. **Enable the Three-Layer Stacked Optimization Playbook for `Qwen3-Omni` & `Qwen3-TTS`**:
   - **Stage-Wise Batching**: Configure `stages[].max_num_seqs` so `Thinker` and `Talker` run **continuous batching** while `Code2Wav` runs **static batching** (completing one vocoder batch before starting the next).
   - **Graph Capture on Critical Decode Stages**: Set `enforce_eager: false` on `Thinker` and `Talker` while keeping `Code2Wav` on eager or length-bucketed graphs as appropriate.
   - **Mutual Dependency of `async_chunk` and Audio Streaming Output**: Always enable **both** `async_chunk: true` and streaming output together:
     - `async_chunk` forwards hidden-state chunks (`Thinker -> Talker`) and codec token chunks (`Talker -> Code2Wav`) as soon as the first chunk is ready, overlapping cross-stage compute and communication.
     - Without `async_chunk`, audio streaming output cannot take effect early; without streaming output, `async_chunk`'s **TTFP** reduction (up to 95–99% reduction in first-packet latency) is hidden behind full-sequence buffering.

2. **Apply `Qwen3-TTS` & `Qwen3-Omni` Code2Wav NPU Runtime and SoC-Specific Patches**:
   - **Code2Wav ACLNN Format Guard (`apply_qwen3_tts_code2wav_patch`)**: Before initializing `Qwen3TTSCode2Wav`, set `torch.npu.config.allow_internal_format = False` and `torch.npu.set_compile_mode(jit_compile=False)` to prevent ACLNN conv2d format failures.
   - **Decoder Weight Format Casting (`_prepare_npu_decoder_weights`)**:
     - Cast all `nn.Linear` weights via `maybe_trans_nz(module.weight.data)`.
     - For `nn.Conv1d` and `nn.ConvTranspose1d` with `groups == 1`: on Ascend 910B/910C, cast weights to `_ACL_FORMAT_FRACTAL_Z` (`4`) via `torch_npu.npu_format_cast(weight.contiguous(), 4)`. On **Ascend 950 (`is_a5() == True`)**, `FRACTAL_Z` conv weight layout is unsupported—keep contiguous `ND` layout.
   - **A5 & 310P Speaker-Encoder Mel Front-End (`_mel_spectrogram_on_cpu = True`)**: On Ascend 950 (`is_a5()`) and Ascend 310P (`is_310p()`), `torch.stft` is unavailable on NPU; `_apply_a5_prompt_embeds_builder_patch()` swaps `Qwen3TTSPromptEmbedsBuilder` in both `prompt_embeds_builder` and `qwen3_tts_talker` so the mel spectrogram runs on CPU and transfers the resulting tensor to NPU. On 310P, `_npu_decoder_runtime_dtype` also overrides the decoder to `torch.float16` and recomputes `precompute_snake_caches()`.

3. **Optimize `MiniCPM-o 4.5` Turn & Full-Duplex Pipelines on NPU**:
   - **NPU Fused CFM DiT Body (`cfm_fused_body`)**: Enabled by default on NPU for FP32 estimator/cache types (independent of CUDA TF32 flags; set `cfm_fused_body: false` in stage connector `extra` to revert). It replaces per-block gather/transpose/copy operations with:
     - Packed QKV projections and in-place KV cache updates.
     - `torch_npu.npu_add_layer_norm` for every modulated residual + LayerNorm.
     - Causal convolutions lowered as a single `im2col` GEMM over contiguous frames-major history (`[k * C_in + c]` columns), gathering both causal caches via `index_select` into one frames-major staging tensor and writing back to `(N, C, W)` in a **single transpose per body** instead of per block.
   - **In-Tree Torch CAM++ Speaker Embedding**: Runs `campplus.onnx` weights on NPU in FP32 via the in-tree Torch module (set `token2wav_device_spk: false` in connector `extra` to fall back to CPU ONNX Runtime for accuracy triage).
   - **Turn-Mode Conditioning Transport & Idle Reference Prefetch**: Pack reference audio and `Thinker` hidden states as owned bytes via `OmniSerializer` across `EngineCore` (`model_intermediate_buffer`) to eliminate Python scalar-list construction. Keep `token2wav_ref_prefetch_max_waiters` at `1` on shared devices unless cold/warm TTFP/RTF measurements at concurrency 4 justify increasing it.
   - **Full-Duplex EOS Discipline**: In duplex mode, Stage 0 (`Thinker`) keeps `async_chunk: false`; Stage 1 (`Talker`) maintains the 16-frame codec penalty history across conditions and flushes a segment **only on an EOS token sampled in the current output**, never on `turn_end` alone.

4. **Adapt Additional Speech/Audio Models (`CosyVoice2/3`, `Step-Audio2`, `MOSS-TTS`, `VoxCPM2`)**:
   - Install `apply_voxcpm2_talker_patch()` inside `NPUOmniPlatform.set_device()` (worker scope, after platform initialization) to avoid re-entering `current_omni_platform` during `__init__`.
   - Use NPU-specialized attention and streaming wrappers (`cosyvoice2_dit_attn.py`, `step_audio2_token2wav.py`, `moss_tts_streaming_decode_wrapper.py`, `qwen3_tts_tokenizer_v2.py`) to keep vocoder and streaming decode steps on fused ACLNN kernels.
