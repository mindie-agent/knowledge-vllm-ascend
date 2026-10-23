---
schema: mindie-graded-unit/1
id: unit-vllm-omni-multistage-pipeline-deploy-and-npu-platform-axioms
title: vLLM-Omni multi-stage PipelineConfig/DeployConfig and NPUOmniPlatform execution axioms
tier: L1
category: axiom
evidence_strength: software_contract
domain_path: vllm-ascend/vllm-omni/architecture
inherits_from: []
conditions:
  framework: vllm-omni
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
    - Ascend 310P
triggers:
  error_signatures:
    - "Ascend paged attention did not expose an integer kernel block size"
    - "acl 507035"
    - "Dropping non-tensor multimodal output key"
  keywords:
    - vllm-omni
    - NPUOmniPlatform
    - PipelineConfig
    - DeployConfig
    - StageRuntime
    - DiffusionKVCacheMode
    - AscendStrictUlyssesPagedBackend
    - requires_contiguous_pa_kv_cache
    - TTFP
    - RTF
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:vllm_omni/platforms/npu/platform.py
  source_digest: 84e591f034d7878a0f59634a23eb0eaf1367577027d3f6433b8cc5b493ae4b2e
  evidence_refs:
    - vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:vllm_omni/platforms/npu/platform.py
    - vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:docs/design/architecture_overview.md
  supersedes: []
revision: 1
---
## Core Invariants

1. **Four-Layer Multi-Stage Separation (`Model -> PipelineConfig -> StageRuntime -> DeployConfig`)**:
   - `vllm-omni` decomposes multimodal understanding and generation into an explicit DAG of logical stages (`PipelineConfig`), where each stage executes under its own paradigm-specific `StageRuntime`:
     - **Autoregressive (AR) Stages** (`Thinker`, `Talker` AR decoder): continuous batching with KV cache and `NPUARWorker` / `NPUARModelRunner`.
     - **Non-Autoregressive Generation Stages** (`Code2Wav` vocoder, CFM flow matching): static or chunked batching with `NPUGenerationWorker` / `NPUGenerationModelRunner`.
     - **Diffusion Stages** (`DiffusionEngine` DiT + VAE): step-wise denoising with `DiffusionWorker`, sequence/CFG/VAE parallelism, and step/block caching.
     - **Full-Duplex Stages**: bi-directional streaming session control where committed turns or codec frames flush only on explicit model EOS tokens rather than client `turn_end` alone.
   - `DeployConfig` maps logical stages onto physical NPU device groups, replica counts, and cross-stage `OmniConnector` transports (`SHMConnector`, `MooncakeTransferEngineConnector`, `YuanrongTransferEngineConnector`).

2. **Modality-Specific Service Metrics**:
   - Text/AR stages are governed by **TTFT** (Time to First Token) and **TPOT** (Time per Output Token).
   - Streaming speech/audio pipelines (`Qwen3-Omni`, `Qwen3-TTS`, `MiniCPM-o 4.5`, `CosyVoice3`) are governed by **TTFP** (Time to First Audio Packet), **RTF** (Real-Time Factor = generation wall time / audio duration; $\text{RTF} < 1.0$ required for real-time streaming), and **E2EL** (End-to-End Latency).
   - Diffusion image/video pipelines (`MiniMax-H3`, `Qwen-Image`, `Wan2.2`, `FLUX`) are governed by per-step DiT denoising latency (`<Pipeline>.diffuse`), VAE decode latency, and end-to-end request wall time.

3. **`NPUOmniPlatform` Initialization & Memory Layout Contracts**:
   - `NPUOmniPlatform` inherits from both `OmniPlatform` and `vllm_ascend.platform.NPUPlatform` (`dist_backend = "hccl"`).
   - **Unpadded Contiguous PA KV Cache Patch**: `apply_contiguous_kv_patch()` overrides `vllm_ascend.attention.utils.requires_contiguous_pa_kv_cache` to return `spec.page_size_bytes == spec.real_page_size_bytes`, preserving unpadded K/V layout for V2 separate K/V allocation while keeping padded pages on the packed path.
   - **Internal Format Preservation**: `NPUOmniPlatform.set_device()` sets `torch.npu.config.allow_internal_format = True` and calls `enable_custom_op()` so `FRACTAL_NZ` weight layouts are preserved for fused Ascend kernels. Exception: `Qwen3TTSCode2Wav` explicitly resets `allow_internal_format = False` and `jit_compile = False` in its worker before casting `Conv1d`/`ConvTranspose1d` weights (`_ACL_FORMAT_FRACTAL_Z = 4` on non-A5 NPUs, contiguous `ND` on Ascend 950 A5).
   - **Diffusion Worker Patching & Paged KV Geometry**: Because `DiffusionWorker` bypasses `NPUWorker.__init__`, `init_diffusion_worker_vllm_config()` must explicitly invoke `adapt_patch()`, `init_ascend_config(vllm_config)`, and `refresh_all_lora_classes()` so `AscendBlockTables` uses the non-UVA NPU buffer path. When `od_config.diffusion_kv_mode == DiffusionKVCacheMode.PAGED_SCHEDULER`, `configure_diffusion_vllm_config()` overrides vLLM's generic `block_size = 16` with `AscendAttentionBackend.get_supported_kernel_block_sizes()[0]` (`128` tokens for Ascend FIA), sets `requires_diffusion_paged_kv_prewrite() = True`, and routes `ulysses_degree > 1` to `AscendStrictUlyssesPagedBackend` to keep strict Ulysses paged FIA out of vLLM's PCP dispatch.

4. **Wire-Level Tensor-Only Multimodal Payload Contract**:
   - `OmniEngineCoreOutput.multimodal_output` and `msgspec` wire serialization require every cross-stage payload entry to be a `torch.Tensor` (`_ensure_tensor_values`). Scalar `int`/`float`/`bool` values are wrapped via `torch.tensor()`, and non-tensorizable objects are dropped with a warning rather than corrupting EngineCore IPC.
