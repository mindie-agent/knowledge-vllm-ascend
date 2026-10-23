---
schema: mindie-graded-unit/1
id: unit-vllm-omni-multistage-ttfp-rtf-and-pipeline-convergence-workflow
title: vLLM-Omni multi-stage TTFP/RTF/E2EL bottleneck attribution and pipeline convergence workflow
tier: L3
category: workflow
evidence_strength: reproduced_empirical
domain_path: vllm-ascend/vllm-omni/workflows
inherits_from:
  - unit-vllm-omni-multistage-pipeline-deploy-and-npu-platform-axioms
  - unit-vllm-omni-npu-ar-generation-runner-and-static-shape-decode-sop
  - unit-vllm-omni-audio-tts-and-duplex-npu-model-adaptation-sop
  - unit-vllm-omni-diffusion-and-mindie-sd-npu-serving-sop
conditions:
  framework: vllm-omni
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "TTFP equals E2EL despite streaming enabled"
    - "RTF > 1.0 under multi-request concurrency"
    - "YuanrongTransferEngineConnector requires an NPU memory pool"
  keywords:
    - vllm-omni-workflow
    - TTFP
    - RTF
    - E2EL
    - stage-bottleneck-attribution
    - YuanrongTransferEngineConnector
    - OmniConnector
    - pipeline-convergence
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:docs/design/qwen3_omni_tts_performance_optimization.md
  source_digest: db8c05846fc8a283e9d493562cc4d35849a7d96d8a155ada7903bea9daa7d8fc
  evidence_refs:
    - vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:docs/design/qwen3_omni_tts_performance_optimization.md
    - vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:vllm_omni/platforms/npu/omni_connectors/yuanrong_transfer_engine_connector.py
    - vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:docs/design/architecture_overview.md
  supersedes: []
revision: 1
---
## Multi-Stage Convergence Workflow

### Step 1: Decompose Stage-Wise Wall Clock & Modality Metrics
1. Record the modality-appropriate vector at target concurrencies ($C \in \{1, 4, 10\}$):
   - **Omni / Speech (`Thinker -> Talker -> Code2Wav`)**: Measure **TTFT** (Stage 0 text), **TTFP** (first audio packet at client), **RTF** (total audio generation time / audio duration), and **E2EL**.
   - **Diffusion (`Encoder -> DiT -> VAE`)**: Measure encoder time, per-step and total `<Pipeline>.diffuse` time, `<Pipeline>.decode` (VAE/TAE) time, and HTTP delivery overhead.
2. Classify the primary bottleneck:
   - **Case A ($\text{TTFP} \approx \text{E2EL}$)**: Cross-stage handoff is serialized (whole-request buffering) or client streaming output is disabled.
   - **Case B ($\text{RTF} < 1$ at $C=1$, $\text{RTF} > 1$ at $C \ge 4$)**: Stage batching (`stages[].max_num_seqs`) or cross-request `Code2Wav` batching is unconfigured, or reference-audio prefetch is starving active streams.
   - **Case C (High `Talker` TPOT / Step Jitter at $C=1$)**: Host launch overhead or per-step ACLGraph FIA `actual_seq_lengths_kv` rebind in Stage 1.
   - **Case D (Diffusion E2EL Dominated by DiT vs VAE/Host)**: If DiT $> 85\%$, route to DiT parallel/sparse/quant/cache convergence; if VAE + Host $\ge 10\%$ (common on 4-step distilled models), route to VAE slicing and host delivery optimization.

### Step 2: Converge Cross-Stage Streaming & `OmniConnector` Transport
1. **Enable `async_chunk` + Streaming Output Together**:
   - Verify `async_chunk: true` on producer/consumer stages and streaming audio output on the serving endpoint so `Thinker` forwards hidden-state chunks and `Talker` forwards codec token chunks incrementally to `Code2Wav`.
2. **Select the Optimal `OmniConnector` for Deployment Topology**:
   - **Single-Node Colocated / Partitioned NPUs**: Use `SHMConnector` or packed `OmniSerializer` bytes across `EngineCore` (`model_intermediate_buffer`) to avoid Python list-to-tensor serialization overhead.
   - **Multi-Node / Disaggregated NPU Stages**: Configure `YuanrongTransferEngineConnector` (`protocol: "ascend"`, `memory_pool_device: "npu"`, 4096-byte aligned `BufferAllocator`, ZMQ control plane + HCCL/Ascend direct NPU memory pool transfer) or `MooncakeTransferEngineConnector`. Verify device stream synchronization (`_sync_tensor_device`) before registering sender buffers so consumers never read in-flight NPU memory.

### Step 3: Converge Stage-Specific NPU Execution (AR / Vocoder / DiT)
1. **Stage 0 & Stage 1 (`Thinker` / `Talker` AR)**:
   - Enable `enforce_eager: false` on `Thinker` and `Talker`.
   - For short-context `Talker` decoders (`max_model_len <= 4096`), verify `OmniStaticShapeAttentionBackend` engages (`[minicpmo] static-shape decode attention on (max_model_len=..., kv_capacity=...)`) to eliminate per-step FIA task updates, and enable `talker_multiframe` with `_narrow_decode_query_len` (`uniform_decode_query_len=1`).
2. **Stage 2 (`Code2Wav` Vocoder / Flow Matching)**:
   - On `Qwen3-TTS` / `Qwen3-Omni`, verify `apply_qwen3_tts_patches()` casts Linear weights to `FRACTAL_NZ` and Conv1d weights to `FRACTAL_Z` (or keeps `ND` with CPU mel-spectrogram on Ascend 950 A5).
   - On `MiniCPM-o 4.5`, verify `cfm_fused_body` is active on NPU (packed QKV, `npu_add_layer_norm`, single-transpose causal conv cache) and `cfm_max_graphs` is sized to cover both steady-state and final-chunk lookahead buckets without exhausting the capture arena.
3. **Diffusion Stages (`DiffusionEngine`)**:
   - Converge lossless parallelism first (`--usp`, `--text-encoder-tp-size` aligned to `dit_world`, `--enable-distributed-layerwise-offload` AllGather path when HBM-bound), then stack lossy accelerators (`--cache-backend cache_dit`, `RAINFUSION_ATTN` on 3D `BSND` video, and `int8`/`mxfp8` online quantization) with runtime counter verification.

### Step 4: Verify End-to-End Acceptance
1. Confirm **TTFP**, **RTF** ($< 1.0$ under target concurrency), and **E2EL** improvements in a same-window A/B comparison against the frozen baseline.
2. Confirm output fidelity (audio ASR/UTMOS or image/video PSNR/SSIM + frame health check) before promoting the deployment profile.
