---
schema: mindie-graded-unit/1
id: unit-vllm-omni-npu-ar-generation-runner-and-static-shape-decode-sop
title: vLLM-Omni NPU AR/Generation ModelRunner, OmniStaticShapeAttentionBackend, and talker_multiframe SOP
tier: L2
category: sop
evidence_strength: software_contract
domain_path: vllm-ascend/vllm-omni/runner-and-attention
inherits_from:
  - unit-vllm-omni-multistage-pipeline-deploy-and-npu-platform-axioms
  - unit-vllm-ascend-npu-runtime-and-graph-invariants
conditions:
  framework: vllm-omni
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
    - Ascend 310P
triggers:
  error_signatures:
    - "acl 507035"
    - "Cannot capture CUDA graphs in an inappropriate time"
    - "AttributeError: 'NPUARModelRunner' object has no attribute 'enable_hamming_sparse'"
    - "State error: sample_tokens() must be called after execute_model() returns None"
  keywords:
    - OmniStaticShapeAttentionBackend
    - static_shape_decode
    - static_shape_prefill
    - NPUARModelRunner
    - NPUGenerationModelRunner
    - talker_multiframe
    - _NPUHostCopyBatch
    - ascend_warmup_patch
    - propose_draft_token_ids
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:vllm_omni/platforms/npu/attention/static_shape_backend.py
  source_digest: 60829eb919542febe93e61fe29c7788e403c7da6432f6bce87f9a045fb8202ec
  evidence_refs:
    - vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:vllm_omni/platforms/npu/attention/static_shape_backend.py
    - vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:vllm_omni/platforms/npu/worker/npu_ar_model_runner.py
    - vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:vllm_omni/platforms/npu/worker/npu_generation_model_runner.py
  supersedes: []
revision: 1
---
## Symptom & Mechanism

In multi-stage Omni models (such as `MiniCPM-o 4.5` and `Qwen3-Omni`), Stage 1 (`Talker`) is a compact, short-context autoregressive decoder (`max_model_len=4096`, e.g., 20 layers, 768 hidden size). Under stock `vllm-ascend` `FULL_DECODE_ONLY` ACLGraph capture, `npu_fused_infer_attention_score` (FIA) takes `actual_seq_lengths_kv` as a host `SymInt[]` list whose tiling is baked into the captured task. Because KV length grows by 1 every step, `update_full_graph_params` (`graph_task_update_begin`/`_end`) must re-issue the FIA op for every layer on every decode step—dominating the Stage 1 main-thread wall time, and faulting during capture on Ascend 910B (`acl 507035`) when multi-frame K-step decode is armed.

## Standard Operating Procedure

1. **Engage `OmniStaticShapeAttentionBackend` for Short-Context Talker Decoders**:
   - `NPUOmniPlatform.get_attn_backend_cls` routes `AscendAttentionBackend` to `vllm_omni.platforms.npu.attention.static_shape_backend.OmniStaticShapeAttentionBackend` and installs `static_shape_decode` into `vllm-ascend` ACLGraph (`install_into_ascend_aclgraph()` and `install_into_ascend_backend()`).
   - Engagement is gated solely by `static_shape_decode.capacity_for(max_model_len, block_size)`: short-context decoders (`max_model_len=4096`) engage per-capacity KV buckets, whereas long-context `Thinker` stages (`max_model_len=32768`) exceed the ceiling and transparently keep the stock attention path.
   - **Prewarm Outside Graph Capture**: `OmniStaticShapeAttentionBackendImpl.__init__` calls `static_shape_decode.prewarm(...)` at model load across all declared query lengths `1 .. 1 + num_speculative_tokens` and buckets so bias buffers are never allocated from a graph's private memory pool during capture.
   - **Guard `cudagraph_mode`**: `OmniStaticShapeMetadataBuilder` only populates `_omni_buckets` when `cudagraph_mode` is `FULL`, `FULL_DECODE_ONLY`, or `FULL_AND_PIECEWISE`. Under `PIECEWISE` or eager mode, `_omni_buckets` is cleared to `()` to prevent runtime lookup of uncaptured per-capacity graph dicts (`Cannot capture CUDA graphs in an inappropriate time`).
   - **Skip Per-Step Rebind**: When `static_shape_decode.is_static_shape_graph(num_tokens)` is true, `OmniStaticShapeAttentionBackendImpl.update_graph_params` calls `static_shape_decode.sync_captured_inputs(num_tokens, seq_lens_device, block_tables)` and returns immediately without calling `graph_task_update_begin`/`_end`.
   - **Single Fresh Prefill Graph Replay (`static_shape_prefill`)**: When a step is not a decode step, `static_shape_prefill.maybe_mark_step(attn_metadata, AscendAttentionState.PrefillNoCache, block_size)` replays a captured prefill bucket for single fresh prefills instead of paying ~18 ms of eager dispatch.

2. **Configure `NPUARModelRunner` Multi-Frame Decode (`talker_multiframe`)**:
   - Stage 1 declares `speculative_config` (`num_speculative_tokens = K - 1`) so the vLLM scheduler reserves $K$ KV slots per step in the block table while `talker_multiframe` generates $K$ codec frames sequentially inside a single `execute_model` call.
   - **Narrow Decode Graph Capture (`_narrow_decode_query_len`)**: Multi-frame replay reads one query row of its graph at a time (a 1-query replay is ~0.6 ms faster than a $K$-query replay). When `talker_multiframe.narrow_replay_enabled(runner)` is active, `_check_and_update_cudagraph_mode` and `_capture_cudagraphs` temporarily force `uniform_decode_query_len = 1` while iterating `sorted(buckets, reverse=True)` under `static_shape_decode.capturing_bucket(capacity)`.
   - **Uniform Multi-Query Decode Identification**: Because `vllm-ascend` rewrites `SpecDecoding` to `ChunkedPrefill` for non-MTP proposers, `OmniStaticShapeMetadataBuilder._is_uniform_multi_query_decode` inspects `actual_seq_lengths_q` and `common.is_prefilling` directly to set `attn_metadata.static_shape_decode_step = True`.
   - **Draft Token Handoff & Triton Warmup Guard**: `NPUARModelRunner.propose_draft_token_ids` emits repeated `continue` tokens up to `talker_multiframe.drafts_this_step(self)` (noting that `vllm-ascend` passes `valid_sampled_token_ids` as the first positional argument rather than `scheduler_output`). Install `apply_ascend_warmup_patch()` before Ascend kernel warmup so Triton rejection/penalties warmup handles multi-frame decode safely.

3. **Enforce Producer-Stream D2H Ordering in `NPUGenerationModelRunner`**:
   - Non-autoregressive stages (`Code2Wav`) use `_NPUHostCopyBatch` (`pin_memory=True`) to issue `host.copy_(tensor, non_blocking=True)` on `torch.npu.current_stream(device)`, call `tensor.record_stream(stream)`, and record a single `torch.npu.Event()` before returning `_AsyncNPUGenerationOutput`. This orders D2H copies before the next forward pass can overwrite graph-owned output buffers and protects allocator reuse if a queued output is discarded.
   - When `num_scheduled_tokens <= 0` after an active step, `NPUGenerationModelRunner` invokes optional model hooks (`on_requests_finished`, `on_requests_added`, `run_idle_prefetch`) inside `_call_optional_model_hook` (catching and logging once per hook so a cache-warming failure never kills `EngineCore`).

4. **Apply Ascend 310P (`_310p`) Worker & TTS Patches When Targeting 310P**:
   - `apply_310p_patches()` activates automatically when `is_310p()` is true, overriding `Qwen3TTSCode2Wav` decoder runtime dtype to `torch.float16` (310P does not support BF16 conv/vocoder paths), running mel-spectrogram `torch.stft` front-ends on CPU, and adapting worker memory/format casts for 310P.
