---
category: sop
conditions:
  component: xlite
  framework: vllm-ascend
domain_path: vllm-ascend/architecture/runner-and-worker/xlite-runtime
evidence_strength: software_contract
id: unit-vllm-ascend-xlite-lightweight-runtime-and-decode-fastpath-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/xlite/xlite_model_runner.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/xlite/xlite.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/xlite/xlite_worker.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/xlite/utils.py
  source_digest: ef24616389c362b12d498b45e9d09cba3e0aa19b5d4c8c658db1a791aa1fc9db
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/xlite/xlite_model_runner.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Xlite lightweight C++/CANN runtime wrapper and decode fast-path execution SOP
triggers:
  error_signatures:
  - XliteModelRunner fails to initialize xlite_wrapper or rejects unsupported model
    architecture
  - xlite decode fastpath mismatches KV cache tensor layout or RoPE cos/sin table
    pointers
  keywords:
  - xlite
  - XliteWorker
  - XliteModelRunner
  - XliteWrapper
  - enable_xlite
  - xlite_model_runner
---
# Xlite lightweight C++/CANN runtime wrapper and decode fast-path execution SOP

## Scope and preconditions

Use this SOP when enabling or troubleshooting the `xlite` lightweight C++ execution engine (`vllm_ascend/xlite/xlite.py`, `xlite_model_runner.py`, `xlite_worker.py`) for low-overhead model forward execution on Ascend NPUs.

## Procedure

1. **Check architecture and configuration gates (`vllm_ascend/xlite/utils.py`):**
   - When `additional_config.enable_xlite` (or Xlite runner selection in `platform.py`) is enabled, `vllm_ascend` routes worker creation to `XliteWorker` and `XliteModelRunner`.
   - Confirm that the target model architecture (e.g., Llama / Qwen dense or MoE variants supported by `XliteWrapper`) and quantization mode match the installed `xlite` native library capabilities.
2. **Bind model weights and KV caches into `XliteWrapper` (`vllm_ascend/xlite/xlite.py`):**
   - After vLLM loads model weights, `XliteWrapper.register_weight` binds each layer's weight tensor pointers (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`, `norm`, `embed_tokens`, `lm_head`) and KV cache block tensors directly into the C++ Xlite runtime without duplicating HBM storage.
   - Verify that any post-load weight format conversion (such as transposition or NZ layout) is completed before `XliteWrapper` captures raw tensor data pointers.
3. **Execute prefill and decode metadata handoff (`XliteModelRunner`):**
   - `XliteModelRunner` prepares flattened token inputs, `positions`, `block_tables`, `seq_lens`, and `slot_mapping` and passes them to `XliteWrapper.forward()`, bypassing Python per-layer dispatch overhead during decode.

## Verification

- Verify that `XliteModelRunner` completes `load_model` and `initialize_kv_cache` without increasing peak HBM beyond the standard weight + KV footprint, and validate greedy token parity against standard `NPUModelRunner`.
