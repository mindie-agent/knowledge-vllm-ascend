---
category: workflow
conditions:
  framework: vllm-ascend
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - vllm-ascend-idle-dp
domain_path: vllm-ascend/optimizations/host-scheduling-and-batching/chunked-prefill-admission
evidence_strength: software_contract
id: unit-vllm-ascend-chunked-prefill-cla-and-kv-admission-retraction-workflow
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-host-scalar-precompute-and-preprocess-elimination-sop
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/cases/online_decode_kv_admission_retraction.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/vllm_ascend_cla_kv_sharing_forces_chunked_prefill.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/qwen3_next_block_size_128_exception_chunked_prefill.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/prefill_micro_batch.md
  source_digest: ea58e21fa0980b66c05f080874bb77541ab5f074e2d60fa904951c2e1cb85a23
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/cases/online_decode_kv_admission_retraction.md
  source_type: cannbot_workflow_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Chunked prefill scheduling, cross-layer KV sharing (CLA) enforcement, and online
  decode KV admission retraction in vLLM-Ascend
triggers:
  error_signatures:
  - Cross-layer attention (CLA) model reads uninitialized KV cache when PrefillNoCache
    path is selected
  - Qwen3-Next chunked prefill or prefix caching silently disables or fails when block_size=128
  - Online decode bursts exhaust free KV blocks causing cascading request preemption
    thrashing
  keywords:
  - online_decode_kv_admission_retraction
  - vllm_ascend_cla_kv_sharing
  - PrefillNoCache
  - ChunkedPrefill
  - qwen3_next_block_size_128
  - max_num_batched_tokens
  - enable_prefix_caching
---
# Chunked prefill scheduling, cross-layer KV sharing (CLA) enforcement, and online decode KV admission retraction in vLLM-Ascend

## Objective

Prevent KV cache corruption on Cross-Layer Attention (CLA) and hybrid linear-attention models (`Qwen3-Next`), and eliminate online decode KV-block exhaustion thrashing under high concurrency in `vllm-ascend`.

## Workflow Steps

1. **Force `ChunkedPrefill` (cache-backed prefill) for Cross-Layer KV-Sharing (CLA) models (`vllm_ascend_cla_kv_sharing_forces_chunked_prefill.md`).**
   - In standard models without prefix caching, `vllm-ascend` may select the `PrefillNoCache` fast path where prefill attention computes directly on transient `K/V` tensors and only scatters them to the paged KV cache for future decode steps.
   - In CLA models (where consumer layers read the KV cache written by an earlier producer layer during the *same* forward pass), `PrefillNoCache` leaves the paged KV cache unpopulated when the consumer layer executes, producing garbage outputs.
   - **Action:** Detect `has_kv_sharing` / CLA layer bindings in model config and unconditionally route prefill through the cache-reading `ChunkedPrefill` attention path.

2. **Align `block_size` constraints for `Qwen3-Next` hybrid GDN + Attention chunked prefill and prefix caching (`qwen3_next_block_size_128_exception_chunked_prefill.md`).**
   - Verify whether custom hybrid kernels (such as `Qwen3-Next` GDN state + full-attention page tables) impose a fixed `block_size` contract (e.g., `block_size == 128` for FIA/MLA vs. Mamba state page sizing). Do not let a hardcoded `block_size != 128` fallback silently disable `enable_chunked_prefill` or `enable_prefix_caching`; reconcile the hybrid block budget via least-common-multiple alignment.

3. **Configure watermark-based online KV admission and proactive decode retraction (`online_decode_kv_admission_retraction.md`).**
   - Under online serving with long outputs, admitting new requests until free KV blocks hit `0` causes simultaneous allocation failures across active decode sequences in subsequent steps.
   - **Action:** Reserve a decode headroom watermark (`kv_admission_watermark`) proportional to `active_running_reqs * horizon_steps / block_size`. When free blocks drop below the retraction threshold, retract the newest low-priority decode requests cleanly (releasing or offloading their KV blocks) before scheduling the next batch.

4. **Budget `max_num_batched_tokens` and `moe_chunk_max_len` to bound prefill activation peak (`prefill_micro_batch.md`).**
   - Cap `max_num_batched_tokens` so intermediate MoE dispatch/GMM activations (`num_tokens * top_k * hidden_size`) stay within available HBM headroom after KV cache allocation.
