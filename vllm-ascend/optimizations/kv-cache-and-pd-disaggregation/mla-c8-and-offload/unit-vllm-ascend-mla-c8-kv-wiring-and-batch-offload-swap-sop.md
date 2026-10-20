---
category: sop
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
domain_path: vllm-ascend/optimizations/kv-cache-and-pd-disaggregation/mla-c8-and-offload
evidence_strength: software_contract
id: unit-vllm-ascend-mla-c8-kv-wiring-and-batch-offload-swap-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/mla_c8_int8_kv_cache_op_wiring.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/experience/interventions/c8_kv_scale_provenance_reuse_offline.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/kvcache_offload_batch_memcpy_async.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/kvcache_offload_sparse_selection_cross_step_reuse.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/experience/interventions/kvcache_offload_host_swap.md
  source_digest: 3dc1acd00c8dcc778e45a9eba1f027870ae629acf67f348464ab6beeed15b3c1
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/mla_c8_int8_kv_cache_op_wiring.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: MLA C8 INT8 KV-cache operator wiring, scale provenance validation, and batched
  async KV host-offload swap in vLLM-Ascend
triggers:
  error_signatures:
  - MLA C8 decode produces NaN or garbage logits when k_nope and k_rope cache layout
    or dequant_scale wiring is mismatched
  - Per-block aclrtMemcpyAsync loop during KV cache host offload saturates host PCIe
    submission overhead
  - Host-offloaded sparse attention decode stalls on H2D bandwidth when re-fetching
    all top-k blocks every step
  keywords:
  - mla_c8_int8_kv_cache_op_wiring
  - c8_kv_scale_provenance
  - aclrtMemcpyBatchAsync
  - kvcache_offload_batch_memcpy_async
  - kvcache_offload_sparse_selection
  - enable_mla_prolog
---
# MLA C8 INT8 KV-cache operator wiring, scale provenance validation, and batched async KV host-offload swap in vLLM-Ascend

## Applicability and symptom scope

Use this SOP when enabling INT8 (`C8`) compressed latent KV cache on MLA/DSA models or when offloading paged KV cache blocks between NPU HBM and Host DRAM in `vllm-ascend`.

## Procedure

1. **Wire MLA C8 packed cache layout and single-scale operator chain (`mla_c8_int8_kv_cache_op_wiring.md`, `c8_kv_scale_provenance_reuse_offline.md`).**
   - Enable `enable_mla_prolog=True` (mandatory precondition for C8 on MLA).
   - In the C8 packed MLA layout, quantized `k_nope` (`512` dims INT8), unquantized `k_rope` (`64` dims BF16 viewed as INT8 bytes), and per-token/per-tile scales are packed into a unified `kv_cache` tensor while `kr_cache` is passed as a preallocated empty tensor `torch.empty((1, 2, 1, 0), dtype=torch.int8, device="npu")`.
   - Route all three stages—`npu_mla_prolog_v2/v3`, `npu_sparse_flash_attention` (or FIA v2), and `npu_lightning_indexer_quant`—to their C8 quantized variants driven by the same calibrated scale tensor.
   - Verify offline scale provenance: C8 static scale packages are tied to a specific checkpoint's activation distribution; re-calibrate or explicitly audit cosine similarity before reusing a sibling model's C8 scale JSON.

2. **Coalesce KV offload block swaps into `aclrtMemcpyBatchAsync` (`kvcache_offload_batch_memcpy_async.md`, `kvcache_offload_host_swap.md`).**
   - Issuing hundreds of individual `aclrtMemcpyAsync` calls (one per paged block and per layer/sub-tensor) during H2D/D2H KV swap makes the offload stream host-launch bound.
   - Coalesce all non-contiguous block pointers (`src_ptrs`, `dst_ptrs`, `sizes`) for a swap batch into a single `torch_npu._C._aclrtMemcpyBatchAsync` call with explicit `aclrtMemcpyBatchAttr` (`srcLoc=DEVICE/HOST`, `dstLoc=HOST/DEVICE`) on a dedicated offload stream, falling back to looped async copy only on CANN runtimes lacking batch-memcpy support or for intra-device D2D copies.
   - Always pin host KV pool memory (`pin_memory=True`) and synchronize the offload stream via `torch.npu.Event` before the attention kernel consumes swapped-in blocks.

3. **Reuse device-resident top-k KV pages across decode steps when combining host offload with sparse selection (`kvcache_offload_sparse_selection_cross_step_reuse.md`).**
   - When the full KV cache resides in Host DRAM and only sparse top-k blocks are fetched to NPU HBM per decode step, maintain a device-resident block cache table so top-k blocks already present on NPU from step $t-1$ are reused in step $t$, issuing H2D transfers only for newly selected delta blocks.
