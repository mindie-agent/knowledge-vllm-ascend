---
category: sop
conditions:
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  - Ascend950
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
domain_path: vllm-ascend/distributed/pd-layout
evidence_strength: software_contract
id: unit-vllm-ascend-pd-connector-layout-compatibility-audit-sop
inherits_from:
- unit-vllm-ascend-pd-kv-connector-layout-and-handoff-invariants
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/b61c9b207f72e7765bf096a5dff2fc928a19b03f/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py#L272
  - https://github.com/vllm-project/vllm-ascend/blob/b61c9b207f72e7765bf096a5dff2fc928a19b03f/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py#L1102-L1323
  - https://github.com/vllm-project/vllm-ascend/blob/b61c9b207f72e7765bf096a5dff2fc928a19b03f/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py#L750
  - https://github.com/vllm-project/vllm/blob/7913c96894cacc348397a377411acde96b63e68a/vllm/v1/kv_cache_interface.py#L917-L1419
  source_locator: vllm-project/vllm-ascend:b61c9b207f72e7765bf096a5dff2fc928a19b03f:vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py
  source_type: upstream_code_audit_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: PD disaggregation cache spec, page-sharing, and TP/PCP/DCP topology compatibility audit SOP
triggers:
  error_signatures:
  - Asymmetric TP, PCP, or DCP between Prefill and Decode scrambles token positions or KV heads
  - Hybrid model with MambaSpec or CircularBufferSpec fails or corrupts state under PD disaggregation
  - Mooncake Hybrid connector corrupts KV when P and D physical block strides differ
  - Page-shared MLA or indexer KVCacheTensor with layer_stride == 0 overwrites sibling views during pull
  keywords:
  - PD layout audit
  - CircularBufferSpec
  - MambaSpec
  - MLAAttentionSpec
  - SlidingWindowSpec
  - KVCacheTensor
  - layer_stride
  - block_stride
  - PCP
  - DCP
  - cp_kv_cache_interleave_size
  - mooncake_hybrid_connector
  - pull_worker
---

# PD disaggregation cache spec, page-sharing, and TP/PCP/DCP topology compatibility audit SOP

## When to Use

Run this audit before enabling a new model cache layout (MLA, GDN/Mamba hybrid, sliding window, indexer) or asymmetric parallelism (`P_tp != D_tp`, `PCP > 1`, `DCP > 1`) under `vllm-ascend` PD disaggregation.

## Step 1 — Check Cache Group Specs (`kv_cache_interface.py:917-1419`)

- **`FullAttentionSpec` / `MLAAttentionSpec`:** Standard block-table indexing; for MLA check whether `kv_c` and `k_pe` are separate tensors or share a physical page.
- **`SlidingWindowSpec`:** Only trailing window blocks stay resident; missing prefix windows require scheduler prefix replay (`num_computed_tokens < N - 1`).
- **`CircularBufferSpec` (`kv_cache_interface.py:917`):** Allocates **1 block per request**, disables prefix cache, and does not use linear slot mapping. Never apply multi-block linear DMA assumptions.
- **`MambaSpec` (`kv_cache_interface.py:1043`):** Valid bytes (`conv_state`, `ssm_state`) are smaller than padded page stride. In `align` mode, block-table rows grow with sequence length while only bounded physical states stay live—copy only active state slots at exact sub-page offsets.

## Step 2 — Check `KVCacheTensor` Page Sharing (`layer_stride == 0`)

Inspect `offset`, `layer_stride`, and `block_stride` (`vllm_ascend/core/kv_cache_interface.py:205,309`). When `layer_stride == 0` (multiple KV/MLA/indexer views sharing one physical buffer), transfer the underlying page once or copy each sub-view at its exact `inner_offset` and `valid_bytes`. Never write the full page per view.

## Step 3 — Audit Asymmetric Topology (TP / PCP / DCP)

1. **TP (`P_tp != D_tp`):** Verify head scatter/gather ratios. For MLA, `k_pe` has 1 shared head across TP ranks—do not slice a 1-head tensor or concatenate replicated copies.
2. **PCP (`PCP > 1`):** PCP splits `Q` while KV is often replicated across PCP ranks. Pull from one complete KV replica on P and verify P/D agree on rank-key flattening order (`tp * PCP + pcp` vs `pcp * TP + tp`).
3. **DCP (`DCP > 1`):** DCP interleaves tokens inside blocks via `cp_kv_cache_interleave_size`. With `block_size=4, DCP=2, interleave=1`, rank 0's first block holds tokens `[0, 2, 4, 6]`; whole-block copy to a `DCP=1` consumer expecting `[0, 1, 2, 3]` scrambles token order unless token-level un-interleaving runs.

## Step 4 — Match Connector Stride Handling

- **Mooncake V2 (`base_worker.py:272`, `pull_worker.py:1102,1183,1323`):** Tracks P and D strides independently and coalesces blocks only when both sides satisfy `stride == payload_length`.
- **Mooncake Hybrid (`mooncake_hybrid_connector.py:750`):** Uses one local stride for both sides; safe only when P and D have identical block layouts.
