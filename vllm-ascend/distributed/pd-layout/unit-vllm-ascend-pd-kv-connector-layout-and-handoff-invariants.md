---
category: axiom
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
id: unit-vllm-ascend-pd-kv-connector-layout-and-handoff-invariants
inherits_from: []
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm/blob/7913c96894cacc348397a377411acde96b63e68a/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L333
  - https://github.com/vllm-project/vllm/blob/7913c96894cacc348397a377411acde96b63e68a/vllm/v1/core/sched/scheduler.py#L3194-L3218
  - https://github.com/vllm-project/vllm/blob/7913c96894cacc348397a377411acde96b63e68a/vllm/v1/kv_cache_interface.py#L917-L1419
  - https://github.com/vllm-project/vllm-ascend/blob/b61c9b207f72e7765bf096a5dff2fc928a19b03f/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1885-L1918
  - https://github.com/vllm-project/vllm-ascend/blob/b61c9b207f72e7765bf096a5dff2fc928a19b03f/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L360
  - https://github.com/vllm-project/vllm-ascend/blob/b61c9b207f72e7765bf096a5dff2fc928a19b03f/vllm_ascend/core/kv_cache_interface.py#L205-L309
  source_locator: vllm-project/vllm-ascend:b61c9b207f72e7765bf096a5dff2fc928a19b03f:vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py
  source_type: upstream_code_audit_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: PD disaggregation KV connector hook timing, N-1 handoff, and stride invariants
triggers:
  error_signatures:
  - D-side KV tensor dump at model entry shows zeros or stale values while attention reads valid data
  - PD disaggregation HTTP 200 succeeds because failed KV load fell back to local recomputation
  - Mooncake or AscendStore multi-block transfer corrupts Mamba state or page-shared MLA/indexer cache
  - Decode starts earlier than N-1 under sliding-window or hybrid cache prefix replay
  keywords:
  - PD disaggregation
  - KVConnectorBase_V1
  - start_load_kv
  - wait_for_layer_load
  - WAITING_FOR_REMOTE_KVS
  - MooncakeConnector
  - AscendStoreConnector
  - MambaSpec
  - CircularBufferSpec
  - KVCacheTensor
  - layer_stride
  - block_stride
---

# PD disaggregation KV connector hook timing, N-1 handoff, and stride invariants

## Condition

`vllm-ascend` Prefill/Decode (PD) disaggregation (`KVConnectorBase_V1`: Mooncake P2P/Layerwise/Hybrid, `AscendStoreConnector`, `AscendMultiConnector`, `sfa_pd_rd2h`) on Ascend NPU.

## Constraint

1. **Hook timing (`start_load_kv` vs `wait_for_layer_load`):** `KVConnectorBase_V1` (`base.py:333`) does not guarantee where data arrives. `AscendStoreConnector.wait_for_layer_load` (`ascend_store_connector.py:360`) waits and relocates Mamba state; `AscendMultiConnector` (`ascend_multi_connector.py:97`) runs child waits; `mooncake_layerwise_connector.wait_for_layer_load` (`mooncake_layerwise_connector.py:2153`) is a no-op; `sfa_pd_rd2h` (`connector.py:222`) only guards P block reuse. Never dump D-side KV at model entry—capture after layer wait/relocation (`model_runner_v1.py:2661`, `mamba_hybrid.py:102,197`) right before the consuming op.
2. **Scoped `N-1` handoff vs replay:** Full remote hit rewinds 1 token on D (`scheduler.py:3194,3218`) and truncates P prefill by 1 when `N > 1` (`mooncake_connector.py:1885,1892,1918`). Under `SlidingWindowSpec` or hybrid caches, uncached windows trigger bounded prefix replay (`num_computed_tokens < N - 1`), which is normal.
3. **Stride span:** Block address is `base + block_id * block_stride + inner_offset`. When `block_stride > valid_bytes` (`MambaSpec` padded pages or shared pages with `layer_stride == 0` in `vllm_ascend/core/kv_cache_interface.py:205,309`), `N` blocks span `(N - 1) * block_stride + valid_bytes`, never `N * valid_bytes`. Coalesce blocks only when P and D strides equal payload length (`mooncake/base_worker.py:272`).
4. **Transfer proof:** `num_computed_tokens` updates during `WAITING_FOR_REMOTE_KVS` (`scheduler.py:1244,2185`), and failed loads can recompute locally (`scheduler.py:2170`) with HTTP 200. Require non-zero transferred bytes and zero recompute fallback.

## Rationale

Dumping before real layer completion, coalescing padded strides, or ignoring recompute fallback causes false corruption alarms or hides broken transfers.

## Evidence

Pinned `vllm:7913c968` and `vllm-ascend:b61c9b20`.
