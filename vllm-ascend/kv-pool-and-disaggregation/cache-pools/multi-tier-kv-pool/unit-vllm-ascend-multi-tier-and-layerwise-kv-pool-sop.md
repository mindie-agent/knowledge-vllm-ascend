---
category: sop
conditions:
  component: ascend_store_kv_pool
  framework: vllm-ascend
domain_path: vllm-ascend/kv-pool-and-disaggregation/cache-pools/multi-tier-kv-pool
evidence_strength: software_contract
id: unit-vllm-ascend-multi-tier-and-layerwise-kv-pool-sop
inherits_from:
- unit-vllm-ascend-hccl-and-dp-collective-invariants
- unit-vllm-ascend-deterministic-batch-invariance-and-eval-axioms
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/kv_pool.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/layerwise_kv_pool.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/KV_Cache_Pool_Guide.md
  source_digest: 544b0b89a70e3b44a3239bcb5ef449f1b5fe541ee4a95f68f1e063b0afdd2040
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: AscendStore multi-tier KV Cache Pool (Mooncake/Memcache/YuanRong) and layerwise
  KV transfer SOP
triggers:
  error_signatures:
  - AscendStoreConnector prefix lookup returns zero hits across nodes due to mismatched
    PYTHONHASHSEED
  - Mooncake KV Pool fails with SEGMENT_NOT_FOUND when SSD offload is enabled with
    default client_ttl
  - use_layerwise KV Pool fails when backend is not memcache or nr_hugepages is unconfigured
  keywords:
  - AscendStoreConnector
  - MooncakeStoreConnector
  - kv_pool
  - layerwise_kv_pool
  - use_layerwise
  - memcache
  - mooncake_master
  - yuanrong
  - kv_load_failure_policy
---
# AscendStore multi-tier KV Cache Pool (Mooncake/Memcache/YuanRong) and layerwise KV transfer SOP

## Scope and preconditions

Use this SOP when deploying or troubleshooting the `AscendStoreConnector` multi-tier KV Cache Pool (`mooncake`, `memcache`, or `yuanrong` backends in `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`) in either bulk or layerwise (`use_layerwise: true`) mode.

## Procedure

1. **Synchronize hashing and configure `AscendStoreConnector` (`docs/source/user_guide/feature_guide/kv_pool.md`):**
   - Export `PYTHONHASHSEED=0` identically on every node/container so `ChunkedTokenDatabase` generates identical prefix block keys across instances.
   - Configure `kv_connector: "AscendStoreConnector"` with `kv_role: "kv_both"` (PD-mixed) or combine it under `MultiConnector` with `MooncakeConnectorV1` on Prefill nodes in PD disaggregation.
   - Set top-level `"kv_load_failure_policy": "recompute"` (default is `"fail"`) on non-hybrid models so transient pool read misses fall back to local prefill recomputation instead of aborting requests.
2. **Verify backend-specific storage invariants (`mooncake_backend.py`, `memcache_backend.py`, `yuanrong_backend.py`):**
   - **Mooncake backend (`backend: "mooncake"`):** Set `protocol: "ascend"`, `device_name: ""`, and align `global_segment_size` to `1GB` multiples in `mooncake.json`. When enabling SSD offload (`enable_ssd_offload: true`, absolute non-symlink `ssd_offload_path`), launch `mooncake_master` with `--enable_offload=true --client_ttl=120` and `--default_kv_lease_ttl 11000` (larger than `ASCEND_CONNECT_TIMEOUT` / `ASCEND_TRANSFER_TIMEOUT`).
   - **Memcache backend (`backend: "memcache"`):** Required when `"use_layerwise": true`. Configure host hugepages (`echo 200000 > /proc/sys/vm/nr_hugepages`) and source `/usr/local/memcache_hybrid/set_env.sh` and `/usr/local/memfabric_hybrid/set_env.sh`.
3. **Pipeline layerwise KV save/load (`layerwise_kv_pool.md`, `attention_fence.py`):**
   - With `use_layerwise: true`, layer $i$ KV blocks are pushed immediately after layer $i$ attention and prefetched `layerwise_prefetch_layers` (default `1`) ahead of compute via `wait_for_layer_load`. Use `h2d_stagger_us` across TP ranks if concurrent H2D DMA saturates PCIe bandwidth.

## Verification

- Issue a warm-prefix request twice and confirm via scheduler logs that `get_num_new_matched_tokens` reports external pool hits and TTFT drops significantly without output divergence.
