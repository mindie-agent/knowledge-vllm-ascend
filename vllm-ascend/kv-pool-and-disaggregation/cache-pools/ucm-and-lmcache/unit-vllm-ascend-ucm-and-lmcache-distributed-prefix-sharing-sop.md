---
category: sop
conditions:
  component: ucm_and_lmcache
  framework: vllm-ascend
domain_path: vllm-ascend/kv-pool-and-disaggregation/cache-pools/ucm-and-lmcache
evidence_strength: software_contract
id: unit-vllm-ascend-ucm-and-lmcache-distributed-prefix-sharing-sop
inherits_from:
- unit-vllm-ascend-hccl-and-dp-collective-invariants
- unit-vllm-ascend-deterministic-batch-invariance-and-eval-axioms
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/distributed/kv_transfer/kv_pool/ucm_connector/connector.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/ucm_deployment.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/lmcache_ascend_deployment.md
  source_digest: 47df19560cfece25ec93c283226d31b95b29b5e6643057ba1893692a33de995f
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/distributed/kv_transfer/kv_pool/ucm_connector/connector.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Unified Cache Manager (UCM PipelineStore) and LMCache-Ascend distributed prefix
  caching SOP
triggers:
  error_signatures:
  - UCMConnector fails to share KV blocks across centralized PD nodes when storage_backends
    is local-only
  - LMCacheAscendConnector registration error after vllm-ascend removed built-in LMCache
    wrapper
  keywords:
  - UCMConnector
  - UcmPipelineStore
  - ucm_deployment
  - LMCache
  - LMCache-Ascend
  - store_pipeline
  - Cache|Posix
  - 3FS
---
# Unified Cache Manager (UCM PipelineStore) and LMCache-Ascend distributed prefix caching SOP

## Scope and preconditions

Use this SOP when deploying external 3-tier KV cache storage (`HBM -> Host DRAM -> Shared SSD/NFS/3FS`) via `UCMConnector` (`vllm_ascend/distributed/kv_transfer/kv_pool/ucm_connector/connector.py`, `docs/source/user_guide/feature_guide/ucm_deployment.md`) or community `LMCache-Ascend` (`docs/source/user_guide/feature_guide/lmcache_ascend_deployment.md`).

## Procedure

1. **Configure UCM 3-tier `UcmPipelineStore` (`docs/source/user_guide/feature_guide/ucm_deployment.md`):**
   - Define `ucm_connectors` in the YAML referenced by `"UCM_CONFIG_FILE"`:
     ```yaml
     ucm_connectors:
       - ucm_connector_name: "UcmPipelineStore"
         ucm_connector_config:
           store_pipeline: "Cache|Posix"
           storage_backends: "/mnt/shared_kv_store"
           cache_buffer_capacity_gb: 64
     enable_event_sync: true
     use_layerwise: false
     ```
   - `store_pipeline: "Cache|Posix"` chains host DRAM (`Cache` tier) with shared POSIX/NFS/3FS persistent storage (`Posix` tier).
   - For Centralized PD Disaggregation (`kv_role: "kv_both"` on both Prefill and Decode instances), `storage_backends` **must** point to a shared filesystem (NFS or 3FS) mounted identically across all Prefill and Decode nodes, and `PYTHONHASHSEED` must be identical across all instances.
2. **Combine UCM with P2P Mooncake in Distributed PD mode:**
   - For lower-latency homogeneous P/D clusters, pair `MooncakeConnectorV1` (direct NPU-to-NPU P2P KV transfer) with `UCMConnector` (persistent prefix cache lookup/offload) so Prefill nodes skip recomputing cached prefixes while streaming only delta KV blocks to Decode.
3. **Use external `LMCache-Ascend` plugin registration (`lmcache_ascend_deployment.md`):**
   - `LMCacheAscendConnector` is no longer registered inside `vllm_ascend` directly; install `lmcache==0.3.12` (`NO_CUDA_EXT=1`) and `LMCache-Ascend` (`pip install --no-build-isolation -e .` with submodules) and pass the connector class provided by the `LMCache-Ascend` package with `--cap-add=SYS_RESOURCE --cap-add=IPC_LOCK` in Docker.

## Verification

- Restart a `vllm serve` instance backed by `UCMConnector` (`Cache|Posix`), resend a previously cached long prompt, and verify that persistent Tier-3 storage serves the prefix cache hit across the process restart.
