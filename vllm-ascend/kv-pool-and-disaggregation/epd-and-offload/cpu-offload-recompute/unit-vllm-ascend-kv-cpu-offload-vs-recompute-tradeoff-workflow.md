---
category: workflow
conditions:
  component: kv_cpu_offload_and_recompute
  framework: vllm-ascend
domain_path: vllm-ascend/kv-pool-and-disaggregation/epd-and-offload/cpu-offload-recompute
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-kv-cpu-offload-vs-recompute-tradeoff-workflow
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/recompute_cpu_offload_connector.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/core/recompute_scheduler.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/simple/simple_cpu_offload_connector.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/recompute_cpu_offload.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/kv_cache_cpu_offload.md
  source_digest: 0202f4859994a1d02a262932992c755ac2ad28e9f1c6c468461edc49aa8d3591
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/recompute_cpu_offload_connector.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Decode RecomputeCPUOffloadConnector vs Native/Simple KV CPU Offload workflow
  and shm sizing
triggers:
  error_signatures:
  - PD-disaggregated Decode node preempts long request and stalls trying to recompute
    prompt under small max_num_batched_tokens
  - RecomputeCPUOffloadConnector OOMs or hangs during container startup due to insufficient
    Docker --shm-size
  - recompute_scheduler_enable enabled in PD-mixed kv_both mode causes startup or
    scheduling error
  keywords:
  - RecomputeCPUOffloadConnector
  - RecomputeScheduler
  - recompute_scheduler_enable
  - cpu_bytes_to_use_per_rank
  - SimpleCPUOffloadConnector
  - OffloadingConnector
  - kv_cache_cpu_offload
  - recompute_cpu_offload
---
# Decode RecomputeCPUOffloadConnector vs Native/Simple KV CPU Offload workflow and shm sizing

## Trigger and scope

Use this workflow when selecting and configuring host CPU DRAM KV cache offloading on `vllm-ascend`: either preserving preempted Decode requests in P/D disaggregation (`RecomputeCPUOffloadConnector` + `RecomputeScheduler`, `recompute_cpu_offload.md`) or expanding local prefix-cache capacity via `OffloadingConnector` / `SimpleCPUOffloadConnector` (`kv_cache_cpu_offload.md`).

## Stage 1 — Distinguish Preemption Preservation vs Prefix-Cache CPU Offloading

1. **Choose the right CPU offload connector for the deployment topology:**
   - **P/D-Disaggregated Decode Preemption Preservation (`RecomputeCPUOffloadConnector`):** On Decode nodes (`kv_role: "kv_consumer"`), `max_num_batched_tokens` is sized for decode (`max_num_seqs * (1 + num_spec_tokens)`), which cannot efficiently recompute a 32K–128K prompt if HBM KV blocks run out. `RecomputeScheduler` + `RecomputeCPUOffloadConnector` saves preempted requests' KV blocks to pinned CPU DRAM before reusing their HBM blocks and restores them via H2D DMA when rescheduled, avoiding kicking the request back to Prefill.
   - **General Prefix-Cache CPU Offloading (`OffloadingConnector` / `SimpleCPUOffloadConnector` in `kv_offload/`):** Used in PD-mixed (`kv_both`) or Prefill deployments to swap evicted prefix blocks to host DRAM and reload them on prefix hit.
2. **Never enable `recompute_scheduler_enable` in PD-mixed mode:**
   - `--additional-config '{"scheduler_config":{"recompute_scheduler_enable":true}}'` is valid **only** when `kv_role` is `kv_producer` or `kv_consumer`.

## Stage 2 — Size `cpu_bytes_to_use_per_rank` and Docker `--shm-size`

1. **Prefer `cpu_bytes_to_use_per_rank` over `cpu_bytes_to_use` (`recompute_cpu_offload.md`):**
   - `cpu_bytes_to_use` is divided by `world_size` (which in DP+TP deployments divides across DP ranks and easily under-allocates per card). Always specify `cpu_bytes_to_use_per_rank` explicitly (e.g., `17179869184` for 16 GiB per NPU rank).
2. **Size Docker `--shm-size` to cover total pinned host allocations:**
   - Pinned CPU KV pools allocate from shared memory. On 16-card Atlas A3 containers with 16–32 GiB per rank, set `docker run --shm-size=512g` or `--shm-size=1024g`; a 1 GiB default `--shm-size` causes startup OOM or silent hang during pinned buffer allocation.

## Stage 3 — Wire `MultiConnector` on P/D Decode nodes and verify preemption recovery

1. **Chain `MooncakeConnectorV1` and `RecomputeCPUOffloadConnector`:**
   - Configure top-level `"kv_connector": "MultiConnector"` (`"kv_role": "kv_consumer"`) with child connectors `[MooncakeConnectorV1, RecomputeCPUOffloadConnector]`. Place any `"kv_load_failure_policy"` on the top-level `MultiConnector` config.
   - Drive Decode concurrency above HBM KV capacity to force preemption and verify that preempted requests resume locally from CPU DRAM without returning to the Prefill instance.
