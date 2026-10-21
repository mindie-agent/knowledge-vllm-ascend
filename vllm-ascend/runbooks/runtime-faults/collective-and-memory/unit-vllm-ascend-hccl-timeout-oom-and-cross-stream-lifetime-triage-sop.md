---
category: sop
conditions:
  framework: vllm-ascend
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  - Ascend950
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - vllm-ascend-idle-dp
domain_path: vllm-ascend/runbooks/runtime-faults/collective-and-memory
evidence_strength: software_contract
id: unit-vllm-ascend-hccl-timeout-oom-and-cross-stream-lifetime-triage-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-hccl-and-dp-collective-invariants
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/hccl_timeout_secondary_to_rank_failure_oom.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/collective_participation_per_rank_consistency.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/hccl_buffsize_too_small.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/moe_dispatch_combine_hccl_buffsize.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/moe_alltoallv_single_expert_missing_d2h_sync.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/camem_allocator_swallows_acl_oom.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/multistream_cross_stream_tensor_lifetime_clone.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/multistream_branch_variable_name_collision_hang.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/graph_mode_multi_stream_sync_failures.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/os_exit_tbe_routeserver_shutdown_noise.md
  source_digest: 78b60ef08b9981a9c5fd80f495a3cc9a07490b98a583e2db63e1a3dba341c964
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/runbooks/runtime/hccl_timeout_secondary_to_rank_failure_oom.md
  source_type: cannbot_runbook_distillation
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
tier: L2
title: 'Runtime fault triage Part 1: Secondary HCCL timeouts, rank-divergent collectives,
  HCCL_BUFFSIZE exhaustion, and cross-stream tensor lifetimes'
triggers:
  error_signatures:
  - HcclAllGather or Notify wait timeout (EI0006 / Get_Socket_Timeout) reported on
    healthy ranks after one peer rank fails with OOM
  - MoE dispatch/combine or multi-node collective fails with HCCL_BUFFSIZE too SMALL
    at large batch size
  - Single-expert AllToAllV crashes or hangs because non-blocking D2H split_sizes
    copy was read before stream sync
  - CaMem allocator swallows ACL_ERROR_RT_MEMORY_ALLOCATION and returns null pointer
    causing downstream illegal address crash
  - rogue process or thread unexpectedly creates NPU context on device 0 / rank 0
  - EngineCore shm_broadcast or ZeroMQ deserialization fails on 367-byte frame while /v1/models stays 200 and /v1/chat/completions hangs
  keywords:
  - hccl_timeout_secondary
  - Get_Socket_Timeout
  - HCCL_BUFFSIZE
  - collective_participation_per_rank
  - camem_allocator
  - ACL_ERROR_RT_MEMORY_ALLOCATION
  - multistream_cross_stream_tensor_lifetime
  - RouteServer
  - aclrtSetDevice
  - lock_page_killable
  - shm_broadcast
---
# Runtime fault triage Part 1: Secondary HCCL timeouts, rank-divergent collectives, HCCL_BUFFSIZE exhaustion, and cross-stream tensor lifetimes

## Applicability and symptom scope

Use this SOP to triage distributed hangs, HCCL watchdog timeouts, silent OOM crashes, and multi-stream memory/synchronization faults in `vllm-ascend`.

## Procedure

1. **Isolate primary rank failure (OOM / exception) from secondary `HcclAllGather` / `Get_Socket_Timeout` symptoms (`hccl_timeout_secondary_to_rank_failure_oom.md`, `bringup_error_root_cause_env_driver_platform.md`).**
   - When one rank throws an uncaught Python exception, NPU OOM (`EL0004`), or AICore fault, all surviving ranks block at their next collective (`AllReduce`, `AllGather`, `AllToAllV`) until `HCCL_EXEC_TIMEOUT` expires and logs `Get_Socket_Timeout` / `Notify wait timeout`.
   - **Rule:** Never diagnose the first rank that prints `HCCL_E_TIMEOUT` as a network failure; sort logs across all ranks by earliest timestamp to find the rank that exited or failed before entering the collective.
   - Check `camem` / custom C++ allocators (`camem_allocator_swallows_acl_oom.md`): ensure allocation wrappers throw immediately on `ACL_ERROR_RT_MEMORY_ALLOCATION` (`207001`) instead of returning a null pointer that surfaces later as an opaque device address fault.

2. **Enforce per-rank collective participation symmetry and explicit D2H barriers (`collective_participation_per_rank_consistency.md`, `moe_alltoallv_single_expert_missing_d2h_sync.md`).**
   - Never gate entry into a collective (`dist.all_reduce`, `dist.all_to_all_single`, MC2 `dispatch`/`combine`) on a rank-local predicate (such as `if local_num_tokens > 0:` or `if has_active_expert:`). Every rank in the process group must enter the collective in identical order, passing a dummy/zero-padded tensor when locally idle.
   - In `all_to_all_v` paths that copy `send_splits` / `recv_splits` from NPU to CPU via `non_blocking=True`, explicitly synchronize the copy stream (`event.synchronize()`) before reading the CPU list to launch `dist.all_to_all_single`.

3. **Size `HCCL_BUFFSIZE` identically across all ranks for MoE dispatch/combine (`hccl_buffsize_too_small.md`, `moe_dispatch_combine_hccl_buffsize.md`).**
   - When `npu_moe_distribute_dispatch_v2` / `combine_v2` or large-batch multi-node `AllGather` fails with `HCCL_BUFFSIZE too SMALL`, compute the required buffer ceiling from `max_num_batched_tokens * top_k * hidden_size * dtype_bytes` across the EP/TP domain and export an identical `HCCL_BUFFSIZE` (e.g., `512` to `2048` MiB) on every rank.

4. **Prevent multi-stream variable shadowing, cross-stream memory leaks, and shutdown noise (`multistream_branch_variable_name_collision_hang.md`, `multistream_cross_stream_tensor_lifetime_clone.md`, `graph_mode_multi_stream_sync_failures.md`, `os_exit_tbe_routeserver_shutdown_noise.md`).**
   - Never reuse the same Python local variable name across concurrent primary/secondary stream branches in the same function scope (which overwrites tensor references before the secondary stream's `record_stream` / `wait_tensor` executes).
   - When a small slice of a large primary-stream tensor is consumed by a long-running communication stream, `.clone()` the slice before passing it to the secondary stream so `record_stream` does not pin the entire parent allocation in HBM.
   - Call `RouteServer.finalize()` (or standard torch_npu cleanup) before `os._exit()` to prevent orphaned CANN TBE compiler subprocess tracebacks on shutdown.

5. **Use low-intrusion uprobes for rogue `device 0` contexts, separate NFS `D`-state stalls from NPU hangs, and preserve IPC frame / profiler step invariants:**
   - **Zero-Intrusion `aclrtSetDevice` Uprobe:** When an unknown process or worker thread intermittently creates a context on `device 0` and restarting vLLM or modifying Python packages is undesirable, attach a user-space uprobe on `libascendcl.so` (`perf probe -x libascendcl.so 'aclrtSetDevice deviceId=%x0:s32'` with `perf record -e probe_libascendcl:aclrtSetDevice -aR -g --call-graph dwarf,16384`) to capture PID, thread name, `deviceId`, and call stack with zero service interruption.
   - **Slow Shared Storage (`D`-State) vs. Real NPU Hang:** Before killing a slow multi-node startup as an HCCL deadlock, inspect `/proc/<pid>/stat` (`D` state), `/proc/<pid>/wchan` (`lock_page` / `lock_page_killable`), `/proc/<pid>/io` `read_bytes` growth, and `nfsiostat` to distinguish cold NFS page-cache reads (`~28 MB/s`) from NPU hangs.
   - **EngineCore `shm_broadcast` / ZeroMQ Corrupted Frame Forensics:** If worker output deserialization fails intermittently (e.g., on a `367 B` empty-batch frame) leaving `/v1/models` returning `200` while `/v1/chat/completions` hangs, never swallow the exception or substitute `EMPTY_MODEL_RUNNER_OUTPUT` (which silently desynchronizes request-batch ordering); log frame count, byte length, error offset, SHA-256 digest, and exception chain, and let the exception propagate.
   - **Profiling Layer Segmentation Invariant:** When segmenting NPU profiler traces on MLA + MTP + Hybrid Dense/MoE models, never treat internal MLA sub-kernels (`MlaPrologV3`, `KvQuantSparseFlashAttention`) as layer-start anchors, and never split short dense prefixes (`3 dense + 75 MoE`) or trailing `MTP` speculative layers into separate steps.
