---
category: sop
conditions:
  component: cpu_binding
  framework: vllm-ascend
domain_path: vllm-ascend/architecture/platform-and-patch/cpu-core-binding
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-numa-cpu-binding-and-irq-isolation-sop
inherits_from:
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/cpu_binding.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/cpu_binding.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/cpu_binding.md
  source_digest: 0e2cf7fd3016804836d61a67d7145f89f94ee309a68411b17345a2ca6cf54b7c
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/cpu_binding.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: NUMA-aware CPU core binding, IRQ core isolation, and ACL thread affinity SOP
triggers:
  error_signatures:
  - decode TPOT exhibits periodic 10-50ms jitter due to cross-NUMA thread migration
    or NIC IRQ contention
  - enable_cpu_binding fails or falls back when lscpu or npu-smi topo affinity is
    missing in container
  keywords:
  - enable_cpu_binding
  - cpu_binding
  - bind_cpus
  - DeviceInfo
  - NUMA
  - IRQ
  - OMP_PROC_BIND
  - OMP_NUM_THREADS
  - jemalloc
---
# NUMA-aware CPU core binding, IRQ core isolation, and ACL thread affinity SOP

## Scope and preconditions

Use this SOP when configuring or troubleshooting host CPU core binding (`additional_config: {"enable_cpu_binding": true}`, `vllm_ascend/cpu_binding.py`) on multi-NPU Atlas A2/A3 servers to eliminate host launch jitter and cross-NUMA memory access penalties.

## Procedure

1. **Understand the three CPU allocation modes in `vllm_ascend/cpu_binding.py`:**
   - `DeviceInfo` inspects visible NPUs (`ASCEND_RT_VISIBLE_DEVICES`), `npu-smi info -t topo`, `lscpu`, and `/proc/interrupts` (`sq_send_trigger` NPU/HCCS IRQs).
   - **NUMA-balanced / affinity mode:** Maps each NPU rank to its PCIe/HCCS-local NUMA node, reserves the lowest cores on each NUMA node for hardware IRQ handlers (`sq_send_trigger`), and partitions the remaining NUMA-local cores evenly among the NPU ranks sharing that NUMA node.
   - Within each NPU rank's assigned CPU pool, `bind_cpus()` assigns the main Python/PyTorch worker thread to the primary slice and pins `acl_thread` / `release_thread` child threads to dedicated sibling cores so ACL task submission never contends with Python input preparation.
2. **Configure host environment variables alongside `enable_cpu_binding`:**
   - Set `export OMP_PROC_BIND=false` and `export OMP_NUM_THREADS=1` (or `10` when OpenMP host kernels are used without core conflict) so OpenMP runtime initialization does not override `sched_setaffinity` masks applied by `bind_cpus()`.
   - Preload `libjemalloc.so.2` via `LD_PRELOAD` and set `sysctl -w kernel.numa_balancing=0` and `vm.swappiness=0` on A2/A3 hosts to prevent Linux auto-NUMA page migration stalls.
3. **Verify container mounts required by topology discovery:**
   - Ensure `npu-smi` and `/usr/local/dcmi` are mounted into the container; if `npu-smi info -t topo` is unavailable, `cpu_binding.py` logs a fallback to uniform core slicing across the allowed `Cpus_allowed_list` mask.

## Verification

- Inspect worker startup logs for the `bind_cpus` allocation table (`main` vs `acl_thread` core ranges per rank) and verify via `taskset -cp <worker_pid>` that no two NPU ranks on the same node overlap their main-thread CPU sets.
