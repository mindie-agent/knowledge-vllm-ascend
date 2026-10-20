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
domain_path: vllm-ascend/optimizations/kv-cache-and-pd-disaggregation/pd-transfer-sync
evidence_strength: software_contract
id: unit-vllm-ascend-pd-hixl-mooncake-layerwise-kv-push-and-release-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-hccl-and-dp-collective-invariants
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/pd_layerwise_kv_push_event_sync.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/pd_prefill_kv_block_release_vs_decode_pull.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/pd_kv_transfer_timeout_from_hccl_rdma_budget.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/hixl_hccs_ipc_2mib_alignment.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/mooncake_ascend_hixl_kv_transfer_bringup.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/vllm_ascend_mooncake_connector_backend_incompat.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/experience/interventions/yuanrong_kv_pool_remote_h2d_ascend_prereqs.md
  source_digest: baa7c0f4baf8c737ecf2f20b923397cd495e2e1316fdc3c28f51fd0ba0f95581
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/pd_layerwise_kv_push_event_sync.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Prefill-Decode (PD) disaggregation KV transfer: 2 MiB HIXL alignment, layerwise
  Event push, ACK-gated block release, and RDMA timeout budgeting'
triggers:
  error_signatures:
  - HIXL or Mooncake protocol=ascend registration fails when KV cache tensor base
    address is not 2 MiB aligned
  - Decode instance reads corrupted KV blocks because Prefill freed blocks before
    all Decode TP/CP ranks completed pull
  - Layerwise KV push stalls Prefill forward when synchronizing the entire model stream
    instead of per-layer npu.Event
  keywords:
  - pd_layerwise_kv_push_event_sync
  - pd_prefill_kv_block_release
  - hixl_hccs_ipc_2mib_alignment
  - mooncake_ascend_hixl
  - HCCL_RDMA_TIMEOUT
  - HCCL_RDMA_RETRY_CNT
  - kv_transfer
---
# Prefill-Decode (PD) disaggregation KV transfer: 2 MiB HIXL alignment, layerwise Event push, ACK-gated block release, and RDMA timeout budgeting

## Applicability and symptom scope

Use this SOP when bringing up or optimizing Prefill-Decode (PD) disaggregated KV cache transfer over HIXL, Mooncake (`protocol="ascend"`), HCCL P2P, or Yuanrong KV-Pool in `vllm-ascend`.

## Procedure

1. **Enforce 2 MiB physical HBM address alignment and Mooncake/HIXL prerequisites (`hixl_hccs_ipc_2mib_alignment.md`, `mooncake_ascend_hixl_kv_transfer_bringup.md`, `vllm_ascend_mooncake_connector_backend_incompat.md`).**
   - HIXL HCCS IPC and RoCE RDMA zero-copy memory registration require every registered KV buffer base pointer to be aligned to `2 MiB` (`2 * 1024 * 1024` bytes). Allocate a raw `uint8` buffer padded by `2 MiB`, slice from `offset = (-base_ptr) % (2 * 1024 * 1024)`, and `.view(dtype).reshape(shape)`.
   - When using Mooncake with `protocol="ascend"`, verify that `ascend_transport` shared libraries are on `LD_LIBRARY_PATH`, explicit `device_id` binding precedes `TransferEngine.initialize()`, and heterogeneous TP/CP page layouts use connector v1 compatible block indexing.
   - When enabling Yuanrong KV-Pool Remote H2D direct transfer (`yuanrong_kv_pool_remote_h2d_ascend_prereqs.md`), verify host 2 MiB HugeTLB allocation (`vm.nr_hugepages`) and RoCE link readiness before enabling remote H2D flags.

2. **Synchronize layerwise KV push with per-layer `torch.npu.Event`, never `stream.synchronize()` (`pd_layerwise_kv_push_event_sync.md`).**
   - In layerwise KV push, calling `torch.npu.current_stream().synchronize()` after each layer blocks host operator dispatch for layer $L+1$ until layer $L$ completes on the NPU.
   - Record a lightweight `event = torch.npu.Event()` on the compute stream immediately after layer $L$'s KV scatter and have the background KV-transfer stream wait on `event` (`transfer_stream.wait_event(event)` or `event.synchronize()` on the dedicated transfer worker thread) so compute dispatch for layers $L+1 \dots N$ continues uninterrupted.

3. **Gate Prefill KV block reclamation on Decode completion ACKs (`pd_prefill_kv_block_release_vs_decode_pull.md`).**
   - Under pull-mode KV transfer (where Decode ranks pull KV blocks asynchronously after Prefill finishes), freeing Prefill KV blocks immediately at the end of the Prefill forward pass allows the next Prefill batch to overwrite those HBM pages while Decode RDMA reads are still in flight.
   - Retain Prefill KV blocks in a `transferring` reference-counted state until all participating Decode TP/DP ranks return pull-completion ACKs (or switch to push-mode where the Prefill transfer thread frees blocks only after `batch_transfer_sync` returns).

4. **Derive KV-transfer timeout from HCCL RDMA retry arithmetic (`pd_kv_transfer_timeout_from_hccl_rdma_budget.md`).**
   - Never hardcode a short KV-transfer wait timeout below the transport layer's retry window. Enforce:
     $$\text{kv\_transfer\_timeout\_s} \ge \text{HCCL\_RDMA\_TIMEOUT} \times (\text{HCCL\_RDMA\_RETRY\_CNT} + 1) + \text{queuing\_grace\_s}$$
     and set Gloo CPU control-plane variables (`GLOO_IGNORE_IPV6=1`, `GLOO_SOCKET_IFNAME=<nic>`) consistently across all instances (`online_gloo_cpu_comm_domain_env_bringup.md`, `multi_instance_online_master_addr.md`).
