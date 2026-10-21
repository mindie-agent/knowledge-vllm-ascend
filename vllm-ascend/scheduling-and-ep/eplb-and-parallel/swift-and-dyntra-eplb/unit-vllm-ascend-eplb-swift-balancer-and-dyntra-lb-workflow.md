---
category: workflow
conditions:
  component: eplb_dyntra_and_balance_scheduling
  framework: vllm-ascend
domain_path: vllm-ascend/scheduling-and-ep/eplb-and-parallel/swift-and-dyntra-eplb
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-eplb-swift-balancer-and-dyntra-lb-workflow
inherits_from:
- unit-vllm-ascend-hccl-and-dp-collective-invariants
- unit-vllm-ascend-patch-hierarchy-and-upstream-migration-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/core/dyntra_lb_scheduler.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/eplb/core/policy/policy_swift_balancer.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/eplb/core/policy/policy_flashlb.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/patch/platform/patch_balance_schedule.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/dyntra_lb.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/eplb_swift_balancer.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/balance_schedule_refactor.md
  source_digest: 882f3dafd94ce82107067ef83524c491ef6f8c2e06ea403c5a4c3f4ea28dd614
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/core/dyntra_lb_scheduler.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: EPLB (Default/SwiftBalancer/FlashLB), DyntraLB intra-decoder DP balancing,
  and Balance Scheduling workflow
triggers:
  error_signatures:
  - DyntraLB enabled on Prefill node or combined with enable_balance_scheduling causes
    startup failure
  - MoE EP all-to-all straggler rank stalls decode step when expert heat skews or
    DP request lengths diverge
  - Balance scheduling all_gather deadlocks when placed inside Scheduler.schedule
    instead of _has_global_unfinished_reqs
  keywords:
  - eplb
  - policy_swift_balancer
  - policy_flashlb
  - dyntra_lb
  - DyntraLBScheduler
  - dyntra_lb_config
  - enable_balance_scheduling
  - BalanceDPEngineCoreProc
---
# EPLB (Default/SwiftBalancer/FlashLB), DyntraLB intra-decoder DP balancing, and Balance Scheduling workflow

## Trigger and scope

Use this workflow when eliminating cross-rank straggler bubbles across Data Parallel (DP) and Expert Parallel (EP) ranks on `vllm-ascend` using:
1. **Expert Parallelism Load Balancer (`EPLB`):** Replicating and packing hot MoE experts (`policy_default_eplb.py`, `policy_swift_balancer.py`, `policy_flashlb.py`);
2. **DyntraLB (`DyntraLBScheduler`):** Intra-Decoder DP request KV-block load balancing in P/D disaggregation (`vllm_ascend/core/dyntra_lb_scheduler.py`, `dyntra_lb.md`);
3. **Balance Scheduling (`enable_balance_scheduling`):** Global admission freeze when any DP rank reaches `max_num_running_reqs` in PD-mixed mode (`patch_balance_schedule.py`, `balance_schedule_refactor.md`).

## Stage 1 — Select DyntraLB vs Balance Scheduling by PD topology (Mutually Exclusive)

1. **PD-Mixed deployments (`kv_role` is `kv_both` or non-disaggregated): use `enable_balance_scheduling`:**
   - Enable via `--additional-config '{"enable_balance_scheduling": true}'`.
   - `BalanceDPEngineCoreProc._has_global_unfinished_reqs` runs `balance_gather` (`dist.all_gather` of each rank's `len(self.running)`) on every active step wave, and freezes new `WAITING` admissions across all DP ranks whenever `max(running) == max_num_running_reqs`. Never combine `enable_balance_scheduling` with `dyntra_lb_config`.
2. **PD-Disaggregated Decode nodes (`kv_role == "kv_consumer"`, single-node `dp_size > 1`): use `DyntraLB`:**
   - Enable **only on Decode instances** via `--additional-config '{"scheduler_config": {"dyntra_lb_config": {"enabled": true, "mode": "dynamic"}}}'`. Never enable DyntraLB on Prefill nodes or combine it with `enable_balance_scheduling` or `profiling_chunk_config`.
   - DyntraLB pipelines decisions across two steps: all DP ranks all-gather running and admissible waiting KV-block counts, and the planner issues `In` (admit), `Out` (pause while keeping KV cache in HBM), and `Freeze` actions for the next step when imbalance exceeds `bubble_threshold` (default `5.0` KV blocks; `< 1.0` means normalized ratio `(max - avg) / max`) and a new request exceeds `long_req_block_threshold` (default `700` blocks).

## Stage 2 — Configure EPLB policy (`Default`, `SwiftBalancer`, `FlashLB`) for MoE expert skew

1. **Choose the EPLB policy (`vllm_ascend/eplb/core/policy/`):**
   - **Hierarchical vs Global Default EPLB (`policy_default_eplb.py`):** Uses group-limited hierarchical node packing when `num_nodes` divides `num_expert_groups` (ideal for prefill with smaller EP), and global replication/packing for large-EP decode.
   - **SwiftBalancer (`policy_swift_balancer.py`):** Minimizes the number of cross-device expert weight migrations per rebalance window—recommended on lower-interconnect-bandwidth A2 clusters.
   - **FlashLB (`policy_flashlb.py`):** Applies layer-wise fluctuation thresholds to skip rebalancing layers whose expert heat drift is below threshold.
2. **Coordinate dynamic EPLB with model loaders:**
   - Note that `RFork` weight transfer (`--load-format rfork`) is incompatible with `dynamic_eplb=true` because live expert placement mutates after seed registration.

## Stage 3 — Validate DP/EP bubble reduction

1. **Benchmark skewed prompt/output distributions:**
   - For DyntraLB, run a short validation with `mode: "static", start_step: 0, enable_diagnostics: true` to confirm `[dyntra_lb]` `In`/`Out`/`Freeze` plans apply cleanly across all Decode DP ranks, then disable `enable_diagnostics` and switch to `mode: "dynamic"` for production throughput measurement.
