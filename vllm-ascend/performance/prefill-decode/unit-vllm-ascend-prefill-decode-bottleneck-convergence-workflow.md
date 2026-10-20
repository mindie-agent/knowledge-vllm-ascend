---
category: workflow
conditions:
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  workload_phase:
  - prefill_decode_mixed
  - decode_heavy
  - prefill_heavy
domain_path: vllm-ascend/performance/prefill-decode
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-prefill-decode-bottleneck-convergence-workflow
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-hccl-and-dp-collective-invariants
- unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py
  source_digest: 61ee8bc56d4ac0e1a7f43d1804bfe41c1a6699181774938907c97174f2348343
  source_locator: vllm-project/vllm-ascend@d8c9e3a673af918c94a7794ef6f1f69c024d9c62:vllm_ascend/worker/model_runner_v1.py
  source_type: hermes_workflow_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: 'Hermes workflow: vllm-ascend TTFT vs TPOT/ITL bottleneck attribution and convergence'
triggers:
  error_signatures:
  - TPOT or ITL regresses under decode concurrency while single-operator latency is unchanged
  - TTFT spikes during chunked prefill or multi-rank DP/TP execution
  - NPU stream timeline shows gaps between consecutive decode steps
  keywords:
  - TTFT
  - TPOT
  - ITL
  - prefill
  - decode
  - host bubble
  - ACLGraph
  - HCCL
  - KV cache
  - bottleneck attribution
  - workflow
---

# Hermes workflow: vllm-ascend TTFT vs TPOT/ITL bottleneck attribution and convergence

## Objective & Entry Criteria

Use this staged workflow when optimizing or diagnosing end-to-end serving throughput (`output_tokens/s`), TTFT, or TPOT/ITL on `vllm-ascend` so single-operator changes are never pursued before isolating whether the critical path is host-dispatch bound, device-compute/memory bound, or collective-communication bound.

## Stage 1 — Decompose Wall-Clock Step Time into Host Dispatch, Device Kernel, and Collective Wait

1. Capture a bounded steady-state window (excluding warmup and graph capture steps) and partition per-step wall-clock time into:
   - `T_host_prep`: scheduler + `model_runner_v1` input preparation + metadata copy before launch,
   - `T_device_compute`: attention (`AscendMetadata` / paged attention / MLA / GDN) + MLP/MoE GEMM execution on NPU streams,
   - `T_collective`: HCCL `all_reduce` / `all_to_all` / `coordinate_batch_across_dp` span and cross-rank arrival skew,
   - `T_stream_gap`: uncovered device idle gaps between kernel launches.

## Stage 2 — Branch by Discriminating Observation

- **Branch A (`T_stream_gap` > 15% of decode step time or frequent eager fallback):**
  - Do **not** tune individual kernel tiling first.
  - Apply `unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop`: inspect ACLGraph bucket hit rate, eliminate `.item()` / `.cpu()` host syncs in `prepare_inputs`, and verify async output processing overlap.
- **Branch B (`T_collective` dominates or diverges across ranks):**
  - Check whether collective time is true bandwidth saturation or arrival imbalance caused by uneven prefill/decode token distribution or missing idle-rank dummy participation (`unit-vllm-ascend-hccl-and-dp-collective-invariants`).
  - Measure per-rank compute duration before the collective barrier; only tune HCCL/comm overlap after confirming ranks enter the collective simultaneously.
- **Branch C (`T_device_compute` dominates with high stream coverage):**
  - Rank the top-3 device kernels by share of `T_device_compute` (`p` in Amdahl's law $S_{\text{est}} = \frac{1}{(1 - p) + p / s}$).
  - Verify `active_execution_proof` (the target operator variant is actually invoked on the active decode/prefill shape and layout) before launching L4 operator mutations.

## Stage 3 — Dual-Baseline Verification

1. Verify every candidate against both the **initial valid baseline** and the **immediately preceding accepted state** on the same request trace / shape distribution.
2. Require zero correctness regression (`unit-vllm-ascend-verification-and-state-parity-gates`) and report TTFT, TPOT/ITL, and `output_tokens/s` separately.
