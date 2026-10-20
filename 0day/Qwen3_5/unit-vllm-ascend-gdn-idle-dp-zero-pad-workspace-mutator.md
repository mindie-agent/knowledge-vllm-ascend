---
schema: mindie-graded-unit/1
id: unit-vllm-ascend-gdn-idle-dp-zero-pad-workspace-mutator
revision: 1
tier: L4
category: mutator
domain_path: 0day/Qwen3_5
evidence_strength: hypothesis
title: 'L4 mutator: GDN idle-DP zero-padded workspace with empty-batch HCCL participation'
conditions:
  model_family:
  - qwen3_5
  - Qwen3.5
  - gated_deltanet
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  dp: 2
  idle_rank_state: no_scheduled_requests
  active_batch: 0
  graph_capture_shape: dynamic
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
- unit-vllm-ascend-idle-dp-gdn-state
provenance:
  source_type: ablation_design_receipt
  source_locator: docs/evidence/l4-gdn-idle-dp-zero-pad-ablation.json
  source_digest: 5a2f8780ca357258b4f53d3c4a42e37df4386a328a22d4d7a99afec5929b3c08
  evidence_refs:
  - docs/evidence/l4-gdn-idle-dp-zero-pad-ablation.json
  - https://github.com/vllm-project/vllm-ascend/pull/15655
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py
  experiment_status: not_run_source_only
  supersedes: []
triggers:
  error_signatures:
  - idle DP rank with an empty batch hangs because the zero-token step skipped coordinate_batch_across_dp
  - the idle rank workspace shape differs from the captured active shape and replay reads stale buffers
  - a zero-token dummy run changes one row of conv_state or ssm_state
  - an active batch is routed through the idle zero-pad path and loses its real tokens
  - a static graph capture is forced through a dynamic zero-pad replay and fails or writes the wrong region
  keywords:
  - Qwen3.5
  - qwen3_5
  - GatedDeltaNet
  - idle DP
  - zero-pad
  - empty batch
  - HCCL
  - coordinate_batch_across_dp
  - conv_state
  - ssm_state
  - graph capture
  - workspace
  - leave-one-out
  - guard workload
---
# L4 mutator: GDN idle-DP zero-padded workspace with empty-batch HCCL participation

## Applicability

`vllm-ascend` Qwen3.5 GatedDeltaNet (GDN) decode on Ascend 910B/910_93, `data_parallel_size: 2`, one idle rank with no scheduled requests and an empty active batch. Graph capture uses a dynamic shape that can include the zero-padded workspace. The observed facts must include `idle_rank_state: no_scheduled_requests`, `active_batch: 0` and `graph_capture_shape: dynamic`; a missing fact is `condition_unknown` and the unit is excluded by default.

Do not apply when the local batch is non-empty or when graph capture is static. Those are different execution paths, not a tuning choice.

## Signals

- The idle rank waits at `coordinate_batch_across_dp` or the peer blocks until HCCL timeout; rank order or world size differs between the idle and active steps.
- `npu-smi`/logs show an idle rank with no requests while the peer is active, or the zero-token step allocates inside the captured region.
- A per-row `conv_state` or `ssm_state` checksum changes after a zero-token dummy step, or later requests on the formerly idle rank diverge.
- The workspace high-water mark grows with every idle step, or replay reads a shape different from the captured shape.
- A candidate removes the HCCL timeout by taking the wrong path or by muting GDN kernels; that is not a fix.

## Transformation

Keep the guard and fallback branches in every arm. Apply the factors in this order.

1. **`zero_pad_idle_workspace`** — allocate or select a workspace padded to the captured maximum shape, zero the unused prefix and tail deterministically, and bind the zero-token metadata (`num_actual_tokens=0`, zero `gdn_query_start_loc`, `max_query_len=0`, all-false `is_prefilling`) to that exact workspace identity. Do not reuse a previous request's tokens, KV blocks or recurrent rows.
2. **`empty_batch_collective_keepalive`** — keep the idle rank in `coordinate_batch_across_dp` with the same rank order, world size, collective sequence and bounded timeout as the active peer. The zero-token step must enter and leave the collective; an early return is the failure being fixed.
3. **`recurrent_state_write_guard`** — snapshot per-row `conv_state`/`ssm_state` before the idle step, skip the GDN state update for the zero-token batch, and prove the rows are unchanged afterwards. A scheduled batch must still advance state and match the eager/fp32 rows. Never pass the skip flag on a real batch.
4. **`dynamic_graph_shape_reuse`** — capture and replay the zero-padded dynamic shape with stable addresses and no allocation inside the captured region. If capture must re-shape, update the workspace identity together with the graph; a static shape is out of scope.

The factors are separable for leave-one-out. Remove `zero_pad_idle_workspace` and the arm must expose a shape or state write; remove `empty_batch_collective_keepalive` and it must reproduce the rank asymmetry or HCCL timeout; remove `recurrent_state_write_guard` and it must fail per-row state invariance; remove `dynamic_graph_shape_reuse` to separate graph-shape reuse from padding. Keep the empty-batch and state-write correctness prerequisites in every arm.

## Fallback boundary

- **Active batch > 0.** Use the scheduled decode path with normal token metadata and the normal GDN state update. The zero-pad mutator must not claim or truncate active tokens.
- **Static graph capture shape.** Use the declared static-shape path or re-capture under the new shape. Do not force a dynamic zero-pad replay into a static capture; report the shape mismatch instead.
- **HCCL or device precondition unavailable.** If the collective cannot be established, report the execution-condition failure and stop. Do not fake participation, suppress the timeout, or disable GDN kernels to make the run exit.
- **State parity unproven.** If per-row checksums are unavailable or a mocked metadata test is the only evidence, do not promote this unit to `reproduced_empirical`.

## Verification and ablation

- **Gate 1 — metadata zero-effect (no NPU).** The builder must emit zero-token GDN metadata and a zero-padded workspace identity; a real batch must default `skip_gdn_state_update=False` and keep its tokens.
- **Gate 2 — state parity (one rank).** Snapshot per-row `conv_state`/`ssm_state`. A pre-fix zero-token step must move state by a synthetic one-token delta; the fixed path must leave both unchanged. A real decode must advance state and match the fp32 logits/state.
- **Gate 3 — collective symmetry (DP2 TP1).** All ranks enter and leave `coordinate_batch_across_dp` and exit 0 under a bounded timeout. The pre-fix negative control must fail the idle-state assertion.
- **Gate 4 — graph replay and guards.** Replay the same inputs and compare eager logits and per-row state deterministically. Measure every guard: idle zero-token step, active peer scheduled batch, both row checksums, collective enter/exit, graph capture/replay parity, and warm-server repeated idle steps.
- **Gate 5 — hidden retention.** Freeze hidden inputs and classify each as `both_pass`, `opt_regression`, `both_fail`, `opt_improvement` or `evaluation_error`. Only comparable `both_pass` items enter a speedup ratio; a guard regression rejects system acceptance even when the main case is faster.
- **Provenance.** This unit is a design-level hypothesis. The linked receipt records factors, arms, guards and fallback boundaries with `experiment_status: not_run_source_only`; it establishes no measured speedup and no full-serving result. Promote only after a real DP2 run, per-row state checks and the final retained combination are independently verified.
