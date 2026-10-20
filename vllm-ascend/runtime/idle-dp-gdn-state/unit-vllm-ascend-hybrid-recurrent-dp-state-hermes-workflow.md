---
category: workflow
conditions:
  affected_state:
    exact_set:
    - conv_state
    - ssm_state
  dp: 2
  idle_rank_state: no_scheduled_requests
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - vllm-ascend-idle-dp
domain_path: vllm-ascend/runtime/idle-dp-gdn-state
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow
inherits_from:
- unit-vllm-ascend-hccl-and-dp-collective-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
- unit-vllm-ascend-idle-dp-gdn-state
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/pull/15655
  - https://github.com/vllm-project/vllm-ascend/pull/15510
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/tests/ut/worker/test_model_runner_v1.py
  source_digest: 61ee8bc56d4ac0e1a7f43d1804bfe41c1a6699181774938907c97174f2348343
  source_locator: vllm-project/vllm-ascend@d8c9e3a673af918c94a7794ef6f1f69c024d9c62:vllm_ascend/worker/model_runner_v1.py
  source_type: hermes_workflow_distillation
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
tier: L3
title: 'Hybrid GDN/Mamba + MLA DP idle-rank state troubleshooting workflow'
triggers:
  error_signatures:
  - idle DP rank with no scheduled requests while peer is active
  - cached GDN conv_state or ssm_state overwritten after an idle dummy step
  - later requests on the formerly idle rank diverge
  - two-rank DP run hangs in coordinate_batch_across_dp
  keywords:
  - idle DP
  - GDN
  - conv_state
  - ssm_state
  - workflow
  - skip_gdn_state_update
  - HCCL
  - state checksum
  - DP2 TP1
  - Ascend910_93
  - reviewed_change
---

# Hybrid GDN/Mamba + MLA DP idle-rank state troubleshooting workflow

## Trigger

Hybrid GatedDeltaNet/Mamba + MLA model, `dp > 1`, one DP rank idle while a peer is active; the idle rank's cached `conv_state`/`ssm_state` changes and later requests diverge. Not for pure attention, small real batches, or checkpoint/logit parity.

## First discriminating observation

Run the no-NPU metadata probe first: does the zero-token branch call `_dummy_run(1, skip_gdn_state_update=True)` and build GDN metadata with `num_actual_tokens=0` and a zero start-loc? This splits "collective not entered" from "collective entered but state written".

## Branches

- **A. Dummy call missing, flag false, or no DP collective:** HCCL participation defect. Fix the zero-token branch before any checksum. Exit only when all ranks enter and leave the collective under a bounded timeout.
- **B. Metadata correct but idle state changes:** the masked kernel still writes state. Run the per-row one-rank checksum on the pre-fix and fixed paths, then the two-rank DP2 TP1 gate. Accept only when idle state is unchanged and active state advances.
- **C. Idle preserved but active state or logits diverge:** do not claim success. Compare with the `fp32` reference and inspect `query_start_loc`/`num_accepted_tokens` reuse.
- **D. All pass but a repeat run is unstable:** treat determinism and negative-control failure as separate gates before distilling.

## Prerequisites

Record revision, CANN/torch stack, NPU/die count, `dp`/`tp`/`pp`, backend, graph mode, model/checkpoint/dtype/cache, the exact schedule and a pre-registered tolerance. A missing fact is `condition_unknown`, not a pass.

## Exploration exit

Stop when the recorded topology passes: all ranks exit `0`; per-row idle `conv_state`/`ssm_state` unchanged; active state advances; logits match reference; the pre-fix control fails. Then distill into a `reviewed_change`; a passing probe is not itself a reviewed change.

## Pitfalls

A whole-cache hash hides the changed row; reusing the idle zero start-loc for real decode; passing the skip flag on a scheduled batch; declaring success from the metadata probe alone; copying the DP2 TP1 receipt to another topology.
