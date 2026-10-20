---
category: workflow
conditions:
  affected_state:
    exact_set:
    - conv_state
    - ssm_state
  dp: 2
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
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
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
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: 'Hermes workflow: hybrid GDN/Mamba + MLA DP idle-rank state correction'
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
  - Hermes
  - workflow
  - skip_gdn_state_update
  - HCCL
  - state checksum
  - DP2 TP1
  - Ascend910_93
  - reviewed_change
---

# Hermes workflow: hybrid GDN/Mamba + MLA DP idle-rank state correction

## When to use

Use this workflow when a hybrid GatedDeltaNet/Mamba + MLA model runs with `dp > 1` and one DP rank has no scheduled requests while a peer is active. The observable failure is that the idle rank's cached `conv_state` or `ssm_state` changes across the idle step, and a later request on that rank diverges. The workflow diagnoses the metadata path, proves state invariance and active-rank update parity on a real topology, and distills only the verified result into a `reviewed_change`.

Do not use it for a pure-attention model, for a real batch that happens to be small, or as a substitute for checkpoint/logit parity. Those are different probes.

## Preconditions and inputs

Record before starting:

- repository and exact revision; `vllm-ascend` package version; CANN/driver and torch/torch_npu versions;
- NPU model and die count; `dp`, `tp`, `pp`, executor backend and graph mode;
- model/checkpoint revision, dtype, cache mode and whether the model is hybrid;
- the exact idle/active schedule and the state rows that must not change;
- the pre-registered checksum tolerance and the bounded collective timeout.

Freeze these facts. A missing fact is `condition_unknown`, not a pass.

## Step 1: Mount the L1 runtime invariants and the L2 SOP

Load, in this order:

1. `unit-vllm-ascend-npu-runtime-and-graph-invariants` — the `execute_model -> _build_attn_group_metadata -> _dummy_run` lifecycle, the stateful versus stateless cache distinction, the `skip_gdn_state_update` / `num_actual_tokens=0` contract, and the `query_start_loc` / `num_accepted_tokens` indexing rules.
2. `unit-vllm-ascend-verification-and-state-parity-gates` — Probe 1/2/3 and the Hermes distillation gate.
3. `unit-vllm-ascend-idle-dp-gdn-state` — the concrete fix pattern and the recorded 2-rank receipt.

Then state the hypothesis explicitly: the idle dummy presents a synthetic one-token GDN schedule, so the recurrent kernel updates a state row that belongs to no request. The competing hypothesis — that the zero-token metadata still writes state — is not assumed false; Step 3 tests it.

## Step 2: Zero-NPU `_dummy_run` / `_build_attn_group_metadata` probe

Run with no NPU and a mocked runner. Assert all of:

```python
runner.parallel_config = SimpleNamespace(
    distributed_executor_backend="external_launcher",
    data_parallel_size=2,
)
runner._dummy_run = MagicMock()
runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=0))
runner._dummy_run.assert_called_once_with(1, skip_gdn_state_update=True)
```

In the builder spy, assert for the GDN group:

- `gdn_query_start_loc` is a zero-filled prefix, not `[0, 1, 2, ...]`;
- `num_actual_tokens == 0`;
- `max_query_len == 0`;
- `is_prefilling` is all false;
- the non-GDN group still gets the dummy one-token shape.

Also assert the real path: a batch with scheduled tokens calls the builder with the default `skip_gdn_state_update=False` and a nonzero token count. Run the pinned unit selection, for example `python -m pytest tests/ut/worker/test_model_runner_v1.py -k dp_dummy_run`, and keep the output.

**Exit:** if any assertion fails, stop. The defect is upstream of state parity and a checksum run would be uninterpretable.

## Step 3: 1-rank and 2-rank state-checksum invariance plus HCCL collectives

### Probe 2 — one rank

1. Allocate the real `conv_state` and `ssm_state` rows; snapshot both and store per-row checksums, not a single whole-cache hash.
2. Run one idle zero-token dummy on the pre-fix path and on the fixed path.
3. On the fixed path, run one real decode for the active request and snapshot again.
4. Compare with the `fp32` reference and the pre-registered tolerance.

Required observations:

- pre-fix idle state moves by a synthetic one-token delta;
- fixed idle state is unchanged within tolerance;
- active state advances;
- logits match the reference; no NaN/Inf or dtype/shape change.

### Probe 3 — two ranks, DP2 TP1

Use exactly two ranks, `dp=2`, `tp=1`, one idle and one active. Bound every collective with a timeout and require all ranks to exit `0`.

Required observations:

- every rank enters and leaves `coordinate_batch_across_dp`; no hang;
- idle rank `conv_state` and `ssm_state` equal their pre-step snapshots;
- active rank state advances and its logits match the Probe 2 reference;
- unrelated KV blocks and other layers are unchanged;
- the pre-fix negative control fails the idle-state assertion.

The recorded reproduction on Ascend910_93 produced this table for both state tensors:

| Path | Idle rank | Active rank | HCCL |
| --- | --- | --- | --- |
| pre-fix | `[1.125, 1.25]` overwritten | `[1.5, 1.75]` | completes |
| post-fix | `[0.125, 0.25]` preserved | `[1.5, 1.75]` | completes |

The original idle state was `[0.125, 0.25]`; the pre-fix synthetic token added exactly one unit per element. The fixed path preserves it while the active rank still advances. This is a receipt for this topology, not a universal tolerance.

**Exit:** a hang, a nonzero exit, an idle-state change, or an active-state non-update is a failure. Do not "fix" it by disabling the GDN kernel, stubbing the recurrent op, dropping the collective, or loosening the tolerance after seeing the result.

## Step 4: Distill the verified pattern and refuted hypotheses into `reviewed_change`

Only after Step 2 and Step 3 pass for the same code state, run the Hermes `knowledge_distill` milestone. The milestone must contain:

- verified root cause: the synthetic one-token GDN schedule on the idle dummy; the fix is `skip_gdn_state_update=True` with zero-filled `gdn_query_start_loc` and `num_actual_tokens=0`;
- refuted hypothesis: "zero-token metadata still writes state" — refuted by the Probe 2/3 checksums;
- negative control and the observed failure;
- topology, model, dtype, graph mode and software stack;
- exact conditions and evidence references;
- the reviewed units as complete bytes with expected content hashes and `base_generation`.

Render a `mindie-reviewed-change/1` plan with `review_reference`, `base_generation`, and `changes`; validate it with `scripts/apply_reviewed_change.py` before any write. A passing probe is not a `reviewed_change`; the reviewed projection is the artifact, and the remote publication is a separate result.

If a probe fails, distill the failure instead: record the exact assertion, the rank and the tensor, mark the hypothesis refuted or unresolved, and do not increment a unit to `reproduced_empirical`.

## Anti-patterns

- Comparing only aggregate cache hashes: they hide which row changed.
- Reusing the zero-filled idle `gdn_query_start_loc` for a real decode.
- Passing `skip_gdn_state_update=True` for a batch with scheduled tokens.
- Declaring success from Probe 1 alone.
- Copying the idle-rank receipt to a different model, TP topology or cache mode without re-running Probe 3.

## Limits

The receipt covers Ascend910_93, DP2 TP1, the recorded model and cache mode. A2/A3 differences, other graph modes, ACLGraph capture, weight dtypes and full serving remain separate conditions. The workflow improves the evidence state; it does not prove universal numerical equivalence.
