---
category: measurement
conditions:
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - vllm-ascend-idle-dp
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
domain_path: _common
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-verification-and-state-parity-gates
inherits_from: []
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/tests/ut/worker/test_model_runner_v1.py
  source_digest: 11e7ef8d243dcca647e6ed35e524a5f261606182e5a6deb39fa9c1dbd8acb6a3
  source_locator: vllm-project/vllm-ascend@d8c9e3a673af918c94a7794ef6f1f69c024d9c62:tests/ut/worker/test_model_runner_v1.py
  source_type: verification_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: vllm-ascend verification and state-parity gates (metadata, single-rank, multi-rank)
triggers:
  error_signatures:
  - unit test passes with mocked metadata but no NPU state checksum was taken
  - single-rank logits match while idle-rank conv_state or ssm_state changed
  - two-rank DP run hangs in HCCL before the idle-rank parity assertion
  - a source-only fix is promoted to reviewed_change without Probe 2 or Probe 3
  keywords:
  - vllm-ascend
  - verification
  - state parity
  - checksum
  - fp32 reference
  - HCCL
  - DP
  - TP
  - probe
  - idle rank
  - reviewed_change
  - Hermes
---

# vllm-ascend verification and state-parity gates

## Purpose

These gates convert a runtime hypothesis into evidence in a fixed order. They are deliberately not a benchmark and not a score: each gate answers one question and can fail closed. A later gate is never implied by an earlier one, and a source-only diagnosis never satisfies an execution gate.

The gates apply to `vllm-ascend` on Ascend 910B (A2) and Ascend 910_93 (A3 dual-Die), with the model, dtype, graph mode and software stack recorded by the consuming task. Conditions are established from observed task facts, never from the fix commit, a gold patch or a reviewer conclusion.

## Gate 0: preconditions

Before any probe, record:

- repository and exact source revision;
- NPU model, die count, driver/CANN and torch/torch_npu versions;
- parallel plan (`dp`, `tp`, `pp`), executor backend and graph mode;
- model checkpoint, config revision, dtype and cache mode;
- the exact idle/active request schedule and the state tensors under test.

If a precondition is unknown, the result is `condition_unknown`, not a pass. Freeze the inputs and the tolerance before looking at target outputs; do not tune a tolerance from the result being judged.

## Probe 1: zero-NPU metadata and builder unit probe

**Question:** does the runner construct the intended metadata at all?

Run the unit lane with no NPU and a mocked runner. At the pinned revision the required assertions are:

```python
runner.parallel_config = SimpleNamespace(
    distributed_executor_backend="external_launcher",
    data_parallel_size=2,
)
runner._dummy_run = MagicMock()
runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=0))
runner._dummy_run.assert_called_once_with(1, skip_gdn_state_update=True)
```

For the builder, assert that the GDN group receives a zero-filled `gdn_query_start_loc`, `num_actual_tokens == 0`, `max_query_len == 0` and all-false `is_prefilling`, while a non-GDN group still receives the dummy one-token metadata. Assert that a real batch calls the builder with the default `skip_gdn_state_update=False` and a nonzero token count.

**Pass:** every metadata assertion holds and the zero-token path still reaches the DP collective call. **Fail:** any assertion is skipped, the flag is passed on a real batch, or the test only checks that no exception was raised.

Probe 1 establishes metadata construction only. It cannot establish that the kernel leaves state unchanged.

## Probe 2: single-rank NPU tensor, logit and state checksum parity

**Question:** on one NPU rank, does the fixed path preserve idle state and match a float32 reference?

1. Seed and pin every input. Disable nondeterministic kernels where the stack allows it.
2. Allocate the model's real `conv_state` and `ssm_state` rows with known values. Snapshot both tensors and a checksum over each row before the step.
3. Run one idle zero-token dummy step on the pre-fix path and on the fixed path. Decode the checksums into per-element vectors so a one-token synthetic write is visible.
4. On the fixed path, run one real one-token or accepted-token decode for the active request and snapshot the same state rows again.
5. Run the same inputs through a CPU `fp32` reference implementation and compare logits, selected hidden states and state rows with a pre-registered absolute/relative tolerance.

**Pass criteria:**

- pre-fix idle state changes by a synthetic token-sized delta; fixed idle state is bitwise or tolerance-identical to its before snapshot;
- the active request's state advances on the fixed path and matches the reference direction and tolerance;
- no NaN/Inf, no shape change, no dtype change;
- logits match the reference at the declared tolerance.

**Fail:** idle state changes, the active state does not advance, only a checksum of the whole cache is compared without row identity, or the reference is the NPU output itself.

Probe 2 establishes single-rank reproduced behavior. It does not establish multi-rank collective completion.

## Probe 3: multi-rank DP/TP HCCL and idle/active state parity

**Question:** with every rank participating, does the collective complete and do idle and active ranks behave differently but correctly?

Use the smallest real topology that exercises the defect: for the idle-DP class, two ranks, `dp=2`, `tp=1`; for a TP claim, at least two ranks in one TP group. Give every rank a bounded timeout and a barrier so a hang is a failure, not an indefinite wait.

**Pass criteria:**

- all ranks enter and leave `coordinate_batch_across_dp`; no rank times out and every process exits `0`;
- the idle rank's `conv_state` and `ssm_state` equal their pre-step snapshots within the declared tolerance;
- the active rank's state advances and its logits match the Probe 2 reference or a recorded cross-rank reference;
- unrelated KV blocks and other layers are unchanged;
- a repeat run gives the same verdict, and a negative control (the pre-fix path) fails the idle-state assertion.

**Fail:** any hang, any nonzero exit, an idle-state change, an active-state non-update, or a pass obtained by disabling the GDN kernel, replacing it with a stub, or dropping the collective.

Probe 3 establishes multi-rank reproduced behavior for the recorded topology only. It does not generalize to another model, cache mode or interconnect.

## Hermes milestone distillation gate

Distillation is allowed only after Probe 2 and Probe 3 pass for the same code state, or after a probe fails in a way that is itself the recorded result. The Hermes milestone must contain:

- the verified root cause, with the exact metadata or state-index path;
- the positive fix pattern and the conditions under which it applies;
- refuted hypotheses, each with the probe that refuted it;
- the negative control and the observed failure;
- the topology, model, dtype, graph mode and software stack;
- the evidence references and the content hashes of the units being changed.

Only then may the milestone be rendered as a `reviewed_change` with `base_generation`, complete unit bytes and expected content hashes. A passing Probe 1 alone produces a source/metadata note, not a `reviewed_change`. A passing Probe 2 alone produces a single-rank result. A `reviewed_change` is a reviewed projection, not proof of remote publication or installed native acceptance.

## Evidence ladder and limits

| Evidence | Minimum gate | What it supports |
| --- | --- | --- |
| source/metadata claim | Probe 1 | builder wiring and control flow |
| single-rank reproduced | Probe 1 + 2 | one-rank state and logit parity |
| multi-rank reproduced | Probe 1 + 2 + 3 | DP/TP collective plus idle/active parity |

These gates do not measure throughput, do not prove numerical equivalence for all inputs, and do not authorize copying one domain's unit into another. A budget exclusion, a skipped rank or an unknown condition is reported, never silently treated as a pass.
