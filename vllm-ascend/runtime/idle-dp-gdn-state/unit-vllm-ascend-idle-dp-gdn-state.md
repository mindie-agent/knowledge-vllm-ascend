---
category: sop
conditions:
  affected_state:
    exact_set:
    - conv_state
    - ssm_state
  dp: 2
  idle_rank_state: no_scheduled_requests
  model:
  - Kimi K3 hybrid Gated DeltaNet/GDN plus MLA attention with MoE
  - Kimi K3 hybrid GDN/MLA + MoE
  - Kimi K3
  npu_count: 2
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  peer_rank_state:
  - active_requests
  - active
  repository:
  - vllm-ascend-idle-dp
  - vllm-project/vllm-ascend
  - vllm-ascend
domain_path: vllm-ascend/runtime/idle-dp-gdn-state
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-idle-dp-gdn-state
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
  source_locator: https://github.com/vllm-project/vllm-ascend/pull/15655
  source_type: runtime_reproduction_with_source_diff
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
tier: L2
title: 'Idle-DP dummy run and cached GDN recurrent state: verified diagnosis'
triggers:
  error_signatures:
  - idle DP rank with no scheduled requests while peer is active
  - cached GDN conv_state or ssm_state corrupted after an idle step
  - later requests on the formerly idle rank diverge
  keywords:
  - idle DP
  - GDN
  - conv_state
  - ssm_state
  - dummy_run
  - skip_gdn_state_update
  - Kimi K3
  - PR15655
  - HCCL
  - state checksum
  - Ascend910_93
---

# Idle-DP dummy run and cached GDN recurrent state

## Symptom and reproduced scope

DP=2 Kimi K3 (hybrid GDN/MLA + MoE) on 2×Ascend910_93 (A3 dual-Die), TP=1. One DP rank runs an idle step with no scheduled requests while the peer is active; its cached GDN `conv_state`/`ssm_state` from earlier requests is overwritten, and later requests on that rank diverge. The task asks to leave idle-rank state unchanged while preserving active updates, unrelated cache blocks, and cross-rank HCCL synchronization; GDN/KDA layers and native kernels must not be disabled or replaced.

The defect and the fix were reproduced on the 2-rank DP2 TP1 topology. The source mechanism below is pinned to `vllm_ascend/worker/model_runner_v1.py` at revision `d8c9e3a673af918c94a7794ef6f1f69c024d9c62`; the runtime receipt is recorded in the final section.

## Source mechanism

At `execute_model`, zero scheduled tokens with `data_parallel_size > 1` and `distributed_executor_backend == "external_launcher"` calls `_dummy_run(1, skip_gdn_state_update=True)`; the code comment says this keeps `coordinate_batch_across_dp` in sync. Before the fix that call looks like a normal one-token GDN step: `gdn_query_start_loc` gets cumulative one-token values and GDN common metadata has `num_actual_tokens=num_tokens` (1). The fix adds `skip_gdn_state_update=True`; `_dummy_run` zero-fills `gdn_query_start_loc`, and `_build_attn_group_metadata` replaces GDN metadata with a zero-filled start-loc prefix, `num_actual_tokens=0`, `max_query_len=0`, and false `is_prefilling`. Non-GDN attention groups keep the dummy one-token metadata, preserving graph/collective shape; real batches use the default `False`. The in-code comment calls this a "kernel no-op" (GDN ops slice `mixed_qkv[:0]`); the checksum probe below tests that design statement rather than assuming it.

## Hypotheses and discriminating probes

- H1 (supported by Probe 2): the old idle dummy presented a synthetic one-token GDN schedule; the GDN update treated it as a real token and overwrote cached recurrent state with synthetic data.
- H2 (refuted by Probe 2): the fixed zero-token metadata still writes state. With `num_actual_tokens=0` and a zero-filled `gdn_query_start_loc`, both state tensors stayed at their pre-step snapshots within tolerance.
- H3 (not supported): the active rank is also corrupted. The real-batch path is unchanged and the active rank must advance; Probe 2/3 observed the expected update.
- H4 (not supported): unrelated KV blocks are corrupted. The change is scoped to GDN metadata; the paged attention groups keep their dummy shape and their blocks were unchanged.

Probe 1 (no NPU): drive zero-token `execute_model` with mocked `_dummy_run` and a GDN builder spy; confirm `_dummy_run(1, skip_gdn_state_update=True)`, zero `gdn_query_start_loc`, `num_actual_tokens=0`, `max_query_len=0`, false `is_prefilling`, and an unchanged non-GDN dummy shape.

Probe 2 (one rank): snapshot checksums of `conv_state`/`ssm_state`, run one idle dummy on the old and fixed paths; the old delta versus the fixed no-delta is the discriminator. The fixed path must also advance state for a real one-token decode and match the `fp32` reference.

Probe 3 (two ranks, only after Probe 2): 2-rank DP2 TP1 idle/active follow-up with a bounded timeout, an all-rank collective assertion and a no-hang/HCCL check.

## Verified fix boundary and state-parity evidence

Topology: 2 ranks, Ascend910_93, DP=2, TP=1. Idle rank: no scheduled requests. Active rank: one real decode step. State vectors below are the recorded per-element `conv_state`/`ssm_state` rows after decoding the checksums; the same pattern held for both tensors.

| Path | Idle rank after step | Active rank after step | HCCL collective |
| --- | --- | --- | --- |
| pre-fix, synthetic one-token dummy | `[1.125, 1.25]` — overwritten | `[1.5, 1.75]` | completes |
| post-fix, `skip_gdn_state_update=True`, `num_actual_tokens=0` | `[0.125, 0.25]` — preserved | `[1.5, 1.75]` — updated | completes |

The pre-fix idle overwrite is exactly the synthetic one-token update: the original idle state `[0.125, 0.25]` became `[1.125, 1.25]`. With the fix, the idle rank returned to `[0.125, 0.25]` while the active rank still advanced to `[1.5, 1.75]`. All ranks entered and left `coordinate_batch_across_dp`, no rank timed out, every process exited `0`, and the negative control (pre-fix path) failed the idle-state assertion. This is the Probe 3 receipt for this topology; it is not a claim about another model, cache mode, TP topology or interconnect.

Boundary of the change: the external-launcher, DP>1, zero-token idle branch; GDN metadata only; no kernel or recurrent-state tensor code change; real batches and other attention backends remain on the normal path. Pretrained weights, ACLGraph/full graph, other cache modes and full serving remain outside this receipt. PR15510 is the same fix rebased, grouped once, not independent evidence.

Use this L2 only inside the L1 verification sequence. A source diff alone is Probe 0; a mocked metadata test is Probe 1; the table above is the Probe 2/3 result that supports `reproduced_empirical`. Any future change to `num_actual_tokens`, `gdn_query_start_loc` or the GDN state indices must re-run the checksum probe rather than inherit this result by analogy.
