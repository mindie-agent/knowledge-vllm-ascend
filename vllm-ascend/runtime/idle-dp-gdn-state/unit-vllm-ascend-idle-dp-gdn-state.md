---
category: sop
conditions:
  affected_state:
    exact_set:
    - conv_state
    - ssm_state
  base_commit: c3394206bdd00b132580ae439555b4f1a70d5bf6
  dp: 2
  idle_rank_state: no_scheduled_requests
  model:
  - Kimi K3 hybrid Gated DeltaNet/GDN plus MLA attention with MoE
  - Kimi K3 hybrid GDN/MLA + MoE
  - Kimi K3
  npu_count: 2
  npu_model: Ascend910
  peer_rank_state:
  - active_requests
  - active
  repository:
  - vllm-ascend-idle-dp
  - vllm-project/vllm-ascend
  - vllm-ascend
content_digest: sha256:72f2bfadc7899c951e711b81719ce6b5854cbfe6dd343b00bc35d75b221deb00
domain_path: vllm-ascend/runtime/idle-dp-gdn-state
evidence_strength: hypothesis
id: unit-vllm-ascend-idle-dp-gdn-state
inherits_from: []
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/pull/15655
  - https://github.com/vllm-project/vllm-ascend/pull/15510
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/tests/ut/worker/test_model_runner_v1.py
  experiment_status: not_run
  source_digest: 61ee8bc56d4ac0e1a7f43d1804bfe41c1a6699181774938907c97174f2348343
  source_locator: https://github.com/vllm-project/vllm-ascend/pull/15655
  source_type: public_source_diff_and_task_input
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
target_domain: vllm-ascend
tier: L2
title: 'Idle-DP dummy run and cached GDN recurrent state: source-scoped diagnosis'
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
---
# Idle-DP dummy run and cached GDN recurrent state

## Symptom and task scope

DP=2 Kimi K3 (hybrid GDN/MLA + MoE) on 2×Ascend910. One DP rank runs an idle step with no scheduled requests while the peer is active; its cached GDN `conv_state`/`ssm_state` from earlier requests is reported corrupted, and later requests on that rank diverge. The task asks to leave idle-rank state unchanged while preserving active updates, unrelated cache blocks, and cross-rank HCCL synchronization; GDN/KDA layers and native kernels must not be disabled or replaced. This is a vllm-ascend runtime record; evidence is source-only.

## Source mechanism

At `execute_model`, zero scheduled tokens with `data_parallel_size > 1` and `distributed_executor_backend == "external_launcher"` (source/test-derived conditioning, not the original instruction) calls `_dummy_run(1)`; the code comment says this keeps `coordinate_batch_across_dp` in sync. Before the fix that call looks like a normal one-token GDN step: `gdn_query_start_loc` gets cumulative one-token values and GDN common metadata has `num_actual_tokens=num_tokens` (1). The fix adds `skip_gdn_state_update=True`; `_dummy_run` zero-fills `gdn_query_start_loc`, and `_build_attn_group_metadata` replaces GDN metadata with a zero-filled start-loc prefix, `num_actual_tokens=0`, `max_query_len=0`, and false `is_prefilling`. Non-GDN attention groups keep the dummy one-token metadata, preserving graph/collective shape; real batches use the default `False`. The in-code comment calls this a "kernel no-op" (GDN ops slice `mixed_qkv[:0]`); this is a source design statement, not measured state invariance.

## Hypotheses and smallest discriminating probe

- H1 (source-supported): the old idle dummy presented a synthetic one-token GDN schedule; if GDN updates state from it, cached recurrent state is overwritten with non-request data.
- H2 (not excluded): the zero-token metadata still writes state; the GDN kernel body is not admitted.
- H3 (not supported): the active rank is also corrupted; the real-batch path is unchanged and the task requires active updates.
- H4 (not supported): unrelated KV blocks are corrupted; this change is scoped to GDN metadata.

Probe 1, no NPU: drive zero-token `execute_model` with mocked `_dummy_run` and a GDN builder spy; confirm call args, zero `gdn_query_start_loc`, and `num_actual_tokens=0`. Probe 2, one rank: snapshot checksums of `conv_state`/`ssm_state`, run one idle dummy on old and fixed paths; old delta vs fixed no-delta is the discriminator. Probe 3, only after 2: 2-rank idle/active follow-up request plus no-hang/HCCL check. A full serving/benchmark run is not needed.

## Fix boundary and unresolved result

The source boundary is the external-launcher, DP>1, zero-token idle branch; GDN metadata only; no kernel or recurrent-state tensor code change; real batches and other attention backends remain on the normal path. Unresolved: whether the kernel truly leaves both state tensors unchanged, and whether HCCL ordering is preserved. Pretrained weights, ACLGraph/full graph, and serving are outside the task conditions. PR15510 is the same fix rebased, grouped once, not independent. Evidence remains L2 source-only: no fail-before/pass-after, state checksum, or HCCL receipt.
