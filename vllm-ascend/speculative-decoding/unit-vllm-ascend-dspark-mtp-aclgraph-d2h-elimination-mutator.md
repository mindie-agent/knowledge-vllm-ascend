---
category: mutator
conditions:
  dspark_graph_path: v2_ascend_dspark_full_graph
  execution_mode: aclgraph
  runtime: vllm-ascend
  soc:
  - Ascend910B2
  - Ascend910B3
  - Ascend910B4
  - Ascend910_93
  workload_phase:
  - decode
  - speculative_decode
domain_path: vllm-ascend/speculative-decoding
evidence_strength: hypothesis
id: unit-vllm-ascend-dspark-mtp-aclgraph-d2h-elimination-mutator
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop
provenance:
  evidence_refs:
  - docs/evidence/l4-dspark-mtp-aclgraph-d2h-elimination-ablation.json
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/utils.py#L1058
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py#L26-L46
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py#L108-L150
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/spec_decode/dspark_proposer.py#L37-L41
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/spec_decode/dspark_proposer.py#L70-L71
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py#L81-L116
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py#L857-L900
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py#L1339-L1348
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py#L1448-L1466
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py#L927-L949
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py#L5186-L5214
  experiment_status: not_run_source_only
  source_commit: d8c9e3a673af918c94a7794ef6f1f69c024d9c62
  source_digest: 99555432d52fc8846ea1870c65a293030914209d0795103f90f35b3e2ef89a33
  source_locator: docs/evidence/l4-dspark-mtp-aclgraph-d2h-elimination-ablation.json
  source_repository: https://github.com/vllm-project/vllm-ascend
  source_type: ablation_design_receipt
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
tier: L4
title: 'v2 DSpark ACLGraph bucket alignment and device-resident D2H elimination'
triggers:
  error_signatures:
  - FIA rejects a replayed v2 DSpark graph because actual_seq_lengths_q[-1] differs
    from the captured query-token count (queryT != last element of actualSequenceLengthQ)
  - a target verification step uses K+1 tokens per request while the captured bucket
    was sized from K * num_reqs
  - num_accepted_tokens.item() or .cpu() or a host seq_lens read appears inside the
    captured speculative step
  - the observed v2 num_query_per_req is absent, or the graph path is v1/eager or
    an unrelated speculative method
  keywords:
  - dspark
  - speculative_decode
  - aclgraph
  - num_query_per_req
  - actual_seq_lengths_q
  - FIA
  - num_accepted_tokens
  - d2h
  - bucket
  - v2
  - hypothesis
---
# v2 DSpark ACLGraph bucket alignment and device-resident D2H elimination

## Evidence status

This entry is a source-derived optimization design (`evidence_strength: hypothesis`, `experiment_status: not_run_source_only`). It defines the three separable factors, leave-one-out ablation arms, and verification criteria grounded in the pinned `vllm-ascend` source (`d8c9e3a673af918c94a7794ef6f1f69c024d9c62`). Live NPU baseline, candidate, and leave-one-out ablation benchmarks have not yet been executed.

## Applicability and mandatory scope condition

Apply only to `vllm-ascend` v2 DSpark full-graph speculative decode on Ascend 910B2/910B3/910B4/910_93 during decode or speculative decode. The mandatory unit condition is `dspark_graph_path: v2_ascend_dspark_full_graph`. A v1 `AscendDSparkProposer` path (`use_cuda_graph=False`, eager), another runner or graph path, or an unrelated speculative method is out of scope even when the runtime, soc, aclgraph and workload-phase fields match; an unknown value is `condition_unknown` and excludes the unit.

Before use, also record `K=num_speculative_tokens`, `TP`, sequence-parallel setting, capture bucket set and the observed v2 `num_query_per_req`. The v2 class consumes that field but delegates initialization to the upstream `DSparkSpeculator` via `super().__init__`; that initializer is not vendored in the pinned tree. Do not infer a v2 default or the v1 K-vs-K+1 formula. If `num_query_per_req` is not observed, stop.

## Source attribution

Pinned source: `vllm-project/vllm-ascend` commit `d8c9e3a673af918c94a7794ef6f1f69c024d9c62`. These are source-reading anchors, not run observations.

- `vllm_ascend/utils.py:1058`: the target verification step has `uniform_decode_query_len = 1 + num_speculative_tokens`, i.e. `K+1`.
- `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py:26-28,41,44-46`: the v2 module imports upstream `vllm.v1.worker.gpu.spec_decode.dspark.speculator.DSparkSpeculator` and the `AscendDSparkSpeculator` class subclasses it; its own `__init__` calls `super().__init__` and sets `input_batch` only, with no `num_query_per_req` initialization.
- `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py:108-150`: `build_draft_attn_metadatas` consumes `self.num_query_per_req` for `num_tokens_padded` and rebuilds `actual_seq_lengths_q`. Its docstring records the FIA TND requirement that the last element equal the graph query-token count, else `queryT != last element of actualSequenceLengthQ`.
- `vllm_ascend/spec_decode/dspark_proposer.py:37-41`: in v1 only, `sample_from_anchor` defaults true; `num_query_per_req` is `K` when true and `K+1` when false. This does not define v2.
- `vllm_ascend/spec_decode/dspark_proposer.py:70-71`: v1 sets `use_cuda_graph=False` and is eager-only; v1 is excluded.
- `vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py:81-116`: the v2 DFlash ACLGraph manager builds draft metadata before replay and updates full-graph parameters; the v2 DSpark docstring says DSpark inherits that path.

The pinned tree does not bind the upstream initializer, so no v2 `num_query_per_req` default or formula is asserted.

## Proposed transformation

The factors are separable for leave-one-out.

1. **`align_target_and_observed_draft_graph_buckets`.** Align the target and draft graphs separately. The target step uses `K+1`; the draft bucket uses the observed v2 `num_query_per_req`. Choose a capture bucket covering the observed padded token count, mark tail slots `slot_mapping=-1` with masked `seq_lens=0`, and never let tail rows reach live KV or recurrent state. The source applies `lcm(K+1, TP)` alignment only on the SP+TP>1 path. Do not force the draft block into the target bucket.
2. **`hoist_attention_metadata_out_of_capture`.** Build the draft/FIA metadata for the observed bucket, including an `actual_seq_lengths_q` whose last element equals the captured query count, before replay, and update graph parameters outside the captured region. Do not allocate or rebuild metadata inside the captured region; no specific sink class is asserted.
3. **`device_side_num_accepted_tokens_and_seq_lens`.** Keep `num_accepted_tokens` and `seq_lens` device-resident for the captured step and update them in place. Do not call `.item()` or `.cpu()` inside the captured step; if a host consumer is unavoidable, synchronize after the step and report it.

## Missing experiment

No experiment was run. Missing: baseline and full-candidate receipts on the same frozen stack and observed `num_query_per_req`; one leave-one-out receipt per factor; raw paired timing; ACLGraph recapture counter; captured-step D2H trace; accepted-token/`seq_lens` parity including rejected-draft and bucket-shrink transitions; TPOT, ITL p50/p90 and `mean_acceptance_length`; guard results and independent receipt identifiers. No timing, recapture, D2H, parity, acceptance or retained-speedup claim is made.

## Future measured acceptance

Freeze the model/checkpoint, source revision, NPU model, CANN/driver, torch/torch_npu, dtype, `K`, `TP`, sequence-parallel setting, observed v2 `num_query_per_req`, source-bound upstream `DSparkSpeculator` revision, observed `dspark_graph_path`, bucket set, workload, warmup and tolerance. Require:

1. **Scope:** `dspark_graph_path=v2_ascend_dspark_full_graph` and an observed `num_query_per_req`; v1/eager, another runner path or an unrelated speculative method is excluded.
2. **Capture stability:** zero recaptures after warmup across every `K` and boundary bucket.
3. **D2H:** zero host D2H synchronizations in the captured-step trace; any `.item()`, `.cpu()` or blocking copy inside replay rejects the combination.
4. **State parity:** device `num_accepted_tokens` and `seq_lens` match an independent reference, including rejected-draft and bucket-shrink transitions.
5. **Paired performance:** report TPOT, ITL p50/p90 and `mean_acceptance_length` for baseline and full candidate on the same stack.
6. **Ablation:** the paired TPOT gain must weaken or disappear when the device-side D2H removal is ablated; bucket-only or metadata-only gains cannot claim this unit.
7. **Guards:** every declared guard passes, or its failure is reported as failure.

One qualifying run supports `single_observation` at most; `reproduced_empirical` requires an independent repeat. Schema validation, worker exit 0, unit tests or a prefilled receipt are not evidence.

## Fallback boundaries

- **V1/eager, non-ACLGraph or other method.** Use the normal speculative path; do not force capture or report a capture result.
- **Different `K`, `TP`, SP, observed `num_query_per_req`, upstream dependency revision or bucket family.** Align a new bucket set and repeat the capture gate; do not reuse this shape assumption.
- **Host scalar still required.** Keep the synchronization outside the captured step and report it; do not hide it or move it into replay.
- **Parity or ablation unproven.** Keep the unit at `hypothesis` or, at most, `single_observation`; never claim `reproduced_empirical` or `ablation_verified`.
- **Graph capture unavailable.** Report the execution-condition failure and stop; do not substitute eager results for a capture result.
