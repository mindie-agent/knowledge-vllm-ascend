---
category: workflow
conditions:
  framework:
  - vllm-ascend
  - any
domain_path: vllm-ascend/speculative-decoding/proposers/hybrid-mtp-rollback
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-step3p5-and-hybrid-mtp-cache-rollback-workflow
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/worker/v2/spec_decode/dspark/speculator.py
  source_digest: 53f5f4f3c7e5430058f5bc20a2329b6880fc343dbf513f4b2b7ef52fc68819ac
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/worker/v2/spec_decode/dspark/speculator.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 3
schema: mindie-graded-unit/1
tier: L3
title: Speculative Decode MTP / DSpark ACLGraph (`vllm_ascend/worker/v2/spec_decode/dspark/speculator.py`)
  3-Stage Tuning Workflow
triggers:
  error_signatures:
  - FIA TND contract violation queryT (12) != last element of actualSequenceLengthQ
    (9)
  - MTP speculative rejection corrupts recurrent GDN/KDA/conv state or compressor
    state on subsequent steps
  keywords:
  - dspark
  - speculator.py
  - dspark_aclgraph_step
  - actualSequenceLengthQ
  - actual_seq_lengths_q
  - num_query_per_req
  - num_accepted_tokens
  - aclgraph
---
# Speculative Decode MTP / DSpark ACLGraph (`vllm_ascend/worker/v2/spec_decode/dspark/speculator.py`) 3-Stage Tuning Workflow

## Trigger and scope

Use this 3-stage convergence workflow when debugging or tuning `dspark_aclgraph_step` in `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py` where v2 DSpark full-graph replay raises `RuntimeError: FIA TND contract violation: queryT (12) != last element of actualSequenceLengthQ (9)` or triggers host D2H synchronization.

## Stage 1 — Provision `1 + K` state slots per active request while keeping draft metadata unpadded

1. Size `target_tokens = num_reqs * (1 + num_speculative_tokens)` while building `actual_seq_lengths_q` from `num_reqs * num_query_per_req` (`9`).
   - **Limitation**: When the draft ACLGraph bucket is padded to `12`, `actual_seq_lengths_q[-1] == 9 != 12`, still raising `RuntimeError: FIA TND contract violation: queryT (12) != last element of actualSequenceLengthQ (9)` (`verdict: "refuted"`).

## Stage 2 — Rebuild `actual_seq_lengths_q` inside the captured step using `.cpu().sum().item()`

1. Pad `actual_seq_lengths_q[-1] = draft_tokens_padded` inside the captured step while reading `int(num_accepted_tokens.cpu().sum().item())` and `seq_lens.cpu().tolist()`.
   - **Limitation**: Calling `.cpu()` or `.item()` inside the captured step introduces `captured_step_d2h_syncs = 2 > 0` and dynamic metadata allocations inside capture (`verdict: "refuted"`).

## Stage 3 — Force draft and target graphs into a single shared `(1 + K) * num_reqs` bucket (`16`)

1. Force `draft_tokens_padded = target_tokens_padded = 16` so both graphs share one bucket.
   - **Limitation**: Over-pads the draft graph (`draft_tokens_padded = 16 != 12`) instead of selecting the tightest capture bucket covering `num_reqs * num_query_per_req = 9` (`12`) (`verdict: "refuted"`).

## Stagnation & RSI Escalation Rule

When all 3 stages above (`Stage 1`, `Stage 2`, `Stage 3`) fail to resolve both the `FIA TND contract violation` and the captured-step D2H synchronization (`3` consecutive `refuted` attempts), record the 3 failed `L3` stage attempts in `logs/experiment-ledger.jsonl` and re-query `knowledge_query` with `ledger_entries` + `bottleneck_summary` (or `mode="explore"`) to trigger RSI automatic switching and mount the `L4` mutator.
