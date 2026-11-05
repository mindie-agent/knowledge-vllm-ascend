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
revision: 4
schema: mindie-graded-unit/1
tier: L3
title: Speculative Decode MTP / DSpark ACLGraph (`vllm_ascend/worker/v2/spec_decode/dspark/speculator.py`)
  Standard Tuning Workflow
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
  - dspark_aclgraph_speculative_step
  - actualSequenceLengthQ
  - actual_seq_lengths_q
  - num_query_per_req
  - num_accepted_tokens
  - aclgraph
---
# Speculative Decode MTP / DSpark ACLGraph (`vllm_ascend/worker/v2/spec_decode/dspark/speculator.py`) Standard Tuning Workflow

## Trigger and scope

Use this standard 3-stage workflow when configuring `dspark_aclgraph_speculative_step` in `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py`.

## Stage 1 — Size `target_tokens` to `num_reqs * (1 + num_speculative_tokens)`

1. Edit `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py` to provision `1 + num_speculative_tokens` state slots per active request while building `actual_seq_lengths_q` from `num_reqs * num_query_per_req`, then run `pytest -q tests/ut/worker/v2/test_dspark_speculator.py` to evaluate Stage 1 and record its `candidate_sha256` outcome in `logs/experiment-ledger.jsonl`.

## Stage 2 — Clamp `actual_seq_lengths_q[-1]` to `draft_tokens_padded` inside the step

1. Edit `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py` to update the last element of `actual_seq_lengths_q` to `draft_tokens_padded` while computing `updated_seq_lens` from `accepted_mask`, then run `pytest -q tests/ut/worker/v2/test_dspark_speculator.py` to evaluate Stage 2 and record its `candidate_sha256` outcome in `logs/experiment-ledger.jsonl`.

## Stage 3 — Share a Single Padded Bucket between Draft and Target Graphs

1. Edit `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py` to set `draft_tokens_padded = target_tokens_padded` to the bucket covering `num_reqs_padded * num_query_per_req` (and eliminate `.item()`/`.cpu()` host syncs), then run `pytest -q tests/ut/worker/v2/test_dspark_speculator.py` to evaluate Stage 3 and record its `candidate_sha256` outcome in `logs/experiment-ledger.jsonl`.

## Verification & Follow-Up Retrieval

1. Always apply and test Stage 1, Stage 2, and Stage 3 as 3 separate incremental edits (running `pytest -q tests/ut/worker/v2/test_dspark_speculator.py` after each edit) so `logs/experiment-ledger.jsonl` records 3 distinct `candidate_sha256` evaluations.
2. Do **not** attempt to brute-force test capture hashes via standalone scripts or inspect parent directories outside the workspace (`..`): the knowledge routing engine gates deeper structural specifications strictly on `logs/experiment-ledger.jsonl` containing 3 distinct `candidate_sha256` entries produced by editing `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py` and running `pytest -q tests/ut/worker/v2/test_dspark_speculator.py` for Stage 1, Stage 2, and Stage 3 in sequence.
3. If the Ascend 950DT check remains `refuted` after all 3 stage edits, call `knowledge_query` again with default `mode="reuse"` (`conditions={"framework": "vllm-ascend", "soc": "Ascend950DT"}`, `token_budget=24000`, `goal`, and `bottleneck_summary`) to retrieve deeper structural remedies.
