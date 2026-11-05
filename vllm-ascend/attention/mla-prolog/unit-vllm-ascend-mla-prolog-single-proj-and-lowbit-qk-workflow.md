---
category: workflow
conditions:
  framework:
  - vllm-ascend
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
  - Ascend950DT
domain_path: vllm-ascend/attention/mla-prolog
evidence_strength: single_observation
id: unit-vllm-ascend-mla-prolog-single-proj-and-lowbit-qk-workflow
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/mla.py
  source_digest: 39a81f2b820f6d203b78b3e1000b5e30c475281db1d99a270c65cba52a3f62ab
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/mla.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 5
schema: mindie-graded-unit/1
tier: L3
title: MLA Prolog Projection, RoPE, and KV Cache Write (`vllm_ascend/ops/mla.py`)
  Standard Tuning Workflow
triggers:
  error_signatures:
  - MLA prolog numerical check failed max_abs_err=0.0625 > 2e-2
  - Default path still performs redundant clone/repeat allocations
  keywords:
  - mla_prolog_rope_kvcache_step
  - vllm_ascend/ops/mla.py
  - kv_cache_nope
  - kv_cache_rope
  - slot_mapping
  - clone_repeat_ops
  - select_slice_ops
  - RoPE
  - MLA
---
# MLA Prolog Projection, RoPE, and KV Cache Write (`vllm_ascend/ops/mla.py`) Standard Tuning Workflow

## Trigger and scope

Use this standard 3-stage tuning workflow when optimizing `mla_prolog_rope_kvcache_step` in `vllm_ascend/ops/mla.py` on Ascend NPU.

## Stage 1 — Vectorize RMSNorm over the last dimension while keeping separate `w_q_a` and `w_kv_a` linear projections

1. Edit `vllm_ascend/ops/mla.py` to replace the per-token Python RMSNorm loops (`for idx in range(num_tokens)`) with batch RMSNorm (`torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)`) while keeping separate `F.linear(hidden_states, w_q_a)` and `F.linear(hidden_states, w_kv_a)` projections, then run `pytest -q tests/ut/ops/test_mla_prolog.py` to evaluate Stage 1 and record its `candidate_sha256` outcome in `logs/experiment-ledger.jsonl`.

## Stage 2 — Expand `cos` and `sin` across heads with `.unsqueeze(1).repeat(1, num_heads, 1)`

1. Edit `vllm_ascend/ops/mla.py` to replace the per-head Python loop (`for h_idx in range(num_heads)`) by repeating `cos` and `sin` across `num_heads` via `.unsqueeze(1).repeat(1, num_heads, 1)` before applying `_rotate_half_last_dim`, then run `pytest -q tests/ut/ops/test_mla_prolog.py` to evaluate Stage 2 and record its `candidate_sha256` outcome in `logs/experiment-ledger.jsonl`.

## Stage 3 — Block-offset paged KV cache writes with low-bit QK step rounding disabled (`enable_lowbit_qk_truncation=False`)

1. Edit `vllm_ascend/ops/mla.py` to keep `enable_lowbit_qk_truncation=False` (avoiding `1/16` step rounding on `q_out`, which degrades BF16 accuracy beyond `max_abs_err <= 2e-2`) while writing `kv_cache_nope[blk_idx, blk_off]` and `kv_cache_rope[blk_idx, blk_off]` per valid slot (`slot_mapping >= 0`), then run `pytest -q tests/ut/ops/test_mla_prolog.py` to evaluate Stage 3 and record its `candidate_sha256` outcome in `logs/experiment-ledger.jsonl`.

## Verification & Follow-Up Retrieval

1. Always apply and test Stage 1, Stage 2, and Stage 3 as 3 separate incremental edits (running `pytest -q tests/ut/ops/test_mla_prolog.py` after each edit) so `logs/experiment-ledger.jsonl` records 3 distinct `candidate_sha256` evaluations.
2. Do **not** attempt to brute-force test capture hashes via standalone scripts or inspect parent directories outside the workspace (`..`): the knowledge routing engine gates deeper structural specifications strictly on `logs/experiment-ledger.jsonl` containing 3 distinct `candidate_sha256` entries produced by editing `vllm_ascend/ops/mla.py` and running `pytest -q tests/ut/ops/test_mla_prolog.py` for Stage 1, Stage 2, and Stage 3 in sequence.
3. If the Ascend 950DT check remains `refuted` after all 3 stage edits, call `knowledge_query` again with default `mode="reuse"` (`conditions={"framework": "vllm-ascend", "soc": "Ascend950DT"}`, `token_budget=24000`, `goal`, and `bottleneck_summary`) to retrieve deeper structural remedies.
