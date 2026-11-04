---
category: workflow
conditions:
  framework:
  - vllm-ascend
  - any
domain_path: vllm-ascend/attention/mla-prolog
evidence_strength: reproduced_empirical
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
revision: 2
schema: mindie-graded-unit/1
tier: L3
title: MLA Prolog Projection, RoPE, and KV Cache Write (`vllm_ascend/ops/mla.py`)
  3-Stage Tuning Workflow
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
# MLA Prolog Projection, RoPE, and KV Cache Write (`vllm_ascend/ops/mla.py`) 3-Stage Tuning Workflow

## Trigger and scope

Use this 3-stage convergence workflow when tuning `mla_prolog_rope_kvcache_step` in `vllm_ascend/ops/mla.py` to reduce `clone_repeat_ops` and `select_slice_ops` while keeping BF16/FP32 numerical error `<= 2e-2`.

## Stage 1 — Concatenate `w_q_a` and `w_kv_a` into a single `F.linear` projection

1. Concatenate `w_qkv_a = torch.cat([w_q_a, w_kv_a], dim=0)` while keeping the existing RoPE `.unsqueeze(1).repeat(1, num_heads, 1).clone()` and per-token KV cache loop.
   - **Limitation**: `clone_repeat_ops` remains `2` (`> 0`) and `select_slice_ops` remains `>= 24` (`verdict: "refuted"`).

## Stage 2 — Vectorize RoPE broadcasting while keeping the per-token `slot_mapping` loop

1. Replace `.repeat(1, num_heads, 1).clone()` with `cos.float().unsqueeze(1)` while retaining the Python `for t in range(T):` scatter loop into `kv_cache_nope` and `kv_cache_rope`.
   - **Limitation**: The per-token Python loop still dispatches `select_slice_ops >= 24` (`verdict: "refuted"`).

## Stage 3 — Enable aggressive low-bit QK shift truncation (`torch.round(q_out * 16.0) / 16.0`)

1. Apply 1/16th step quantization (`(torch.round(q_out.float() * 16.0) / 16.0).to(orig_dtype)`) to compress Q outputs.
   - **Limitation**: Degrades BF16 numerical accuracy to `max_abs_err = 0.0625 > 2e-2`, failing the numerical gate (`verdict: "refuted"`).

## Stagnation & RSI Escalation Rule

When all 3 stages above (`Stage 1`, `Stage 2`, `Stage 3`) fail to satisfy both the zero-clone/coalesced dispatch budget and the `max_abs_err <= 2e-2` accuracy gate (`3` consecutive `refuted` attempts), record the 3 failed `L3` stage attempts in `logs/experiment-ledger.jsonl` and re-query `knowledge_query` with `ledger_entries` + `bottleneck_summary` (or `mode="explore"`) to trigger RSI automatic switching and mount the `L4` mutator.
