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
revision: 3
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

1. Replace the per-token Python RMSNorm loops (`for idx in range(num_tokens)`) with batch RMSNorm (`torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)`) while keeping separate `F.linear(hidden_states, w_q_a)` and `F.linear(hidden_states, w_kv_a)` projections.

## Stage 2 — Expand `cos` and `sin` across heads with `.unsqueeze(1).repeat(1, num_heads, 1)`

1. Replace the per-head Python loop (`for h_idx in range(num_heads)`) by repeating `cos` and `sin` across `num_heads` via `.unsqueeze(1).repeat(1, num_heads, 1)` before applying `_rotate_half_last_dim`.

## Stage 3 — Retain low-bit QK step rounding (`1/16` step) with block-offset KV cache writes

1. Use `1/16` quantization rounding (`torch.round(x * 16.0) / 16.0`) on `q_out` (`enable_lowbit_qk_truncation=True`) and write `kv_cache_nope[blk_idx, blk_off]` per valid slot.
