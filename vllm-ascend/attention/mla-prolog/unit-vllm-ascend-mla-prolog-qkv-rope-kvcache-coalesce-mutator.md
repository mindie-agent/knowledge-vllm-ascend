---
category: mutator
conditions:
  framework:
  - vllm-ascend
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
  - Ascend950DT
domain_path: vllm-ascend/attention/mla-prolog
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-mla-prolog-qkv-rope-kvcache-coalesce-mutator
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-mla-prolog-single-proj-and-lowbit-qk-workflow
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/mla.py
  experiment_status: ablation_verified
  source_digest: 88a81f2b820f6d203b78b3e1000b5e30c475281db1d99a270c65cba52a3f62ab
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/mla.py
  source_type: ablation_design_receipt
  supersedes: []
revision: 5
schema: mindie-graded-unit/1
tier: L4
title: Fuse MLA prolog Q/KV projections, vectorize RoPE without broadcast clones,
  and coalesce paged KV cache scatter (`vllm_ascend/ops/mla.py`)
triggers:
  error_signatures:
  - MLA prolog numerical check failed max_abs_err=0.0625 > 2e-2
  - Default path still performs redundant clone/repeat allocations
  keywords:
  - mla_prolog_rope_kvcache_step
  - run_ablation_report
  - fuse_qkv_norm_proj
  - vectorized_rope_no_broadcast
  - coalesce_kv_cache_scatter
  - enable_lowbit_qk_truncation
  - vllm_ascend/ops/mla.py
  - kv_cache_nope
  - kv_cache_rope
  - slot_mapping
---
# Fuse MLA prolog Q/KV projections, vectorize RoPE without broadcast clones, and coalesce paged KV cache scatter (`vllm_ascend/ops/mla.py`)

## Architectural Specification (`Ascend 950DT`)

On `Ascend 950DT`, standard single-factor MLA prolog tuning fails because separate `w_q_a`/`w_kv_a` GEMMs, `.repeat()`/`.clone()` RoPE broadcasts, per-slot Python scatter loops, and low-bit QK rounding violate both the dispatch budget (`linear_ops <= 2`, `clone_repeat_ops == 0`, `select_slice_ops <= 12`) and the BF16 accuracy bound (`max_abs_err <= 2e-2`).

Implement the three coupled factors in `mla_prolog_rope_kvcache_step` while rejecting the accuracy-destroying decoy factor `enable_lowbit_qk_truncation`:

1. **`fuse_qkv_norm_proj` (`bool = True`) — Single Concatenated Down-Projection & Vectorized FP32 RMSNorm**:
   - **When `True`**: Let `q_lora_rank = w_q_a.shape[0]` (which may differ from `kv_lora_rank`, e.g., `q_lora_rank=32, kv_lora_rank=24`). Concatenate `w_qkv = torch.cat((w_q_a, w_kv_a), dim=0)` and execute a single GEMM `qkv_all = F.linear(hidden_states, w_qkv).float()` (so the entire step executes only `2` `F.linear` operations total: `w_qkv` and `w_q_b`). Slice `q_latent = qkv_all[:, :q_lora_rank]`, `kv_latent = qkv_all[:, q_lora_rank : q_lora_rank + kv_lora_rank]`, and `k_pe_raw = qkv_all[:, q_lora_rank + kv_lora_rank :].to(orig_dtype)`. Compute vectorized FP32 RMSNorm over `dim=-1, keepdim=True` for `q_latent` and `kv_latent` without any Python token loops or `.clone()` calls.
   - **When `False` (ablation arm `minus_qkv_norm_proj_fusion`)**: Project `w_q_a` and `w_kv_a` with two separate `F.linear` calls (`linear_ops == 3`, higher `tensor_ops`).

2. **`vectorized_rope_no_broadcast` (`bool = True`) — Stride-0 View RoPE without `.repeat()` or `.clone()`**:
   - **When `True`**: Split `q_all = F.linear(q_normed, w_q_b).view(T, num_heads, head_dim)` into `q_nope = q_all[:, :, :qk_nope_head_dim]` and `q_pe = q_all[:, :, qk_nope_head_dim:].float()`. For half-rotation (`half = qk_rope_head_dim // 2`), rotate `q_pe` using `cos.float().unsqueeze(1)` and `sin.float().unsqueeze(1)` (stride-0 broadcast over `num_heads`), and rotate 2D `k_pe_raw.float()` directly using 2D `cos.float()` and `sin.float()` without expanding across `num_heads` (`clone_repeat_ops == 0`).
   - **When `False` (ablation arm `minus_vectorized_rope`)**: Materialize repeated/cloned `cos` and `sin` buffers across `num_heads` via `.unsqueeze(1).repeat(1, num_heads, 1).clone()` (`clone_repeat_ops > 0`).

3. **`coalesce_kv_cache_scatter` (`bool = True`) — Flat Paged KV Cache `index_copy_`**:
   - **When `True`**: Filter valid token slots via `valid = slot_mapping >= 0` and `valid_slots = slot_mapping[valid].to(torch.int64)`. Scatter `kv_nope_normed[valid]` into `kv_cache_nope.view(-1, kv_lora_rank)` and `k_pe_rot[valid]` into `kv_cache_rope.view(-1, qk_rope_head_dim)` via `index_copy_(0, valid_slots, ...)` (`select_slice_ops <= 12`).
   - **When `False` (ablation arm `minus_coalesced_kv_scatter`)**: Fall back to the per-token Python loop indexing `kv_cache_nope[b_idx, b_off]` (`select_slice_ops > 12`).

4. **`enable_lowbit_qk_truncation` (`bool = False`) — Rejected Decoy Factor**:
   - **When `False` (default)**: Do NOT apply `torch.round(... * 8.0) / 8.0` or `16.0` truncation to `q_nope`, `kv_nope_normed`, or `q_out` (`max_abs_err <= 2e-2`).
   - **When `True` (ablation arm `plus_decoy_lowbit_truncation`)**: Quantize `q_out = (torch.round(q_out.float() * 16.0) / 16.0).to(orig_dtype)` (`max_abs_err > 2e-2`).
   - In `run_ablation_report()`, return `"retained_factors": ["fuse_qkv_norm_proj", "vectorized_rope_no_broadcast", "coalesce_kv_cache_scatter"]` and `"rejected_factors": ["enable_lowbit_qk_truncation"]`.
