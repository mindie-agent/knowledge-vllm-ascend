---
category: mutator
conditions:
  framework:
  - vllm-ascend
  - any
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
revision: 2
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

## Transformation & Leave-One-Out Ablation Contract (`vllm_ascend/ops/mla.py`)

Combine all 3 retained factors in `mla_prolog_rope_kvcache_step` while explicitly rejecting the accuracy-destroying decoy factor `enable_lowbit_qk_truncation=False`, and expose `run_ablation_report()`:

```python
def mla_prolog_rope_kvcache_step(
    hidden_states: torch.Tensor,
    w_q_a: torch.Tensor,
    q_a_rms_weight: torch.Tensor,
    w_q_b: torch.Tensor,
    w_kv_a: torch.Tensor,
    kv_a_rms_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_cache_nope: torch.Tensor,
    kv_cache_rope: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    eps: float = 1e-6,
    *,
    fuse_qkv_norm_proj: bool = True,
    vectorized_rope_no_broadcast: bool = True,
    coalesce_kv_cache_scatter: bool = True,
    enable_lowbit_qk_truncation: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_dtype = hidden_states.dtype
    T = hidden_states.shape[0]
    q_lora_rank = w_q_a.shape[0]
    head_dim = qk_nope_head_dim + qk_rope_head_dim

    if fuse_qkv_norm_proj:
        w_qkv = torch.cat((w_q_a, w_kv_a), dim=0)
        qkv_all = F.linear(hidden_states, w_qkv).float()
        q_latent = qkv_all[:, :q_lora_rank]
        kv_latent = qkv_all[:, q_lora_rank : q_lora_rank + kv_lora_rank]
        k_pe_raw = qkv_all[:, q_lora_rank + kv_lora_rank :].to(orig_dtype)
    else:
        q_latent = F.linear(hidden_states.clone(), w_q_a).float()
        kv_all = F.linear(hidden_states.clone(), w_kv_a).float()
        kv_latent = kv_all[:, :kv_lora_rank].clone()
        k_pe_raw = kv_all[:, kv_lora_rank:].clone().to(orig_dtype)

    q_rstd = torch.rsqrt(q_latent.pow(2).mean(dim=-1, keepdim=True) + eps)
    q_normed = (q_latent * q_rstd * q_a_rms_weight.float()).to(orig_dtype)
    kv_rstd = torch.rsqrt(kv_latent.pow(2).mean(dim=-1, keepdim=True) + eps)
    kv_nope_normed = (kv_latent * kv_rstd * kv_a_rms_weight.float()).to(orig_dtype)

    q_all = F.linear(q_normed, w_q_b).view(T, num_heads, head_dim)
    q_nope = q_all[:, :, :qk_nope_head_dim]
    q_pe = q_all[:, :, qk_nope_head_dim:].float()
    half = qk_rope_head_dim // 2

    if vectorized_rope_no_broadcast:
        q_rot_half = torch.cat((-q_pe[..., half:], q_pe[..., :half]), dim=-1)
        q_pe_rot = (q_pe * cos.float().unsqueeze(1) + q_rot_half * sin.float().unsqueeze(1)).to(orig_dtype)
        k_pe_f = k_pe_raw.float()
        k_rot_half = torch.cat((-k_pe_f[..., half:], k_pe_f[..., :half]), dim=-1)
        k_pe_rot = (k_pe_f * cos.float() + k_rot_half * sin.float()).to(orig_dtype)
    else:
        cos_q = cos.float().unsqueeze(1).repeat(1, num_heads, 1).clone()
        sin_q = sin.float().unsqueeze(1).repeat(1, num_heads, 1).clone()
        q_rot_half = torch.cat((-q_pe[..., half:].clone(), q_pe[..., :half].clone()), dim=-1)
        q_pe_rot = (q_pe * cos_q + q_rot_half * sin_q).to(orig_dtype)
        k_pe_f = k_pe_raw.float().clone()
        k_rot_half = torch.cat((-k_pe_f[..., half:].clone(), k_pe_f[..., :half].clone()), dim=-1)
        k_pe_rot = (k_pe_f * cos.float().clone() + k_rot_half * sin.float().clone()).to(orig_dtype)

    q_out = torch.cat((q_nope, q_pe_rot), dim=-1)
    if enable_lowbit_qk_truncation:
        q_out = (torch.round(q_out.float() * 16.0) / 16.0).to(orig_dtype)

    if coalesce_kv_cache_scatter:
        valid = slot_mapping >= 0
        valid_slots = slot_mapping[valid].to(torch.int64)
        kv_cache_nope.view(-1, kv_lora_rank).index_copy_(0, valid_slots, kv_nope_normed[valid])
        kv_cache_rope.view(-1, qk_rope_head_dim).index_copy_(0, valid_slots, k_pe_rot[valid])
    else:
        block_size = kv_cache_nope.shape[1]
        for t in range(T):
            slot = int(slot_mapping[t].item())
            if slot >= 0:
                b_idx = slot // block_size
                b_off = slot % block_size
                kv_cache_nope[b_idx, b_off].copy_(kv_nope_normed[t])
                kv_cache_rope[b_idx, b_off].copy_(k_pe_rot[t])

    return q_out, kv_nope_normed, k_pe_rot


def run_ablation_report() -> dict[str, object]:
    return {
        "retained_factors": [
            "fuse_qkv_norm_proj",
            "vectorized_rope_no_broadcast",
            "coalesce_kv_cache_scatter",
        ],
        "rejected_factors": [
            "enable_lowbit_qk_truncation",
        ],
    }
```
