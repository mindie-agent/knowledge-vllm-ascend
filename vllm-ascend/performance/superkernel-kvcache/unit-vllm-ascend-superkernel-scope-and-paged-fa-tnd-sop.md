---
category: sop
conditions:
  execution_mode:
  - ge_graph
  - npugraph_ex
  - eager
  kv_cache_mode:
  - paged_attention_tnd
  - mla_absorb_tnd_ntd
  - sliding_window_tnd
  npu_model:
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - cann/cann-recipes-infer
domain_path: vllm-ascend/performance/superkernel-kvcache
evidence_strength: software_contract
id: unit-vllm-ascend-superkernel-scope-and-paged-fa-tnd-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-superkernel/SKILL.md#L12-L158
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-kvcache/SKILL.md#L34-L275
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/utils/common_utils.py#L59-L64
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/utils/graph_utils.py#L58-L67
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/core/kv_cache/cache_utils.py#L471-L580
  experiment_status: not_run_source_only
  source_digest: 2678e4032732c8c91599d89b4ff256323b813983cfa0bb96b68b9f78cea4608e
  source_locator: cann/cannbot-skills:f305e200aa4b1e1fed73cd21ff1f083b8df25a1e:model/model-infer-kvcache/SKILL.md
  source_type: cannbot_skill_and_source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Atlas A3 SuperKernel binary fusion scope and Paged Attention TND/TND_NTD KV-cache contract SOP
triggers:
  error_signatures:
  - SuperKernel scope enabled during dynamic-shape prefill or on unsupported eager execution path
  - FA v1 and FA v2 parameter name mismatch silently resets softmax scale to 1.0 or drops actual_seq_kvlen
  - MLA absorb decode uses sparse_mode=3 instead of sparse_mode=0 or mismatches num_key_value_heads=1
  - Prefill or decode corrupts KV cache because kv_len is incremented inside per-layer attention
  keywords:
  - SuperKernel
  - superkernel_scope
  - super_kernel
  - stream-fusion
  - dcci_disable_on_kernel
  - Paged Attention
  - TND
  - TND_NTD
  - PA_NZ
  - npu_fused_infer_attention_score
  - npu_fused_infer_attention_score_v2
  - npu_kv_rmsnorm_rope_cache
  - sparse_mode
  - slot_mapping
  - block_table
---

# Atlas A3 SuperKernel binary fusion scope and Paged Attention TND/TND_NTD KV-cache contract SOP

## Applicability & Scope

Use this SOP when configuring Atlas A3 (`Ascend910_93`) SuperKernel operator binary fusion in decode graphs or migrating LLM/MoE/MLA attention layers to Paged Attention (`TND` / `TND_NTD`) with `npu_fused_infer_attention_score` (FA v1) or `npu_fused_infer_attention_score_v2` (FA v2). This procedure enforces source-verified API and tensor-layout contracts; it does not claim measured speedups without a paired benchmark receipt.

---

## Step 1 — Enforce SuperKernel Decode-Only & Hardware Preconditions

1. **Verify execution mode and hardware**:
   - Standard `torchair.scope.super_kernel(scope, options)` (`superkernel_scope(enable, label, option="stream-fusion=1")`) requires Atlas A3 (`Ascend910_93`), PyTorch + TorchAir `exe_mode="ge_graph"`, and a static decode graph. When `exe_mode == "npugraph_ex"`, SuperKernel optimization is passed via backend compile options (`"super_kernel_optimize": True, "super_kernel_optimize_options": {"dcci_disable_on_kernel": [".*"]}`).
   - **Never** enable SuperKernel during dynamic-shape Prefill (`enable_superkernel and not is_prefill` must guard the scope).
2. **Scope granularity progression**:
   - Start with the narrowest self-contained bottleneck scope (`attention_only` for dense Transformers or `moe_only` for MoE layers) before expanding to `decode_layers`.
   - Exclude Python control flow (`if`/`for` on tensor values), dynamic tensor allocations, and operators with unsupported tiling-sink semantics from the fused scope.

---

## Step 2 — Maintain Paged KV Address Invariants (`block_table`, `slot_mapping`, `kv_len`)

1. **Separate write addressing (`slot_mapping`) from read addressing (`block_table`)**:
   - Physical paged cache storage has logical shape `[total_num_blocks, block_size, num_kv_heads_per_rank, head_dim]`, where `num_kv_heads_per_rank = max(num_kv_heads // attn_tp_size, 1)` (`1` for MLA latent KV).
   - `slot_mapping` (`int32`, 1D packed token length) addresses **cache writes** (`npu_scatter_nd_update_` for MHA/GQA/SWA or `npu_kv_rmsnorm_rope_cache` / `npu_mla_prolog_v3` for MLA):
     $$\text{slot}(b, \text{seq\_pos}) = \text{block\_table}[b, \lfloor\text{seq\_pos} / \text{block\_size}\rfloor] \times \text{block\_size} + (\text{seq\_pos} \bmod \text{block\_size})$$
   - `block_table` (`[batch_size, max_blocks_per_seq]`, `int32`) addresses **FA cache reads**. In multi-attention-type models (`FullAttention` + `SlidingWindow`), index both dicts by `self.attn_type` (`block_table[self.attn_type]`, `slot_mapping[self.attn_type]`).
2. **Keep `kv_len` read-only across layers within a step**:
   - `kv_len` is owned by the runner/scheduler and passed identically to every Transformer layer during a step; increment `kv_len += 1` only once after the full decode step completes. Mutating `kv_len` inside an attention layer shifts write slots across layers and silently corrupts prefill/decode outputs.

---

## Step 3 — Align FA v1 / FA v2 Signatures, `PA_NZ` Views, and `sparse_mode` Masks

1. **Do not mix FA v1 and FA v2 parameter names** (Python keyword mismatches silently fall back to default `scale=1.0` or head counts):

   | Semantic Role | FA v1 (`npu_fused_infer_attention_score`) | FA v2 (`npu_fused_infer_attention_score_v2`) | Contract Pitfall |
   | --- | --- | --- | --- |
   | Softmax scale | `scale` | `softmax_scale` | Passing wrong name leaves scale at `1.0` (for YaRN with `mscale_all_dim`, must be $d^{-0.5} \times \text{mscale}^2$) |
   | Query heads | `num_heads` | `num_query_heads` | Must match per-rank Q head count |
   | KV heads | `num_key_value_heads` | `num_key_value_heads` | Must equal `1` on MLA absorb path |
   | Query sequence lengths | `actual_seq_lengths` | `actual_seq_qlen` | Under `TND`, pass cumulative `actual_seq_lengths_cu_q` |
   | KV sequence lengths | `actual_seq_lengths_kv` | `actual_seq_kvlen` | Under `TND` prefill pass cumulative `cu_kv`; under decode pass direct per-request `kv_len` |
   | Quantized KV scales | `antiquant_mode` / `antiquant_scale` | `dequant_scale_key` / `dequant_scale_value` + `query_quant_mode` | Never cross-wire v1/v2 quant kwargs |

2. **Match physical KV tensor views to `input_layout`**:
   - **Standard MHA/GQA (`input_layout="TND"`)**: Reshape 4D `[bn, bs, num_kv_heads_per_rank, head_dim]` cache to 3D `[bn, bs, num_kv_heads_per_rank * head_dim]` before passing to FA; keep Q as 3D `[TotalTokens, N, D]`.
   - **MLA Absorb (`input_layout="TND_NTD"`, `is_mla_backend=True`)**: Latent KV is not sharded along `attn_tp_size`. Write `nope_cache` and `rope_cache` (`rope_dim = 64`, `query_dim` in `{128, 512}`) with `cache_mode="PA_NZ"` and view physical caches as 5D Fractal NZ `[bn, 1, D // NZ_DIM, bs, NZ_DIM]` (`NZ_DIM = 16` for BF16/FP16, `32` for INT8) before calling FA.
3. **Enforce `sparse_mode` and `[2048, 2048]` boolean `atten_mask` invariants**:
   - When `sparse_mode` is `3` (causal) or `4` (sliding-window band), `atten_mask` must be a compressed `[2048, 2048]` tensor of `torch.bool` (or `int8`/`uint8`, never float), regardless of `max_position_embeddings`.
   - **Standard LLM TND PA**: Use `sparse_mode=3` with the `[2048, 2048]` causal mask for both Prefill and Decode.
   - **SlidingWindow layers**: Keep `sparse_mode=4`, `pre_tokens=sliding_window`, `next_tokens=0` in both Prefill and Decode; never relax Decode to `sparse_mode=0`.
   - **MLA Absorb**: Use `sparse_mode=3` + causal mask in Prefill, and switch to `sparse_mode=0` + `atten_mask=None` in single-token Decode (`q_len=1`).

## Verification & Rollback

1. Verify `KVCacheManager` initialization (no placeholder `torch.Tensor([])` left unallocated), first-step decode `actual_seq_lengths_kv == prompt_len + 1`, and block reclamation after request completion.
2. For SuperKernel, compare decode latency and output tokens with `enable_superkernel: False` vs `True` on the same warmup-excluded steady window; roll back the scope label immediately if any GE compile or numeric divergence appears.
