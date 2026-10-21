---
category: workflow
conditions:
  component: context_parallel
  framework: vllm-ascend
domain_path: vllm-ascend/attention/context-parallel/ring-ulysses-dcp
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-ring-ulysses-dcp-kv-partition-and-lse-workflow
inherits_from:
- unit-vllm-ascend-hccl-and-dp-collective-invariants
- unit-vllm-ascend-attention-v1-mla-dsa-sfa-fa3-routing-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/attention/context_parallel/common_cp.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/attention/context_parallel/mla_cp.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/attention/context_parallel/dsa_cp.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/attention/context_parallel/sfa_cp.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/worker/dcp_utils.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/context_parallel.md
  source_digest: 55b44b753cfe25794f0a9a42463950bd71ec59d2b4b44f0f6bf4e4c1e46bdf24
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/attention/context_parallel/common_cp.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Context Parallelism (PCP Ring, Ulysses, DCP, DSA-CP) KV partitioning and LSE
  log-sum-exp combine workflow
triggers:
  error_signatures:
  - long-context output diverges when pcp_size or dcp_size is greater than 1
  - LSE log-sum-exp reduction produces NaN or Inf on empty KV shards in Decode Context
    Parallelism
  - enable_dsa_cp fails or deadlocks when chunked prefill mixes requests of unequal
    sequence lengths
  keywords:
  - context_parallel
  - pcp_size
  - dcp_size
  - enable_dsa_cp
  - mla_cp
  - sfa_cp
  - dsa_cp
  - LSE
  - softmax_lse
  - dcp_utils
---
# Context Parallelism (PCP Ring, Ulysses, DCP, DSA-CP) KV partitioning and LSE log-sum-exp combine workflow

## Trigger and scope

Use this workflow when configuring or debugging Prefill Context Parallelism (`pcp_size`), Decode Context Parallelism (`dcp_size`), or DeepSeek/GLM Sparse Context Parallelism (`enable_dsa_cp: true`) across `vllm_ascend/attention/context_parallel/` (`common_cp.py`, `attention_cp.py`, `mla_cp.py`, `dsa_cp.py`, `sfa_cp.py`) and `vllm_ascend/worker/dcp_utils.py`.

## Stage 1 — Partition sequence and KV cache across PCP and DCP groups

1. **Distinguish PCP vs DCP (`docs/source/developer_guide/Design_Documents/context_parallel.md`):**
   - **Prefill Context Parallelism (`pcp_size`):** Splits long prompt sequences across `pcp_size` ranks during prefill using head-tail balanced zig-zag chunking (`pcp_manager.py`) so every PCP rank performs an equal amount of causal triangle compute.
   - **Decode Context Parallelism (`dcp_size`):** Splits the historical KV cache blocks across `dcp_size` ranks within the TP group so each rank stores only `1 / dcp_size` of the KV blocks for decode (`dcp_kv_cache_interleave_size`).
   - **DSA/SFA Context Parallelism (`enable_dsa_cp`):** Shards sequence/KV tokens across TP/CP ranks during sparse indexer scoring and MLA compressor updates (`dsa_cp.py`, `sfa_cp.py`).
2. **Verify divisibility and block interleaving:**
   - Ensure `tp_size % dcp_size == 0` and that for GQA models `num_kv_heads` replication/partitioning aligns with `dcp_size`. For MLA/DSA/SFA (`num_kv_heads == 1`), KV blocks are interleaved across DCP ranks at granularity `dcp_kv_cache_interleave_size` (typically `block_size`).

## Stage 2 — Audit partial attention and FP32 Log-Sum-Exp (LSE) reduction

1. **Compute local attention with `softmax_lse` output (`common_cp.py`, `dcp_utils.py`):**
   - Each CP/DCP rank computes local attention output $O_i$ and local log-sum-exp $L_i = \log \sum \exp(S_i)$ in `float32`.
   - Combine partial outputs across the CP/DCP process group using the numerically stable LSE rescaling identity:
     $$L_{	ext{global}} = \log \sum_i \exp(L_i - L_{\max}) + L_{\max}, \quad O_{	ext{global}} = \sum_i \exp(L_i - L_{	ext{global}}) O_i$$
2. **Guard empty local KV shards (`seq_len_local == 0`):**
   - When a short request has zero KV blocks assigned to a given DCP rank, its local $L_i$ must be initialized to $-\infty$ (and $\exp(-\infty - L_{	ext{global}}) = 0$) in FP32 so it contributes zero weight without producing `0 * NaN` during LSE reduction.

## Stage 3 — Validate CP parity against single-sequence eager oracle

1. **Run exact logit parity check:**
   - Compare `pcp_size=1, dcp_size=1` against `pcp_size > 1, dcp_size > 1` (or `enable_dsa_cp=true`) on prompts spanning block boundaries (`127, 128, 129, 4096, 32768` tokens). Require cosine similarity `> 0.999` and identical greedy top-1 tokens.
