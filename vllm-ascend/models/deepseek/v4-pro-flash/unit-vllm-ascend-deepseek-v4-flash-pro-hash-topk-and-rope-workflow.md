---
category: workflow
conditions:
  framework:
  - vllm-ascend
  - any
domain_path: vllm-ascend/models/deepseek/v4-pro-flash
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-deepseek-v4-flash-pro-hash-topk-and-rope-workflow
inherits_from:
- unit-vllm-ascend-dsv4-v41-qli-v2-compressor-and-engram-axioms
- unit-vllm-ascend-dsv4-v41-sparse-mla-lightning-indexer-and-mtp-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/deepseek_v4/model.py
  source_digest: 257c415f820f6d203b78b3e1000b5e30c475281db1d99a270c65cba52a3f62ab
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/deepseek_v4/model.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 6
schema: mindie-graded-unit/1
tier: L3
title: DeepSeek-V4/V4.1 SparseMLA decode (`sparse_mla_decode.py`) standard 3-stage
  tuning workflow
triggers:
  error_signatures:
  - AICore invalid GM address negative block id (-1) reached npu_sparse_flash_attention
  - DeepSeek-V4-Pro multi-node W4A8-MTP times out during weight load or HCCL collective
    init
  keywords:
  - DeepSeek-V4
  - DeepSeek-V4.1
  - SparseMLA
  - sparse_mla_decode
  - sparse_mla_decode_step
  - npu_sparse_flash_attention
  - npu_quant_lightning_indexer
  - negative block id
  - invalid GM address
  - QLI-v2
---
# DeepSeek-V4/V4.1 SparseMLA decode (`sparse_mla_decode.py`) standard 3-stage tuning workflow

## Trigger and scope

Use this standard 3-stage workflow when configuring or tuning `vllm_ascend/models/deepseek_v4/sparse_mla_decode.py` (`sparse_mla_decode_step`) on Ascend NPU.

## Stage 1 — Clamp `max_seqlen_k` to `padded_capacity`

1. Pass `max_seqlen_k = padded_capacity` to `npu_sparse_flash_attention` instead of reading `int(seqused_k.max().item())` on the host.

## Stage 2 — Keep explicit `.contiguous()` on `q_idx` and `k_idx`

1. Materialize contiguous buffers `q_idx = q_packed[..., :index_dim].contiguous()` and `k_idx = k_idx_cache.contiguous()` before calling `npu_quant_lightning_indexer`.

## Stage 3 — Clamp `topk_indices` with `torch.clamp(topk_indices, min=0)` and pass `torch.clamp(seqused_k, max=topk)`

1. Clamp negative `topk_indices` via `torch.clamp(topk_indices, min=0)` and pass `seqused_k = torch.clamp(seqused_k.to(torch.int32), max=topk)` to `npu_sparse_flash_attention`.
