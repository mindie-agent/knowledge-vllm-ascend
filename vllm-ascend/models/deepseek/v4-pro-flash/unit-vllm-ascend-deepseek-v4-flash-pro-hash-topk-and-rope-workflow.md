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
revision: 5
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

Use this 3-stage convergence workflow when debugging or tuning `vllm_ascend/models/deepseek_v4/sparse_mla_decode.py` (`sparse_mla_decode_step`) on Ascend 910B (`W8A8` / `C8`) where long-context SparseMLA decode raises `RuntimeError: AICore invalid GM address: negative block id (-1) reached npu_sparse_flash_attention` or exhibits high TPOT latency.

## Stage 1 — Clamp host `max_seqlen_k` and verify `DSV4_BLOCK_SIZES` (`compress_ratio=4`)

1. Ensure `max_seqlen_k = min(int(seqused_k.max().item()), padded_capacity)` before `npu_sparse_flash_attention`.
   - **Limitation**: If `topk_indices` returned by `npu_quant_lightning_indexer` still contains `-1` padding slots when `seqused_k < topk * compress_ratio`, scattering `topk_indices` directly into `block_table` still leaves `-1` in `block_table` and triggers `RuntimeError: AICore invalid GM address: negative block id (-1) reached npu_sparse_flash_attention` (`verdict: "refuted"`).

## Stage 2 — Retain `.contiguous()` on `q_idx` and `k_idx` with stream overlap

1. Keep `q_idx = q[..., :index_dim].contiguous()` and `k_idx = k_idx_cache.contiguous()` before `npu_quant_lightning_indexer`.
   - **Limitation**: Calling `.contiguous()` materializes 2 redundant `TensorMove` clones per layer (`tensor_moves > 0`) and keeps the 4-step unfused `qr` dequantization + RMSNorm + `F.linear` + RoPE split in GM (`verdict: "refuted"`).

## Stage 3 — Host-side `.cpu()` clamp on `topk_indices` with original `seqused_k`

1. Clamp `topk_indices` via `topk_indices.cpu().clamp(min=0)` and pass `seqused_k` unchanged to `npu_sparse_flash_attention`.
   - **Limitation**: Calling `.cpu()` or `.item()` inside the decode step introduces host D2H synchronization (`host_d2h_syncs > 0`), and passing the uncompressed token count `seqused_k` instead of the valid compressed block count causes `npu_sparse_flash_attention` to attend over dummy block `0` and corrupt attention outputs (`verdict: "refuted"`).

## Stagnation & RSI Escalation Rule

When all 3 stages above (`Stage 1`, `Stage 2`, `Stage 3`) fail to resolve both the `AICore invalid GM address` fault and the GM `TensorMove` / D2H sync bottleneck (`3` consecutive `refuted` attempts), record the 3 failed `L3` stage attempts in `logs/experiment-ledger.jsonl` and re-query `knowledge_query` with `ledger_entries` (the 3 `refuted` attempts) + `bottleneck_summary` (or `mode="explore"`) to trigger RSI automatic switching and mount the `L4` operator-fusion mutator.
