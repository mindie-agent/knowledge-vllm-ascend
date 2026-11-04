---
category: sop
conditions:
  framework:
  - vllm-ascend
  - any
domain_path: vllm-ascend/models/deepseek-v4-v41
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-dsv4-v41-sparse-mla-lightning-indexer-and-mtp-sop
inherits_from:
- unit-vllm-ascend-dsv4-v41-qli-v2-compressor-and-engram-axioms
provenance:
  evidence_refs:
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/50809f1fefde44f4671b682248b8a55c32debd931359dd887585022e0a68fa79/index.md
  source_digest: 491d49d20724fa41b120b69ad94dbab25f34d74641f298cb0de0f2fbc8c1b9b9
  source_locator: docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/50809f1fefde44f4671b682248b8a55c32debd931359dd887585022e0a68fa79/index.md
  source_type: session_distill
  supersedes: []
revision: 3
schema: mindie-graded-unit/1
tier: L2
title: DeepSeek-V4/V4.1 SparseMLA service configuration, C8 latent scale, and MTP
  alignment SOP
triggers:
  error_signatures:
  - C8 latent cache dequantization mismatches the per-token store scale after W8A8
    or W4A8 quantization
  - MTP draft and target disagree on num_accepted_tokens, query_start_loc or block_tables
  keywords:
  - SparseMLA
  - QLI-v2
  - softmax_scale
  - C8
  - W8A8
  - W4A8
  - MTP
---
# DeepSeek-V4/V4.1 SparseMLA service configuration, C8 latent scale, and MTP alignment SOP

## Scope and preconditions

Use this procedure when a `vllm-ascend` DSV4/V4.1 model runs SparseMLA with optional C8 latent cache, W8A8 or W4A8 weights, and MTP speculative heads. The procedure assumes the parent axiom's geometry (`kv_lora_rank=512`, `qk_rope_head_dim=64`, indexer `head_dim=128`, known `topk` and `compress_ratio`).

## Procedure

1. **Verify the QLI/QLI-v2 score scale.**
   Evaluate `float(cfg.softmax_scale) * float(cfg.index_n_heads) ** -0.5` once in `__init__` as an FP32 Python scalar and compare prefill and decode configurations.

2. **Bind the C8 latent cache and the W8A8/W4A8 scales to the correct contraction stage.**
   - `k_norm` is an FP32 RMSNorm over the 512-wide latent and runs before C8 quantization; the norm weight is FP32.
   - Only the 512-wide latent is stored as int8 under C8 (`page_size_bytes` 584 B or 656 B).

3. **Gate end-to-end service launch.**
   Start `vllm serve` with `--max-model-len 131072 --tensor-parallel-size 8 --quantization ascend --kv-cache-dtype int8 --trust-remote-code`.
