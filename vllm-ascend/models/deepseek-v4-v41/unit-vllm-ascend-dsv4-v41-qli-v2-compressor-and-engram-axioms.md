---
schema: mindie-graded-unit/1
id: unit-vllm-ascend-dsv4-v41-qli-v2-compressor-and-engram-axioms
revision: 1
tier: L1
category: axiom
evidence_strength: software_contract
domain_path: vllm-ascend/models/deepseek-v4-v41
title: DeepSeek-V4/V4.1 MLA latent, LightningIndexer/QLI-v2, Compressor/Engram, and absorption invariants
conditions:
  framework: vllm-ascend
  model_family:
  - deepseek-v4
  - deepseek-v4.1
  - dsv4
  - dsv3.2
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
triggers:
  error_signatures:
  - LightningIndexer or QLI-v2 selects a block id outside the paged KV or block table
  - AICore invalid GM address or cross-device timeout after npu_sparse_flash_attention or npu_quant_lightning_indexer
  - C8 latent KV dequantizes to garbage because k_norm or the per-token scale is applied in the wrong order
  - Engram lookup returns different embeddings on different TP or DP ranks for identical token ids
  - decode logits diverge after switching between absorbed and non-absorbed MLA paths in a mixed batch
  - compressor output is consumed before its group-completion flag is set
  keywords:
  - deepseek-v4
  - deepseek-v4.1
  - dsv4
  - dsv3.2
  - MLA
  - kv_lora_rank
  - qk_rope_head_dim
  - LightningIndexer
  - QLI-v2
  - Compressor
  - Engram
  - weight absorption
  - C8
  - W8A8
  - W4A8
  - topk
provenance:
  source_type: session_distill
  source_locator: docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/edc44803e0feab76d8fbbb587a1b1ac7e8e1f444f3dd21e8cff3f89e177324cf/index.md
  source_digest: d688b8e09beac9ce4d77e155f906689494cfeed6a7c851f4682a797772841194
  evidence_refs:
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/edc44803e0feab76d8fbbb587a1b1ac7e8e1f444f3dd21e8cff3f89e177324cf/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/8329f666c3f1ed8962ee75cda1081fa980e25ccfa3c7179f83a7737b37860a28/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/808bb55d9e61d35c1139101ac03f327490c2379febbe012966cd3ec42b2dbfc5/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/bad4bf3370420a4fa68a1300b59acf74e43c293bf54a187d85299b012073e8ed/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/a15c80208a9b43704a944b69a8c518645a2e4e42965218e0c397986cde2ec745/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/99e97cb3574e1288dd1faa397b490f01e93349bdaee11d3f6909f2e5c383fac4/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/0bc94d24162470fe8ce3fc189b7eb239aa868d074ff53c78706cad0c5c608243/index.md
  supersedes: []
---

# DeepSeek-V4/V4.1 MLA latent, LightningIndexer/QLI-v2, Compressor/Engram, and absorption invariants

**Condition.** Applies when a `vllm-ascend` DeepSeek-V4, V4.1, DSV4 or DSV3.2-lineage model is served on Ascend910B, Ascend910_93 (A3) or Ascend950 (A5) with SparseMLA, a LightningIndexer/QLI-v2 path, Compressor state, Engram lookup tables, or MLA weight absorption. The layer contract is per layer, not per model: SWA layers have no indexer, C4/CSA layers instantiate the indexer, and C128/HCA layers are dense attention over compressed history plus a short SWA branch and have no LightningIndexer. The axiom needs a known `kv_lora_rank` (512 in the observed DSV4/V4.1 checkpoints), `qk_rope_head_dim` (64), indexer `head_dim` (128), `topk`, and `compress_ratio`. If any of these is unknown, or a draft model uses a different KV geometry, do not apply these numbers.

**Constraint.**

1. **The MLA latent cache is one 576-element record per token per layer.** `kv_a_proj_with_mqa` emits `[T, kv_lora_rank + qk_rope_head_dim]` = `[T, 512 + 64]`. The paged cache stores the 512-wide latent and the 64-wide RoPE slice as one logical MLA record, not as separate conventional K and V tensors. `kv_b_proj` expands the latent to `[H, 128 + 128]` for non-absorbed prefill; decode consumes the latent directly. A cache spec must carry `compress_ratio` and derive `storage_block_size = block_size // compress_ratio`. The upstream DeepSeek V4/V3.2 per-token record sizes of 584 B and 656 B are deliberate layout variants, not a 576 B rounding error, and the Ascend `scale_dim` and `page_size_bytes`/`real_page_size_bytes` computation must not bypass them.

2. **The indexer cache is a separate allocation from the MLA latent cache.** In the observed C4 geometry the indexer has 64 query heads, one key head, `head_dim=128`, `topk=512`, `compress_ratio=4`, `layout_query=TND`, key cache `PA_BSND`, and causal `sparse_mode=3`. Its Q and K tensors are quantized per token-head (A3 W8A8: INT8 with FP16 scale; 950 MXFP8: FP8 E4M3 with FP32 scale; 950 HiF8: per-tensor with FP32 scale) and the K cache is written through the indexer compressor epilogue. The indexer cache has a different width, dtype, owner and page stride from the 576-wide MLA latent; it cannot alias or be sliced out of the MLA KV buffer. Any materialized index-score buffer is a third, separate allocation: it is indexed by `(token, compressed_block)`, not by `(block, page, head, dim)`, and its row stride must be aligned; never write scores into the key-cache tail or reuse a latent page for them. Every index-cache record, score row, scale record, and block-table base must satisfy the kernel-declared alignment: at least 32 B for vector/UB transactions and the 512 B GM base alignment assumed by the paged-attention ABI. The metadata record is a fixed 1024-int32 core-split buffer and is reusable only across layers whose C4 geometry is identical.

3. **The top-k output is a compressed-block selection with an explicit validity contract.** `npu_quant_lightning_indexer` returns `[T, topk]` int32 indices into the compressed block table, ordered by ascending position, with invalid entries at the tail set to `-1`. The valid count is `min(topk, ceil(seq_len / compress_ratio))`; when `seq_len <= topk * compress_ratio` the selection is exhaustive and the result must be numerically equivalent to dense attention. `-1` is a sentinel, not a block id: normalize it to a safe allocated dummy block on device and carry the valid count in `seqused_k` or an equivalent mask. Never pass `-1`, a raw-token block id, or a block id beyond the allocated compressed table to `npu_sparse_flash_attention` or `fused_infer_attention_score`.

4. **Compressor state is chunk-boundary state, not KV cache.** A C2/C4 compressor pools `compress_ratio` tokens into one compressed entry. The pool kernel writes only completed pairs; incomplete and padding rows are skipped through `[-1, -1]` slot coordinates. `compressor_metadata` accepts 2-D `[T, H]` RoPE inputs (logical TH over `FORMAT_ND`) and returns `[T_compressed, 1, 1, H]`; the main compressor accepts TH and BSH inputs, and the two contracts must not be conflated. A compressed entry may be consumed only after its group-completion flag is set. `index_slots` and `long_slots` are the same masked coordinates later consumed by `update_keys` and `scatter_cache_v2`, so moving the mask earlier must preserve both names and their buffer addresses.

5. **Engram is a replicated hash lookup, not a sharded write.** N-gram hash tables live once in HBM, or in an explicitly selected offload tier; they are not silently replicated per TP rank. Table rows are split by `(ngram_order, hash_head)` bucket with `H = (engram_max_ngram_size - 1) * engram_n_heads`, so a given head is always owned by a fixed rank in the TP x node-local-DP topology. Lookups are read-only gathers. After the node-local DP all-gather of hashes, every TP rank must reconstruct identical embeddings for identical token ids. A one-time host-to-device table upload, if used, must be an asynchronous copy ordered before the first lookup; a per-step H2D gather or a CPU-side permutation that decides the gather order is a defect, not a placement choice. The observed V4.1 path enforced TP8; changing TP/DP without recomputing the bucket-to-rank map changes which rank owns a head and is a correctness change, not a tuning change.

6. **Absorption is a decode-path algebra, not a weight-layout toggle.** Non-absorbed prefill contracts a 128-wide `q_nope` with the `kv_b_proj`-expanded K/V. Absorbed decode folds `W_UK` into `W_Q`, so the query carries the 512-wide latent, and folds `W_UV` into `W_O`, so attention emits `[T, H, 512]` before `W_O`. The query width is therefore 576 (512 + 64) in absorbed decode and 192 (128 + 64) in non-absorbed prefill, while the `W_O` input is 512 versus 128. A mixed prefill/decode batch cannot share one GEMM shape or one scale binding: the chunked-prefill rows carry the 192-wide query and the decode rows the 576-wide query, so either split the batch by path and run two launches, or pad/zero-extend the prefill rows to the absorbed width and mask them. Padding changes the scale binding and must be validated explicitly; it is not free. Quantization scales follow the contraction order: W8A8/W4A8 `kv_a_proj_with_mqa` uses a per-channel or per-group weight scale times a per-token activation scale; the C8 latent store uses the per-token scale captured at store time; `k_norm` runs in FP32 before that quantization. A per-tensor scale must never be substituted for a per-token store scale.

**Rationale.** The latent cache, index cache and top-k block table have different widths, dtypes, ranks and page strides. Treating them as one buffer can produce plausible index scores on short contexts and out-of-range GM reads only when the context grows past `topk * compress_ratio`. A `-1` top-k entry is a mask value that becomes an address if it reaches AICore; no later mask can repair an illegal read. Compressor and Engram are stateful or read-only structures whose correctness is a rank contract, not a numerical tolerance; a different head owner or a stale compressed entry changes tokens rather than adding noise. Absorption changes which weight is contracted at which stage, so a shape-compatible mixed batch with a reused scale is the common silent-corruption path.

**Evidence.** This axiom is a software-contract distillation, not a hardware measurement. Its structural claims come from source and checkpoint audits in the DSV4/V4.1 adaptation transcripts:

- The V4-Flash layer map and indexer geometry: SWA layers 0 and 1; C4 layers 2, 4, ..., 42 (21 layers) instantiate the indexer with `topk=512`; C128 layers 3, 5, ..., 41 do not; query heads 64, key heads 1, `head_dim=128`, `layout_query=TND`, `PA_BSND`, `sparse_mode=3`, and a 1024-int32 metadata record.
- The three quantization paths for the indexer: A3 W8A8 INT8 per-token-head with FP16 scale, 950 MXFP8 FP8 E4M3 per-token-head with FP32 scale, and 950 HiF8 per-tensor with FP32 scale, with distinct `torch.ops.custom` versus `torch.ops.cann_ops_transformer` bindings.
- `compressor_metadata` accepting only 2-D `[T, H]` RoPE inputs and returning `[T_compressed, 1, 1, H]`, while the main compressor supports TH and BSH.
- The C2 compressor path applying `hidden_states.float()` twice (`wkv` and `wgate`) before it was deduplicated, and the pool kernel skipping incomplete rows through `[-1, -1]` coordinates so `out.zero_()` was not required for correctness.
- The V4.1 Engram prefill path containing roughly four D2H and six H2D copies per step (input ids, positions, metadata counts, gathered metadata, ordered ids, order, mask), and the device-side hash/history plus fixed-length TP8 x DP2 communication that removed the host split-size path.
- The cache-spec audit finding that the Ascend `real_page_size_bytes` override bypassed the upstream DeepSeek V4/V3.2 584/656 B special layout while the DSV4 indexer still needs a custom `scale_dim`, and that `compress_ratio` and `storage_block_size` belong to the merged spec.
- The layer-semantics audit that C1 is pure SWA, C4 is sparse plus SWA, and C128/HCA is dense-over-compressed plus SWA.
- The tiling audit that SparseFlashMla and QLI-v2 keys can be trimmed per architecture (A2/A3 versus A5) only while preserving the CSA single-query-head and deterministic-mode branches, and that compression ratio and QLI candidate mode are runtime parameters rather than new tiling keys.

**Boundary.** This unit does not assert that every listed SoC and CANN version supports C8, MXFP8, HiF8, QLI-v2 or a given `topk`. Those are runtime checks in the companion SOP. It does not cover Kimi K3 GQA drafts, conventional GQA/MHA attention, or non-MLA caches. If `kv_lora_rank`, `qk_rope_head_dim`, indexer `head_dim`, `topk`, `compress_ratio`, or the layer's compression class differs, re-derive the geometry and the scale binding rather than reusing these constants.
