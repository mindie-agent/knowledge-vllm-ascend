---
schema: mindie-graded-unit/1
id: unit-vllm-ascend-dsv4-v41-query-dequant-and-lighting-indexer-fusion-mutator
revision: 1
tier: L4
category: mutator
evidence_strength: reproduced_empirical
domain_path: vllm-ascend/models/deepseek-v4-v41
title: Fuse DeepSeek-V4/V4.1 query dequant and RoPE split, alias LightningIndexer views, and keep top-k block masks device-resident
conditions:
  framework: vllm-ascend
  model_family:
  - deepseek-v4
  - deepseek-v4.1
  - dsv4
  - dsv3.2
  attention_backend: sparse_mla
  quantization:
  - w8a8
  - w4a8
  - c8
inherits_from:
- unit-vllm-ascend-dsv4-v41-qli-v2-compressor-and-engram-axioms
- unit-vllm-ascend-dsv4-v41-sparse-mla-lightning-indexer-and-mtp-sop
triggers:
  error_signatures:
  - three TensorMove or Memcpy passes per MLA layer around qr dequantization, RMSNorm and the RoPE/NoPE split
  - explicit contiguous clone on q_indexer or k_indexer before npu_lightning_indexer
  - host clamp or item on the top-k valid length between npu_lightning_indexer and sparse MLA
  - invalid GM address when a -1 top-k block id reaches npu_sparse_flash_attention
  - TPOT regresses at 32k or 128k while the 1k cell is unchanged
  keywords:
  - DeepSeek-V4
  - DeepSeek-V4.1
  - SparseMLA
  - QLI-v2
  - LightningIndexer
  - query dequantization
  - RMSNorm
  - RoPE split
  - contiguous view
  - top-k block mask
  - device-resident metadata
  - C8
  - W8A8
  - W4A8
provenance:
  source_type: ablation_design_receipt
  source_locator: docs/evidence/l4-dsv4-query-dequant-lighting-indexer-ablation.json
  source_digest: 1cddab93c1c9ee3fad0eca71a03b5fa69286f2079ed75d1e817a738acfb55ae6
  evidence_refs:
  - docs/evidence/l4-dsv4-query-dequant-lighting-indexer-ablation.json
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/808bb55d9e61d35c1139101ac03f327490c2379febbe012966cd3ec42b2dbfc5/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/50809f1fefde44f4671b682248b8a55c32debd931359dd887585022e0a68fa79/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/edc44803e0feab76d8fbbb587a1b1ac7e8e1f444f3dd21e8cff3f89e177324cf/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/99e97cb3574e1288dd1faa397b490f01e93349bdaee11d3f6909f2e5c383fac4/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/0bc94d24162470fe8ce3fc189b7eb239aa868d074ff53c78706cad0c5c608243/index.md
  experiment_status: not_run_source_only
  supersedes: []
---

# Fuse DeepSeek-V4/V4.1 query dequant and RoPE split, alias LightningIndexer views, and keep top-k block masks device-resident

## Applicability

Apply this mutator to `vllm-ascend` DeepSeek-V4/V4.1 SparseMLA serving on Ascend910B, Ascend910_93 or Ascend950 when all of the following hold:

- the MLA layers use the absorbed decode algebra (`W_UK` folded into `W_Q`, `W_UV` folded into `W_O`) with `kv_lora_rank=512` and `qk_rope_head_dim=64`;
- the attention backend is SparseMLA with a LightningIndexer/QLI-v2 path, and the layer's compression class is C4/CSA (SWA and C128/HCA layers have no indexer and are out of scope);
- the weights are W8A8, W4A8 or C8, and the C8 scale granularity is known (per-token store and per-token load, or a consistent per-tensor pair);
- a baseline has been captured on the same model, weights, TP/DP topology, context lengths, concurrency and graph mode, and at least one `Signals` observation reproduces.

Do not apply it to non-absorbed prefill-only execution, BF16 dense MLA without an indexer, Kimi K3 GQA drafts, or a batch that mixes absorbed and non-absorbed tokens without a path split. The three factors are separable, but factor 1 is decode/absorbed-path only; factors 2 and 3 apply to both chunked prefill and decode.

The transformation is source-only until the ablation gate below runs. The inefficiencies and failures in `Signals` were observed in the DSV4/V4.1 transcripts, but the three-factor combination has not been executed as a controlled baseline/full/leave-one-out experiment. The `reproduced_empirical` strength refers to those reproduced baseline signals, not to an unrun candidate speedup.

## Signals

- **Per-layer query materialization.** The profiler shows three GM round trips around the query path: dequantize `qr` to a temporary, RMSNorm to a second temporary, project to a full `q`, then split into NoPE/RoPE slices with additional writes. The layer's kernel list contains `TensorMove`/`Memcpy` entries whose shapes scale with the token count, and the same layer has no fused dequant-norm-rope kernel.
- **Indexer view clones.** `q_indexer` or `k_indexer` is produced by slicing a fused projection or the paged cache and then calling `.contiguous()`. A `TensorMove`/`Memcpy` appears immediately before `npu_lightning_indexer` or `npu_quant_lightning_indexer`, and the clone size grows with `T` even though the last dimension was already contiguous.
- **Host-side top-k length.** Between `npu_lightning_indexer` and `npu_sparse_flash_attention`, the code calls `.item()`/`.cpu()` on `seqused_k`, `topk` valid length, or `qli_seqused_k.max()`, clamps on the host, and copies the scalar back. Under graph capture this forces a D2H sync and sometimes a recapture; when the clamp is skipped, a `-1` or out-of-range block id reaches the attention kernel.
- **Long-context fault or regression.** The 1k cell is normal, but at 32k-128k the device log shows an `AICore` invalid GM address, a cross-device timeout, or a TPOT regression that scales with the number of selected blocks. The selected count is at or near the full compressed block count, so the sparse path is either inactive or over-selecting.
- **Quantized-path divergence.** The C8/W8A8/W4A8 candidate diverges from BF16 only after the indexer or the latent cache is enabled, and the divergence disappears when the indexer is forced to exhaustive selection. This localizes the fault to the query/indexer scale or the top-k block contract rather than to MoE.

## Transformation

The mutation has three separable factors. Apply them in the order below, keep each as an independently revertible commit or patch, and run the leave-one-out arms in `Verification and ablation`. Do not bundle a numerical change (scale or quantization order) into a view-aliasing factor.

1. **fused_q_qr_dequant_and_rope_split**

   **Baseline.** For each MLA layer, `qr_int8` is dequantized with a per-token scale into a temporary; the temporary is RMSNorm'ed into a second temporary; `q_b_proj` (with `W_UK` already absorbed for decode) writes a full `q`; the NoPE and RoPE slices are split into separate tensors; RoPE writes the 64-wide slice; the final concat writes the packed query again. This is three or more GM read/write passes per layer beyond the unavoidable input and output.

   **Change.** Replace the chain with one fused kernel that reads the int8 `qr` and its per-token scale, dequantizes in FP32, applies the RMSNorm with its FP32 weight, multiplies by the absorbed `W_Q`, and splits the `512`-wide NoPE slice from the `64`-wide RoPE slice in registers. Apply RoPE only to the 64-wide slice and write one packed `[T, H, 576]` query. No temporary `qr`/norm/split tensors are materialized in GM.

   ```python
   # Target fused binding; this is the mutation's new op, not an existing kernel.
   q_packed = torch.ops._C_ascend.fused_q_dequant_rmsnorm_rope_split(
       qr_i8,            # [T, q_lora_rank] int8
       qr_scale,         # [T] FP32, per-token
       rms_weight,       # [q_lora_rank] FP32
       w_q_absorbed,     # [H * (512 + 64), q_lora_rank], W_UK folded in
       cos, sin,         # [T, 64] or the layout's RoPE cache
       eps=1e-6,
       n_heads=H,
       nope_dim=512,
       rope_dim=64,
   )                     # [T, H, 576], last dim contiguous
   ```

   **Invariants.** `w_q_absorbed` contains the `W_UK` absorption for the decode path; the output width is `512 + 64`; RoPE touches only the last 64 elements of each head; accumulation is FP32; `qr_scale` is per token and `rms_weight` is FP32; the output base is 32 B aligned and the last dimension is contiguous. The non-absorbed prefill path keeps its `128 + 64` query and is not routed through this kernel.

   **Expected effect.** Eliminate three intermediate GM passes per layer and one kernel launch chain per layer, reducing decode TPOT and indexer-adjacent latency without changing the mathematical result beyond the unavoidable FP32/quantization ordering.

   **Ablation signature.** Removing this factor restores the dequant, RMSNorm, split and concat `TensorMove` entries and the corresponding TPOT regression; the index scores must stay equal, so a score change indicts the fusion's scale or RoPE slice, not the removed copies.

2. **contiguous_view_alias_for_lightning_indexer**

   **Baseline.** `q_indexer` and `k_indexer` are made contiguous with an explicit `.contiguous()` before `npu_lightning_indexer`, even when the producer already satisfies the operator's stride contract. In a fused QKV layout the clone copies the whole token-major tensor per layer; in the paged-cache path it copies pages that the operator could read in place. The clone is a `TensorMove`/`Memcpy` that grows with batch and context.

   **Change.** Pass a stride-compliant slice view instead of a clone. The view must have `stride(-1) == 1`, a 32 B-aligned base, and an axis-0 stride that the QLI tiling honors (`TND` for the query, `PA_BSND` for the paged key cache). Build the request offsets from the device `query_start_loc`, not from a host list. If the operator rejects the view, keep the copy but move it to the metadata/prepare stage or confine it to prefill, and record the fallback in the experiment receipt.

   ```python
   # q_fused is the producer output; the slice keeps stride(0) and the storage offset.
   q_idx = q_fused[..., :index_dim]        # [T, H, 128]
   assert q_idx.stride(-1) == 1
   assert q_idx.data_ptr() % 32 == 0
   assert q_idx.stride(0) == q_fused.stride(0)

   # k_idx is the PA_BSND paged index-key cache; no clone.
   torch.ops._C_ascend.npu_quant_lightning_indexer(
       q_idx, k_idx, weights, q_scale, k_scale,
       block_table=block_table,   # int32, axis-0 stride preserved
       query_start_loc=query_start_loc,
       seqused_k=seqused_k,
       topk=topk,
       compress_ratio=compress_ratio,
       layout_query="TND",
       layout_key="PA_BSND",
   )
   ```

   **Invariants.** No allocation in the captured forward; `data_ptr()` is unchanged before and after the call; the last dimension is contiguous; the base and the row stride satisfy the kernel's 32 B/512 B alignment contract; the key cache remains `PA_BSND` with its original page strides; the block table remains int32 and axis-0 strided. A copy that is removed but changes index scores is a stride bug, not a valid optimization.

   **Expected effect.** Remove one token-scaled clone per layer and per indexer call, lowering decode TPOT and the device memory bandwidth spent on copies, with bitwise-equal index scores and top-k selections.

   **Ablation signature.** Removing this factor restores the `.contiguous()` `TensorMove` entries and the associated TPOT regression, while the top-k selections remain equal. If the operator raises a stride/contiguity error instead, the fallback boundary applies and the factor is not retained.

3. **device_resident_topk_block_mask_guard**

   **Baseline.** After `npu_lightning_indexer`, the code reads `seqused_k` or the top-k valid length on the host, clamps it, and copies the result back; the `topk_indices` tail still contains `-1`. Under graph capture this is a D2H sync, and an unclamped `-1` or out-of-range block id reaches `npu_sparse_flash_attention`/`fused_infer_attention_score`, where it becomes an illegal GM read.

   **Change.** Keep the selected indices, the valid-length tensor and the block-table guard on the device. Compute the valid count with a device clamp, replace `-1` with a safe allocated dummy block, and pass `seqused_k` so the dummy is excluded from attention. Keep the graph's block table and `seqused_k` as persistent device buffers updated in place; if a host scalar is genuinely required, compute it once in the metadata/prepare stage outside capture and reuse the padded capacity. The QLI/FIA path reads `seqused_k` directly, so do not add a host reduction just to fill `max_seqlen_k`.

   ```python
   valid = torch.clamp(seqused_k, min=0, max=topk)          # [num_reqs], int32, device
   safe_idx = torch.where(topk_indices < 0,
                          torch.zeros_like(topk_indices),  # block 0 is allocated and masked
                          topk_indices)
   block_table.scatter_(1, topk_slots, safe_idx)            # persistent device buffer
   torch.ops._C_ascend.npu_sparse_flash_attention(
       q, k_cache, v_cache,
       block_table=block_table,
       seqused_k=valid,        # the dummy block is outside the valid prefix
       max_seqlen_k=padded_capacity,   # padded, not a host-derived exact length
       sparse_mode=3,
   )
   ```

   **Invariants.** `-1` never reaches AICore; the dummy block is a real allocated block whose contents are excluded by `seqused_k` or the mask; the block-table row length is at least `topk` and covers the padded capacity; `seqused_k` is int32 on device and non-increasing per request when the context shrinks; no `.item()`/`.cpu()` occurs inside the captured step; graph replay observes the updated buffers rather than a stale host snapshot.

   **Expected effect.** Remove the D2H synchronization and the host clamp from the indexer-to-attention handoff, prevent invalid GM reads at long context, and stabilize TPOT when the selected block count changes across steps.

   **Ablation signature.** Removing this factor restores the host clamp and the `-1` tail; the D2H count rises and the adversarial long-context case exposes an invalid GM address or a wrong selected set. A surviving speedup without the invalid-address guard does not validate this factor.

## Fallback boundary

- **Non-absorbed prefill or BF16 dense MLA.** Factor 1 does not apply; keep the `128 + 64` prefill query and the `kv_b_proj` expansion. Factors 2 and 3 still apply only if an indexer is present.
- **Different geometry.** If `kv_lora_rank`, `qk_rope_head_dim`, indexer `head_dim`, `topk`, `compress_ratio`, or the layer's compression class differs, re-derive the widths and the scale binding. Do not reuse 512/64/128/512/4 as a universal tuple.
- **Operator rejects the strided view.** Keep `.contiguous()` at the call site or in the metadata/prepare stage, report the copy and its size, and do not claim factor 2. A forced view that produces wrong scores is a correctness regression.
- **Per-tensor C8 layout.** Factor 3's per-token scale assumption is invalid. Use the per-tensor path consistently at store and load, and do not mix the two granularities.
- **Host scalar still required.** If an operator cannot consume a device valid-length tensor, keep the synchronization outside capture and report it. Do not hide it inside the replay or mute the operator to pass the gate.
- **Different SoC or binding.** A2/A3 and A5 have different QLI tiling keys and bindings (`torch.ops.custom` versus `torch.ops.cann_ops_transformer`). Re-check the available kernel, the supported dtype and the layout before applying any factor.
- **Unrun ablation.** Until the full candidate and all three leave-one-out arms run on the same frozen configuration, keep `provenance.experiment_status: not_run_source_only` and do not report a speedup as established.

## Verification and ablation

**Freeze the comparison.** Use the same model revision, weight descriptor, TP/DP/EP topology, CANN/`torch_npu` versions, `max_model_len`, `max_num_batched_tokens`, graph mode, prompt sets and concurrency for every arm. Record the baseline command line and the baseline run ID. Do not compare the candidate against a historical run from a different service version or cache state.

**Correctness gates.**

- 1k context (below `topk * compress_ratio`): greedy output token-for-token equal to the BF16 dense reference; the indexer selection is exhaustive.
- 32k and 128k contexts (above the threshold): selected count strictly below the total compressed blocks for at least one request; GSM8K or the task's accuracy metric within the pre-registered tolerance; long-context retrieval not below baseline; no repetition or degenerate output.
- Top-k selection parity against a Python reference on synthetic block tables containing `-1`, a partial last block, and a boundary at `topk * compress_ratio`.
- C8/W8A8/W4A8 latent dequantization parity at sampled token positions; no NaN and no scale-buffer overflow.
- No `AICore` invalid GM address, no cross-device timeout, no graph recapture, and no host D2H sync in the captured step.

**Performance gates.** Report TTFT, TPOT p50/p90, ITL p50/p90 and output throughput separately for 1k, 32k and 128k. Report the indexer kernel time and the number of `TensorMove`/`Memcpy` kernels per layer. Repeat each cell at least three times and reject a win smaller than twice the run-to-run spread. A short-context-only win cannot justify the mutation for long-context SparseMLA.

**Leave-one-out arms.**

| Arm | Removed factor | Required discriminating result |
|---|---|---|
| `full_candidate` | none | All correctness gates pass; TPOT and copy/sync counts improve over baseline |
| `minus_q_fusion` | `fused_q_qr_dequant_and_rope_split` | The dequant/RMSNorm/split copies return and decode TPOT regresses; index scores stay equal |
| `minus_view_alias` | `contiguous_view_alias_for_lightning_indexer` | The Q/K `TensorMove` clones return and TPOT regresses; top-k selections stay equal |
| `minus_device_guard` | `device_resident_topk_block_mask_guard` | The host clamp/D2H returns; the adversarial long-context case exposes an invalid block id, a wrong selection, or an invalid GM address |

Run at least the full candidate, the baseline, and the three `minus_*` arms. A factor is retained only if its removal produces the stated discriminating result; otherwise mark it non-retained and keep the weaker observation. Do not promote `experiment_status` to `ablation_verified` unless the full candidate passes every correctness and performance gate and all three arms behave as specified. Until then, report the result as `not_run_source_only` or `single_observation`, never as a measured speedup.

**Reporting skeleton.** For each arm record: run ID, graph mode, context length, concurrency, quantization, TPOT p50/p90, TTFT p50/p90, output throughput, selected-block count, `TensorMove`/`Memcpy` count per layer, host D2H count, accuracy metric, and the first failing device-log line (if any). This is the minimum evidence needed to distinguish a real factor effect from a cache-state or version artifact.
