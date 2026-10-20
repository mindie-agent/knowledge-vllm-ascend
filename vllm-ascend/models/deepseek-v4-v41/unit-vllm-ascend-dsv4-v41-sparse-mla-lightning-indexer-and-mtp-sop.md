---
schema: mindie-graded-unit/1
id: unit-vllm-ascend-dsv4-v41-sparse-mla-lightning-indexer-and-mtp-sop
revision: 1
tier: L2
category: sop
evidence_strength: reproduced_empirical
domain_path: vllm-ascend/models/deepseek-v4-v41
title: DeepSeek-V4/V4.1 SparseMLA, LightningIndexer/QLI-v2, C8 scale, and MTP alignment SOP
conditions:
  framework: vllm-ascend
  model_family:
  - deepseek-v4
  - deepseek-v4.1
  - dsv4
  - dsv3.2
  attention_backend: sparse_mla
inherits_from:
- unit-vllm-ascend-dsv4-v41-qli-v2-compressor-and-engram-axioms
triggers:
  error_signatures:
  - top-k block id -1 reaches npu_sparse_flash_attention and triggers an AICore GM read fault
  - QLI-v2 selects the current partial compressed block or a future block under chunked prefill
  - explicit contiguous or TensorMove clones appear before lightning_indexer or mla_Graded
  - C8 latent cache dequantization mismatches the per-token store scale after W8A8 or W4A8 quantization
  - MTP draft and target disagree on num_accepted_tokens, query_start_loc or block_tables
  - item or cpu on seqused_k or num_accepted_tokens inside the captured step
  - accuracy is normal at 1k but degrades or faults once seq_len exceeds topk times compress_ratio
  keywords:
  - SparseMLA
  - npu_lightning_indexer
  - QLI-v2
  - softmax_scale
  - causal mask
  - topk
  - block_table
  - seqused_k
  - C8
  - W8A8
  - W4A8
  - MTP
  - num_accepted_tokens
  - query_start_loc
  - TPOT
provenance:
  source_type: session_distill
  source_locator: docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/50809f1fefde44f4671b682248b8a55c32debd931359dd887585022e0a68fa79/index.md
  source_digest: 491d49d20724fa41b120b69ad94dbab25f34d74641f298cb0de0f2fbc8c1b9b9
  evidence_refs:
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/50809f1fefde44f4671b682248b8a55c32debd931359dd887585022e0a68fa79/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/808bb55d9e61d35c1139101ac03f327490c2379febbe012966cd3ec42b2dbfc5/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/40fd4789f8c5706d39fce27e2fe6da9b72e38c3afd28b9adbfe4eb4eee2e8a6c/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/a677bfbfcb76c7232fdd04d27da6800d1d154e00f49d8e9b45c1c3fa5aaccfec/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/b25f088ec92d529b589da9b927614f27c3bced925bb6d5f3dbe256652bbc026a/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/bad4bf3370420a4fa68a1300b59acf74e43c293bf54a187d85299b012073e8ed/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/edc44803e0feab76d8fbbb587a1b1ac7e8e1f444f3dd21e8cff3f89e177324cf/index.md
  - docs/evidence/l4-dsv4-query-dequant-lighting-indexer-ablation.json
  supersedes: []
---

# DeepSeek-V4/V4.1 SparseMLA, LightningIndexer/QLI-v2, C8 scale, and MTP alignment SOP

## Scope and preconditions

Use this procedure when a `vllm-ascend` DSV4/V4.1 model runs SparseMLA with a LightningIndexer/QLI-v2 path, optional C8 latent cache, W8A8 or W4A8 weights, and MTP or Eagle-style speculative heads. The procedure assumes the parent axiom's geometry (`kv_lora_rank=512`, `qk_rope_head_dim=64`, indexer `head_dim=128`, known `topk` and `compress_ratio`). Freeze the following before changing code:

- exact model revision, tokenizer, and weight quantization descriptor;
- `vllm-ascend` revision, CANN and `torch_npu` versions, SoC (A2 = Ascend910B, A3 = Ascend910_93, A5 = Ascend950), TP/DP/EP topology;
- attention backend, `kv_cache_dtype`, `max_model_len`, `max_num_batched_tokens`, graph mode;
- one short prompt set (about 1k tokens) and one long set (32k and 128k tokens), with `max_tokens=1` for prefill-only timing and a fixed decode length for TPOT;
- the baseline run ID and the exact command line, so every step can be replayed on the same service.

Do not mix a short-context win with a long-context run, or a BF16 baseline with a quantized candidate. If `topk`, `compress_ratio`, or the layer compression class is unknown, stop and resolve the model config first.

## Procedure

1. **Verify the QLI/QLI-v2 scale, the chunked-prefill causal boundary, and the top-k block-table padding.**

   Locate the scale and top-k constants:

   ```bash
   grep -rn "softmax_scale\|n_heads\*\*-0.5\|topk\|top_k\|compress_ratio" \
     vllm_ascend/models/deepseek_v41/indexer.py \
     vllm_ascend/attention/dsa_v41.py
   ```

   Freeze the effective score scale. The observed V4.1 indexer computes `self.softmax_scale * self.n_heads ** -0.5` in `select()`. Evaluate it once in `__init__` as an FP32 Python scalar and compare the value used by prefill and decode:

   ```python
   import math
   expected = float(cfg.softmax_scale) * float(cfg.index_n_heads) ** -0.5
   got = float(indexer.select_scale)          # map to the installed attribute name
   assert math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12)
   ```

   For chunked prefill, let `p = context_len + local_pos` be the absolute query position and `b = p // compress_ratio` the last compressed block that may contain a visible token. The selected indices must satisfy `0 <= idx <= b`; the partial block at `b` is included only if it contains at least one token with position `<= p`. At a chunk boundary, `context_len` must be the already-cached length, not the scheduler's total prompt length.

   Build an adversarial block table and top-k tensor before trusting the operator:

   ```python
   import math, torch, torch_npu

   def normalize_topk(seq_len, topk=512, ratio=4, block_size=128, max_blocks=4096):
       valid = min(topk, math.ceil(seq_len / ratio))
       idx = torch.full((1, topk), -1, dtype=torch.int32, device="npu")
       idx[0, :valid] = torch.arange(valid, dtype=torch.int32, device="npu")
       safe = torch.where(idx < 0, torch.zeros_like(idx), idx)   # device-side sentinel guard
       table = torch.zeros((1, max_blocks), dtype=torch.int32, device="npu")
       table[0, :valid] = safe[0, :valid]
       return valid, safe, table

   for s in (1, 127, 128, 129, 2047, 2048, 2049, 32768, 131072):
       valid, safe, table = normalize_topk(s)
       assert valid == min(512, (s + 3) // 4)
       assert int(safe.min()) >= 0 and int(safe.max()) < table.shape[1]
   ```

   Invariants: the effective scale is a configuration constant, identical in the indexer and in any reference implementation; `topk_indices` has `[valid..., -1...]` rows; `seqused_k` (or the mask) carries `valid`; the block table is int32 and every id that reaches `npu_sparse_flash_attention` or `fused_infer_attention_score` is within the allocated compressed table; `-1` is normalized to a safe allocated dummy block on device before the call.

   Verification: run the synthetic cases with `ASCEND_LAUNCH_BLOCKING=1`, then replay them under graph capture while changing `positions` and `seqused_k`. Require finite output, the expected valid count, no graph recapture, and no `AICore` invalid-GM or cross-device timeout in the device log. Compare the selected set against a Python reference that masks future blocks and sorts by position.

2. **Remove redundant intermediate materialization before `mla_Graded` and `lightning_indexer`.**

   Find the copies and transposes on the hot path:

   ```bash
   grep -rn "\.contiguous()\|\.transpose(\|\.clone()\|MakeTensorContiguous" \
     vllm_ascend/attention/dsa_v41.py \
     vllm_ascend/models/deepseek_v41/indexer.py
   ```

   Capture one steady-state decode step and one chunked-prefill step with the NPU profiler. In the exported `kernel_details.csv`, group by kernel name and look for `TensorMove`, `Memcpy`, `Transpose` and `Contiguous` entries whose input shape tracks the token count:

   ```bash
   python - <<'PY'
   import pandas as pd
   df = pd.read_csv("kernel_details.csv")   # use the column names of the installed profiler export
   name = next(c for c in df.columns if c.lower() in {"name", "op_name"})
   mask = df[name].astype(str).str.contains("TensorMove|Memcpy|Transpose|Contiguous", regex=True)
   print(df.loc[mask, [name, "Duration(us)"]].head(50))
   PY
   ```

   Replace the clone with a stride-compliant alias when all of these hold: the last dimension is contiguous (`stride(-1) == 1`), the slice base is 32 B aligned, the operator's TND/PA_BSND path honors `stride(0)`, and the request offsets come from `query_start_loc` rather than a Python list:

   ```python
   q_idx = fused_q[..., :index_dim]          # [T, H, 128], view of the fused projection
   assert q_idx.stride(-1) == 1
   assert q_idx.data_ptr() % 32 == 0
   assert q_idx.stride(0) == fused_q.stride(0)
   # pass q_idx directly; do not call .contiguous()
   ```

   Keep the Q RoPE and NoPE views in one fused buffer: RoPE writes only the 64-wide slice, NoPE is the 512-wide slice in the absorbed path (128 in the non-absorbed path), and neither slice may be reallocated per layer.

   Invariants: no allocation and no `.contiguous()` in the captured forward; the input pointer is unchanged from the producer; the base is 32 B aligned and the kernel tiling honors axis-0 stride; if the kernel rejects the view, the copy is moved to the metadata/prepare stage or confined to prefill, and the fallback is reported rather than silently retained in decode.

   Verification: assert `data_ptr()` equality before and after the call, confirm the `TensorMove`/`Memcpy` entries for Q/K disappear from the layer's kernel list, and require bitwise-equal index scores and logits against the clone baseline. A copy that disappears but changes scores is a stride bug, not an optimization.

3. **Bind the C8 latent cache and the W8A8/W4A8 scales to the correct contraction stage.**

   Inspect the cache spec and the projection:

   ```bash
   grep -rn "kv_cache_dtype\|scale_dim\|k_norm\|kv_a_proj_with_mqa\|compress_ratio\|storage_block_size" \
     vllm_ascend/models/deepseek_v41 \
     vllm_ascend/core/kv_cache_interface.py
   ```

   Freeze the split and the scale chain:

   ```python
   out = model.kv_a_proj_with_mqa(x)          # [T, 576] before optional quantization
   latent, rope = out[..., :512], out[..., 512:]
   assert latent.shape[-1] == 512 and rope.shape[-1] == 64
   spec = model.kv_cache_spec
   assert spec.storage_block_size == spec.block_size // spec.compress_ratio
   assert spec.page_size_bytes in (584, 656) or spec.scale_dim > 0
   ```

   Contract to enforce:

   - `k_norm` is an FP32 RMSNorm over the 512-wide latent and runs before C8 quantization; the norm weight is FP32. The per-token quantization scale is computed from the normalized latent and stored in the side scale buffer at the same token slot.
   - Only the 512-wide latent is stored as int8 under C8. The 64-wide RoPE slice follows the positional-cache dtype of the selected 584 B or 656 B layout and is not silently quantized with the latent scale.
   - W8A8 dynamic uses a per-token activation scale times a per-channel weight scale; W4A8 uses the packed int4 weight with its own per-group/per-channel scale. Do not reuse a W8A8 per-channel scale for a W4A8 weight.
   - Store-time and load-time scale granularity must match. A per-token store read with a per-tensor scale is a silent rescale; a per-tensor store read with a per-token scale is an out-of-bounds scale read.
   - `compress_ratio`, `storage_block_size` and `scale_dim` travel in the cache spec; do not bypass `page_size_bytes`/`real_page_size_bytes` with a hand-computed 576 B page.

   Verification: dequantize one sampled token from the C8 cache with its stored scale and compare with the BF16 `kv_a_proj_with_mqa` latent at the same position; require the error to stay inside the quantization bound. Repeat with the wrong granularity and confirm it fails, so the test is discriminating. Run 32k and 128k prompts with C8 on and off; require no NaN, no scale-buffer overflow, and accuracy within the pre-registered tolerance.

4. **Align MTP draft and target metadata without host synchronization.**

   Locate the handoff:

   ```bash
   grep -rn "num_accepted_tokens\|query_start_loc\|block_tables\|seq_lens" \
     vllm_ascend/spec_decode vllm_ascend/attention | head -100
   ```

   Enforce one source of truth for the accepted-token state:

   ```python
   assert num_accepted.dtype == torch.int32 and num_accepted.device.type == "npu"
   assert query_start_loc.numel() == num_reqs + 1
   assert torch.all(query_start_loc[1:] >= query_start_loc[:-1])
   # target: cumsum(num_accepted + 1); draft: cumsum(K), both from the same accepted state
   assert query_start_loc[-1].item() == int((num_accepted + 1).sum())
   assert block_tables.dtype == torch.int32
   ```

   Contract to enforce:

   - `num_accepted_tokens` is a device int32 tensor updated with `index_copy_`, `scatter_` or a masked in-place write. No `.item()` or `.cpu()` in the captured speculative step.
   - The target builds `query_start_loc` from `num_accepted_tokens + 1`; the draft builds its own `query_start_loc` from the same accepted state and the draft depth `K`. The two arrays need not be identical, but they must be derived from one device tensor and use the same `[num_reqs + 1]` layout.
   - The draft and target consume the same `block_tables` content and `seq_lens`. The speculative tail must have at least one free slot in the last allocated block before the draft writes. A rejected draft token must not truncate or rewind the block table; it only reduces the next accepted count.
   - When `num_accepted_tokens == 0`, the next draft step starts from the target's last token. A rejected draft token must never be reused as context.
   - If an operator still needs `max_seqlen_k` as a host scalar, compute it once in the metadata/prepare stage outside capture and reuse the padded capacity. Prefer passing `seqused_k`, which the QLI/FIA path reads directly.

   Verification: run a temperature-0 speculative decode with forced draft rejection (perturb one draft token) and compare accepted tokens, output token ids and logprobs against an eager reference. Capture once, run at least 100 steps, and require zero recaptures and zero host D2H syncs in the captured region. On TP8/DP4, all ranks must agree on the accepted count per request after the collective.

5. **Gate end-to-end accuracy and TPOT at short and long context.**

   Start the service with the frozen configuration:

   ```bash
   vllm serve "$MODEL" --served-model-name dsv4 \
     --max-model-len 131072 --tensor-parallel-size 8 \
     --quantization ascend --kv-cache-dtype int8 --trust-remote-code
   ```

   Run the short accuracy cell (about 1k context, below the sparse threshold):

   ```bash
   lm_eval --model local-completions \
     --model_args "model=dsv4,base_url=http://localhost:8000/v1/completions,tokenizer=$MODEL" \
     --tasks gsm8k --num_fewshot 5 --limit 200 --batch_size 1
   ```

   Run the long-context timing cells with `max_tokens=1` for prefill-only and a fixed decode length for TPOT:

   ```bash
   python benchmarks/benchmark_serving.py --backend vllm \
     --base-url http://localhost:8000 --model dsv4 \
     --dataset-name random --random-input-len 131072 --random-output-len 1 \
     --num-prompts 8 --request-rate 1 --percentile-metrics ttft,tpot,itl
   ```

   Confirm that sparsity is actually active and that the dense-equivalent case is not:

   ```python
   selected = (topk_indices >= 0).sum(dim=-1)
   total = torch.ceil(seq_len / compress_ratio).to(torch.int32)
   assert bool((selected == total).all())          # 1k: exhaustive, dense-equivalent
   assert bool((selected < total).any())           # 32k/128k: sparse selection active
   ```

   Invariants: at 1k, `seq_len <= topk * compress_ratio`, the selection is exhaustive, and greedy output must match a BF16 dense reference token for token. At 32k and 128k, `seq_len > topk * compress_ratio`, the selected count is strictly below the total compressed blocks for at least one request, and accuracy stays within the tolerance fixed before the run (for example GSM8K within 1 point, long-context retrieval not below baseline, no repetition or degenerate output). TPOT p50/p90, TTFT and output throughput are reported separately for short and long cells; never compare across service versions or cache states.

   Verification: repeat each cell at least three times and report the median and spread; reject a claimed win smaller than twice the run-to-run spread. Read back the acceptance length and the indexer kernel time, and require no `AICore` error, no NaN, no graph recapture, and no new host sync. If a long-context failure appears, localize it in this order: top-k block validity, indexer scale/causal boundary, C8 scale, absorption path, then MoE.

## Exit criteria

All five steps must pass on the same frozen configuration. A candidate is not ready when any of the following is true: `-1` or an out-of-range block id reaches the attention operator; a `.contiguous()` or `TensorMove` clone remains on the decode hot path without a reported fallback; the C8 store and load scales use different granularity; `num_accepted_tokens` or `query_start_loc` is read on the host inside the captured step; the 1k greedy output differs from the BF16 dense reference; or the 32k/128k accuracy or TPOT gate is missed. Keep the failing observation as `single_observation` and do not promote the unit or its mutator to `ablation_verified` until the corresponding L4 arms have run.
