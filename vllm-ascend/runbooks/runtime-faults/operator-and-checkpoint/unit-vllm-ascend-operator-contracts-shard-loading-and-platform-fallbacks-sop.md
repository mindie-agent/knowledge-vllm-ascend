---
category: sop
conditions:
  framework: vllm-ascend
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - vllm-ascend-idle-dp
domain_path: vllm-ascend/runbooks/runtime-faults/operator-and-checkpoint
evidence_strength: software_contract
id: unit-vllm-ascend-operator-contracts-shard-loading-and-platform-fallbacks-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/fia_actual_seq_lengths_kv_backend_type.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/fia_qlens_block_table_num_reqs_batch_contract.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/npu_moe_init_routing_integer_contract.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/mc2_fused_comm_op_compatibility_fallback.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/shard_local_index_vocab_padding_mtp_layer.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/shared_fused_op_default_spec_mismatch.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/triton_grid_coredim_uint16_overflow.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/ascendc_kernel_unsigned_underflow_invalid_row.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/graphmode_padding_stale_position_gatherv3_out_of_range.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/vendored_custom_op_vs_cann_samename_op_conflict.md
  source_digest: 14aed78ad41ce8d9ee6735d3ab413984ffae79a819ce318257577294ba290e85
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/runbooks/runtime/fia_actual_seq_lengths_kv_backend_type.md
  source_type: cannbot_runbook_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Runtime fault triage Part 2: FIA/MoE operator integer & batch contracts, MC2
  fallbacks, Triton uint16 grid overflow, and sharded checkpoint indexing'
triggers:
  error_signatures:
  - npu_fused_infer_attention_score fails type check on actual_seq_lengths_kv between
    torchair GE graph and eager/npugraph_ex
  - npu_moe_init_routing_v2 raises dtype/type error when topk_ids is int64 or active_expert_range
    is tensor instead of Python int list
  - Triton-Ascend kernel launch fails or truncates blocks when grid[0] exceeds CoreDim
    uint16 ceiling (65535)
  - Graph-mode padded tail tokens with stale positions trigger GatherV3 out-of-bounds
    AICore exception
  keywords:
  - fia_actual_seq_lengths_kv
  - fia_qlens_block_table
  - npu_moe_init_routing_v2
  - active_expert_range
  - mc2_fused_comm_op
  - triton_grid_coredim_uint16
  - GatherV3
  - padding_idx
---
# Runtime fault triage Part 2: FIA/MoE operator integer & batch contracts, MC2 fallbacks, Triton uint16 grid overflow, and sharded checkpoint indexing

## Applicability and symptom scope

Use this SOP when triaging operator shape/type assertion failures, AICore out-of-bounds faults (`GatherV3` / unsigned underflow), Triton grid launch limits, and sharded weight loading errors in `vllm-ascend`.

## Procedure

1. **Enforce `npu_fused_infer_attention_score` (FIA) backend-specific type and batch contracts (`fia_actual_seq_lengths_kv_backend_type.md`, `fia_qlens_block_table_num_reqs_batch_contract.md`).**
   - **Backend Type Split:** `actual_seq_lengths` and `actual_seq_lengths_kv` must be a **host `list[int]`** under `torch.ops.npu` eager and `npugraph_ex`, but must be a **device `torch.Tensor` (`int64`/`int32`)** under `torchair` GE graph mode.
   - **Batch Dimension Equality:** The first dimension of `block_table` (`block_table.shape[0]`), `len(actual_seq_lengths)`, and `len(actual_seq_lengths_kv)` must all equal the active request count `num_reqs` (or the captured bucket batch size $B$ when padded—never pass the un-sliced max-buffer `block_table[:max_reqs]` when `actual_seq_lengths` has length $B < \text{max\_reqs}$, and do not let SP sequence padding inflate `block_table` rows).

2. **Enforce `npu_moe_init_routing_v2` integer contracts and MC2 platform fallbacks (`npu_moe_init_routing_integer_contract.md`, `mc2_fused_comm_op_compatibility_fallback.md`).**
   - `npu_moe_init_routing_v2` requires `topk_ids` in **`torch.int32`** (cast `topk_ids.to(torch.int32)` if `torch.topk` returns `int64`) and `active_expert_range=[start_id, end_id]` as a **Python `list[int]`**, never a Tensor.
   - Check MC2 hardware compatibility before enabling fused communication ops: Atlas A2 caps expert count per MC2 dispatch/combine domain lower than Atlas A3, `npu_mm_reduce_scatter_base` / `npu_all_gather_base_mm` have specific TP/shape constraints, and `fullmesh_v2` requires fallback when local expert count or topology violates its mesh invariant.

3. **Zero-clamp padded graph positions and avoid unsigned integer underflow in kernels (`graphmode_padding_stale_position_gatherv3_out_of_range.md`, `ascendc_kernel_unsigned_underflow_invalid_row.md`, `triton_grid_coredim_uint16_overflow.md`).**
   - In graph mode, always zero-fill the padded tail of `positions` (`positions[num_valid_tokens:] = 0`) before replay; leaving stale large position indices from a prior long-context request in the padded tail causes downstream RoPE `GatherV3` index-out-of-bounds AICore crashes.
   - In AscendC kernels, compute remaining valid lengths `(seq_len - offset)` using signed `int32_t` before clamping with `Max(rem, 0)` (on `uint32_t`, a negative difference wraps to a huge positive integer and `Max(x, 0)` is a no-op, triggering out-of-bounds GM reads).
   - In Triton-Ascend kernels, ensure `grid[0] <= 65535` (`uint16` `CoreDim` hardware limit); fold large token/batch grids into a 2D grid or persistent grid-stride loop over `num_aicore`.

4. **Localize sharded `padding_idx`, strip MTP layers dynamically, and guard shared fused-op defaults (`shard_local_index_vocab_padding_mtp_layer.md`, `shared_fused_op_default_spec_mismatch.md`, `vendored_custom_op_vs_cann_samename_op_conflict.md`).**
   - When `VocabParallelEmbedding` shards the vocabulary across TP ranks `[vocab_start, vocab_end)`, translate global `padding_idx` to `padding_idx - vocab_start` (or `None` if outside the rank's shard) so `padding_idx` does not zero out an unrelated token row.
   - Filter out MTP weights (`model.layers.<k>.*` where `k >= config.num_hidden_layers`) during target-model checkpoint loading instead of hardcoding layer index `61`.
   - Never assume shared fused MoE/MLP wrappers default to the right activation or bias rule across model families (e.g., `GEGLU` vs. default `SiLU` in Gemma-4, clamped `SwiGLU` in GPT-OSS, or dense bias dropped by MXFP8 linear wrappers).
