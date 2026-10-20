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
domain_path: vllm-ascend/runbooks/precision-and-validation/bisect-and-alignment
evidence_strength: software_contract
id: unit-vllm-ascend-precision-bisect-cp-lse-rope-moe-and-validation-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/precision/cp_dcp_online_softmax_merge_mask_out_not_only_lse.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/precision/zigzag_cp_short_seq_single_segment_window.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/precision/interleaved_rope_npu_op_selection.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/precision/moe_router_norm_topk_prob_fused_topk_npu.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/precision/activation_quant_substring_gating_w4a8.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/precision/compressed_tensors_scheme_gating_pitfalls.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/precision/quant_state_misattached_to_cp_branch.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/precision/sp_gate_narrow_env_predicate_misses_shared_expert_dp.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/precision/mtp_draft_dummy_run_in_profile_run_random_experts.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/validations/longbench_input_max_len_sysprompt.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/validations/moe_ep_gmm_unconditional_zero_token_diagnosis.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/validations/prefill_profiler_gated_off_long_input.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-precision-debug/SKILL.md
  source_digest: 888a763bc3490d05dd7a3f3e29ed65debafb9dc218fa93c3df54ae27754de593
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/runbooks/precision/cp_dcp_online_softmax_merge_mask_out_not_only_lse.md
  source_type: cannbot_runbook_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Precision & validation runbook distillation: CP online-softmax LSE masking,
  ZigZag-CP short-seq window, RoPE layout, MoE top-k norm, and LongBench/profiler
  gates'
triggers:
  error_signatures:
  - CP/DCP online-softmax LSE merge produces NaN (0 * inf) when empty KV shards only
    set LSE=-inf without zeroing attention output
  - Sliding-window attention with ZigZag-CP silently corrupts short sequences when
    all valid tokens fall in segment 0
  - Interleaved RoPE model diverges when routed to half-split (NeoX-style) npu_rotary_mul
    mode without layout conversion
  - Enabling enable_shared_expert_dp alone corrupts outputs because SP branch checks
    only FlashComm env var
  keywords:
  - cp_dcp_online_softmax_merge
  - zigzag_cp_short_seq
  - interleaved_rope_npu_op
  - norm_topk_prob
  - sp_gate_narrow_env_predicate
  - quant_state_misattached_to_cp
  - longbench_input_max_len
  - moe_ep_gmm_unconditional
---
# Precision & validation runbook distillation: CP online-softmax LSE masking, ZigZag-CP short-seq window, RoPE layout, MoE top-k norm, and LongBench/profiler gates

## Applicability and symptom scope

Use this SOP to diagnose silent numerical divergence, `NaN` propagation under Context Parallelism (CP/DCP/ZigZag-CP), RoPE/MoE router mismatches, and validation/profiling blind spots in `vllm-ascend`.

## Procedure

1. **Zero-mask attention output alongside `LSE = -inf` when merging empty CP/DCP KV shards (`cp_dcp_online_softmax_merge_mask_out_not_only_lse.md`, `cp_prefill_empty_kv_shard_none_lse_merge.md`).**
   - In Context Parallelism (CP / DCP / Ring Attention), a rank whose local KV shard is empty for a causal query chunk is the identity element of online softmax: `(out = 0, lse = -inf)`.
   - Never return `None` from an empty shard (which crashes the combiner), and never set only `lse = -inf` while leaving uninitialized/garbage values or `NaN` in `out` (`0.0 * NaN` or `exp(-inf - (-inf))` produces `NaN`). Explicitly mask `out = torch.where(valid_mask, out, 0.0)` and `lse = torch.where(valid_mask, lse, -float("inf"))`.

2. **Guard ZigZag-CP sliding-window index wrap-around on short sequences (`zigzag_cp_short_seq_single_segment_window.md`).**
   - When a short prompt falls entirely inside the first ZigZag-CP segment (`last_rank == 0`), computing the previous segment index via `(last_rank - 1)` wraps around to `-1` (the last rank's empty tail segment) under Python negative indexing. Explicitly clamp or branch when `last_rank == 0` so sliding-window metadata reads only the valid prefix in segment `0`.

3. **Align Interleaved vs. Half-Split RoPE and MoE Router `norm_topk_prob` (`interleaved_rope_npu_op_selection.md`, `moe_router_norm_topk_prob_fused_topk_npu.md`).**
   - Distinguish GPT-J **interleaved** RoPE (`[x_0, x_1, x_2, x_3]`) from Llama **NeoX half-split** RoPE (`[x_0, x_{d/2}, ...]`). Either pass the matching `rotary_mode` (`"interleave"` vs. `"half"`) to the NPU rotary kernel or permute query/key weights `[2, head_dim // 2] -> transpose -> reshape` at load time.
   - Verify whether the model config sets `norm_topk_prob=True` (renormalizing selected top-k weights to sum to `1` *after* top-k selection) or `False`, and pass the exact boolean to fused `npu_moe_gating_top_k_softmax`.

4. **Decouple quantization/SP state initialization from narrow parallel predicates (`quant_state_misattached_to_cp_branch.md`, `sp_gate_narrow_env_predicate_misses_shared_expert_dp.md`, `mtp_draft_dummy_run_in_profile_run_random_experts.md`).**
   - Never attach C8 per-token scale buffer initialization inside an `if cp_size > 1:` block (which silently drops C8 scales when `cp_size == 1`), and never gate Sequence Parallelism tensor slicing solely on the FlashComm environment variable when `enable_shared_expert_dp=True` also forces SP layout.
   - Include the MTP draft model in `profile_run` dummy warmup with valid clamped expert IDs (`0 .. num_experts - 1`) so profiling captures worst-case peak memory without out-of-range expert indices.

5. **Validate LongBench `input_max_len` and pure-EP GMM profiling invariants (`longbench_input_max_len_sysprompt.md`, `moe_ep_gmm_unconditional_zero_token_diagnosis.md`, `prefill_profiler_gated_off_long_input.md`).**
   - Assert `input_max_len > len(system_prompt)` in LongBench evaluation builders (`input_max_len - len(system_prompt) < 0` silently slices off the prompt body via negative indexing).
   - In pure EP decode (`moe_ep_size > 1, moe_tp_size == 1`), `npu_grouped_matmul` is invoked unconditionally on every rank even when a rank receives `0` tokens (`expert_token_num` all zeros); if a rank's profiler trace lacks `GroupedMatmul` entirely, diagnose a control-flow bug rather than "zero routed tokens", and check whether `input_max_len > moe_chunk_max_len` gated off prefill profiling.
