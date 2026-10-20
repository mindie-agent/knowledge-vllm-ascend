---
category: workflow
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
domain_path: vllm-ascend/models/multimodal-and-vla/vlm-and-dit
evidence_strength: software_contract
id: unit-vllm-ascend-multimodal-vlm-mrope-and-video-dit-adaptation-workflow
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/vllm_cuda_multimodal_npu_bringup.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/mm_bringup_config_scan_and_oot_op_rank_contract.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/mm_encoder_packed_attention_seqlen_semantics.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/mrope_triton_ascend_vs_npu_mrope_decode.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/vision_encoder_rel_pos_bias_via_pse_shift.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/video_dit_blockwise_sparse_attention.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/video_dit_ulysses_sequence_parallel.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/dit_step_cache_fbcache_teacache.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/dit_step_cache_cfg_branch_slotting.md
  source_digest: 0364429b91b0149ebdce10e9e56961f2f8c270acf56d9f6266ef05bb9b178845
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/runbooks/runtime/vllm_cuda_multimodal_npu_bringup.md
  source_type: cannbot_workflow_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Multimodal VLM (Qwen-VL, InternVL) M-RoPE/ViT adaptation and Video DiT sparse
  attention, Ulysses SP, and step-cache workflow on Ascend NPU
triggers:
  error_signatures:
  - Multimodal ViT encoder packed attention produces wrong features when raw per-image
    lengths are passed to cumulative-sum TND kernel
  - M-RoPE decode throughput regresses on npu_mrope compared to triton_mrope
  - Video DiT with TeaCache/FBCache under serial CFG corrupts residuals when cond
    and uncond share one cache slot
  - Ulysses sequence parallel on Video DiT fails when RoPE freqs are not padded to
    world_size divisibility
  keywords:
  - vllm_cuda_multimodal_npu_bringup
  - mm_encoder_packed_attention
  - triton_mrope
  - npu_mrope
  - pse_shift
  - video_dit_blockwise_sparse
  - video_dit_ulysses
  - TeaCache
  - FBCache
---
# Multimodal VLM (Qwen-VL, InternVL) M-RoPE/ViT adaptation and Video DiT sparse attention, Ulysses SP, and step-cache workflow on Ascend NPU

## Objective

Adapt and optimize Multimodal Vision-Language Models (Qwen2.5-VL, Qwen3-VL, InternVL, GLM-V) and Diffusion Transformer (DiT) video/image generation pipelines (`HunyuanVideo`, `Wan2.2-I2V`, `Sana-Video`, `HunyuanImage-3.0`) on Ascend NPU.

## Workflow Steps

1. **Resolve Vision Encoder packed-sequence `seq_len` semantics and relative position bias (`vllm_cuda_multimodal_npu_bringup.md`, `mm_encoder_packed_attention_seqlen_semantics.md`, `vision_encoder_rel_pos_bias_via_pse_shift.md`).**
   - Replace CUDA-hardcoded `flash_attn_varlen_func` and `xformers` calls in the ViT encoder with `torch_npu.npu_fusion_attention` (or `npu_prompt_flash_attention`).
   - **Critical `seq_len` Contract:** Unpad-style attention kernels take **raw per-window lengths** `[L_1, L_2, ...]`, whereas `npu_fusion_attention` in `TND` layout requires **prefix-sum (`cumsum`) lengths** `[L_1, L_1 + L_2, ...]` (or `[0, L_1, L_1 + L_2, ...]` depending on operator signature). Never swap the two without converting between raw lengths and `cu_seqlens`.
   - For SAM-style decomposed relative position bias in vision encoders, pass the bias tensor through the `pse_shift` argument of `npu_prompt_flash_attention` rather than falling back to unfused eager SDPA.

2. **Select `triton_mrope` for Multimodal 3D Rotary Position Embedding in decode (`mrope_triton_ascend_vs_npu_mrope_decode.md`, `mm_bringup_config_scan_and_oot_op_rank_contract.md`).**
   - For 3D `[temporal, height, width]` `M-RoPE` (`positions` shape `[3, num_tokens]`), route decode through the `triton-ascend` `triton_mrope` kernel when available (which fuses the 3-section gather and rotation with lower launch overhead than `npu_mrope`), and verify that OOT custom ops accept both `[num_tokens, head_dim]` 2D and `[batch, seq, head_dim]` 3D rank contracts.

3. **Enable Ulysses Sequence Parallelism and Blockwise Sparse Attention on Video DiT (`video_dit_ulysses_sequence_parallel.md`, `video_dit_blockwise_sparse_attention.md`).**
   - When sharding long video token sequences ($T \times H \times W > 64\text{K}$) across Ulysses SP (`sp_size`), pad both the hidden states *and* the 3D RoPE `freqs_cis` tensor to a multiple of `sp_size`, and validate that `num_heads % ulysses_size == 0` (or enable UAA head-padding).
   - For Blockwise Sparse Attention (TopK or Spatial-Visual-Grid SVG) in DiT, always mark the **first video frame and all text conditioning tokens as `must-keep` dense blocks**; use offline attention-map profiling when calibrating static TopK masks.

4. **Slot DiT Step Caches (`FBCache` / `TeaCache` / `TaylorSeer`) per CFG branch and synchronize SP decisions (`dit_step_cache_cfg_branch_slotting.md`, `dit_step_cache_fbcache_teacache.md`, `dit_cache_with_sequence_parallel.md`).**
   - When Classifier-Free Guidance (CFG) runs `cond` and `uncond` forward passes serially on the same model instance, maintain **separate cache slots keyed by `branch_id` (`cond` vs. `uncond`)** so `uncond` step $t$ never overwrites or reads the residual delta of `cond` step $t$.
   - When combining data-dependent step skipping (`TeaCache` / `FBCache` L1 relative-diff metric) with multi-card Sequence Parallelism, `all_reduce` the skip decision norm across the SP group before branching so all SP ranks skip or execute the DiT block in lockstep.
