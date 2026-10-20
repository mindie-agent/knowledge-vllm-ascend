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
domain_path: vllm-ascend/optimizations/quantization/w4a8-w8a8-mxfp
evidence_strength: software_contract
id: unit-vllm-ascend-w4a8-w8a8-mxfp8-mxfp4-gmm-and-outlier-clip-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/moe_expert_gmm_weight_lowbit_prep.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/950_mxfp8_moe_activation_scale_packed_layout.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/precision_low_bit_quant_clip_clamp_outlier.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/w4a8_moe_gmm_prefill_skip_bias.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/moe_role_based_precision_allocation.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/quantization/w4a8c8.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/quantization/hybrid_mxfp8_mxfp4.md
  source_digest: 9bf0db08da94e72f5a6845810c6c655f6959e98921e700c5cc199223f6c83a6e
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/moe_expert_gmm_weight_lowbit_prep.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'W4A8, W8A8, and MXFP8/MXFP4 quantization recipes: MoE GMM weight packing,
  bias-assist matrices, and outlier clipping in vLLM-Ascend'
triggers:
  error_signatures:
  - W4A8 MoE GroupedMatmul fails or outputs corrupted values due to missing bias-assist
    compensation matrix or un-repacked INT4 N-dim
  - Ascend 950 MXFP8 fused MoE raises shape error when activation scale is not packed
    to [M, N//2, 2]
  - W4A8 routed expert quantization suffers accuracy collapse without outlier clipping
    on gate_up/down projections
  keywords:
  - moe_expert_gmm_weight_lowbit_prep
  - 950_mxfp8_moe_activation_scale
  - w4a8_moe_gmm_prefill_skip_bias
  - precision_low_bit_quant_clip
  - moe_role_based_precision
  - npu_grouped_matmul
  - W4A8C8
  - MXFP4
---
# W4A8, W8A8, and MXFP8/MXFP4 quantization recipes: MoE GMM weight packing, bias-assist matrices, and outlier clipping in vLLM-Ascend

## Applicability and symptom scope

Use this SOP when implementing or debugging low-bit quantized inference (`W8A8`, `W4A8`, `W4A16`, `W4A8C8`, `MXFP8`, `MXFP4`) for dense and MoE models on Ascend NPU.

## Procedure

1. **Allocate precision by architectural role in large MoE models (`moe_role_based_precision_allocation.md`, `w4a8c8.md`, `hybrid_mxfp8_mxfp4.md`).**
   - Routed MoE experts dominate parameter count (`> 85%` of total weights) and decode HBM bandwidth but are individually sparse per token; quantize routed experts most aggressively (`W4A8` per-group on Atlas A2/A3 or `MXFP4` weight / `MXFP8` activation on Ascend 950).
   - Keep high-sensitivity shared modules at higher precision:
     - Router / Gate classifier (`gate.weight`) and RMSNorm weights: `BF16` / `FP32`.
     - Attention projections (`q_a_proj`, `kv_a_proj`, `q_b_proj`, `o_proj`) and Shared Experts: `W8A8` dynamic (or `MXFP8` on Ascend 950), never uncalibrated `INT4`.
     - Embedding and `lm_head`: `BF16` (or `W8A8` only after perplexity verification).

2. **Prepare W4A8 / W4A16 MoE GroupedMatmul (`npu_grouped_matmul`) weights at load time (`moe_expert_gmm_weight_lowbit_prep.md`, `w4a8_moe_gmm_prefill_skip_bias.md`).**
   - In `process_weights_after_loading`:
     - **W4A8 Triplet:** Convert packed INT4 weights to `FRACTAL_NZ` (`npu_format_cast`), precompute the static signed-activation offset compensation (`bias` assist matrix $= 8 \times \sum_k w_{k} \cdot s_{w}$ when asymmetric uint4 storage is shifted to int4), and cast per-group `scale` to the `uint64` / `int64` packed layout required by `npu_grouped_matmul`.
     - **Phase-Gated Bias Skip (`w4a8_moe_gmm_prefill_skip_bias.md`):** Where the W4A8 GMM kernel decomposes bias addition into a separate overhead pass during large-M prefill, gate the optional zero-mean bias assist by execution phase if validated by accuracy tolerances, while always retaining the full compensation triplet in decode.
     - **W4A16 Repacking:** Transpose and pack every two 4-bit elements along the $N$ dimension into `int32` / `int8` words matching the `npu_weight_quant_batchmatmul` / GMM schema.

3. **Mitigate activation and weight outliers via calibration-time `clip` and runtime `clamp` (`precision_low_bit_quant_clip_clamp_outlier.md`).**
   - For `W4A8` routed experts, unbounded activation spikes after `SwiGLU` inflate the per-token dynamic INT8 scale and crush small channels to zero. Apply symmetric activation clamping before `npu_dynamic_quant` and enable weight clipping (`clip_ratio`) during ModelSlim / Golden-Stick PTQ calibration.

4. **Centralize Ascend 950 MXFP8 MoE activation scale packing (`950_mxfp8_moe_activation_scale_packed_layout.md`).**
   - On Ascend 950 (`950PR`/`950DT`), `npu_dynamic_mx_quant` produces microscaling E8M0 scales that must be reshaped/packed into `[M, N // 2, 2]` before entering fused MoE `npu_grouped_matmul`. Centralize this scale-packing step inside a single MoE quant-dispatch helper rather than duplicating shape-fragile `.view()` calls across model files.
