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
domain_path: vllm-ascend/optimizations/quantization/calibration-and-loading
evidence_strength: software_contract
id: unit-vllm-ascend-quant-weight-loading-scheme-gating-and-li-quant-sop
inherits_from:
- unit-vllm-ascend-w4a8-w8a8-mxfp8-mxfp4-gmm-and-outlier-clip-sop
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/quant_weight_load_layout_and_naming.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/deepseekv3_mixed_precision_quant_config_pitfalls.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/precision_quant_convert_model_config_pitfalls.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/precision_lightning_indexer_low_bit_quant.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/quant_lightning_indexer_latency.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-quantization/SKILL.md
  source_digest: abbb1d3205a3c75c29589445c80ce64c149479a951c9053c7a1a66bd5944c4ed
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/quant_weight_load_layout_and_naming.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Quantized checkpoint loading, substring scheme-gating pitfalls, and INT8 Lightning
  Indexer Hadamard rotation in vLLM-Ascend
triggers:
  error_signatures:
  - Substring match on layer name or quant_mode (e.g. 'w4a8' in 'w4a8_dynamic') routes
    layer to wrong quantization method
  - Zero weight_offset tensor passed to quantized linear kernel triggers unnecessary
    asymmetric dequant path or precision drift
  - INT8 quantized Lightning Indexer degrades top-k recall when Hadamard rotation
    is omitted or scale dtype is BF16 instead of FP32
  keywords:
  - quant_weight_load_layout
  - deepseekv3_mixed_precision_quant
  - precision_quant_convert_model
  - npu_quant_lightning_indexer
  - Hadamard
  - compressed-tensors
  - modelslim
---
# Quantized checkpoint loading, substring scheme-gating pitfalls, and INT8 Lightning Indexer Hadamard rotation in vLLM-Ascend

## Applicability and symptom scope

Use this SOP when converting, loading, or debugging mixed-precision quantized checkpoints (ModelSlim, `compressed-tensors`, or custom `convert_model.py` exports) and quantized Lightning Indexer paths in `vllm-ascend`.

## Procedure

1. **Enforce exact token/regex matching when gating quantization schemes by layer name or mode (`deepseekv3_mixed_precision_quant_config_pitfalls.md`, `precision_quant_convert_model_config_pitfalls.md`).**
   - Never use loose substring predicates such as `"mlp" in layer_name` (which matches both `mlp.experts` and `mlp.shared_experts` / `mlp.gate`) or `"w8a8" in quant_mode` without anchored boundaries.
   - Explicitly exclude router gates (`*.mlp.gate`), norms, embeddings, and MTP projection headers that must remain BF16, and verify that `convert_model.py` exports MTP layer shards (`layer >= num_hidden_layers`) with the exact quantization metadata expected by the MTP runner.

2. **Normalize scale/offset parameter names and strip all-zero `weight_offset` tensors at load time (`quant_weight_load_layout_and_naming.md`).**
   - When loading symmetric W8A8/W4A8 checkpoints that still serialize an all-zero `weight_offset` tensor, strip or set `weight_offset = None` in `process_weights_after_loading` so the NPU kernel takes the faster symmetric matmul path.
   - Gate `FRACTAL_NZ` format casting (`npu_format_cast`) strictly by tensor dtype (`int8` / packed `int4` / `bf16`, never `fp32` scales or `meta` tensors) and exclude `RowParallelLinear` layers that execute via MC2 `npu_mm_all_reduce_base` (which requires `ACL_FORMAT_ND`).

3. **Accelerate and preserve recall on INT8 Lightning Indexer with Hadamard rotation and FP32 scales (`quant_lightning_indexer_latency.md`, `precision_lightning_indexer_low_bit_quant.md`).**
   - Switching Lightning Indexer from BF16 to `npu_quant_lightning_indexer` (INT8 storage + INT8 compute) halves indexer KV memory bandwidth and reduces single-op latency.
   - To prevent outlier channels in indexer query/key projections from distorting top-k block selection scores:
     - Apply an orthogonal **Hadamard transform** (or fused `indexer_prolog` rotation) to both query and key vectors before INT8 dynamic quantization;
     - Keep quantization scales in **strict `torch.float32`** (`scale.dtype == torch.float32`)—casting indexer dequant scales to `bfloat16` introduces cumulative rounding error in top-k score sorting.
