---
category: sop
conditions:
  component: quantization_modelslim
  framework: vllm-ascend
domain_path: vllm-ascend/quantization/modelslim-and-mxfp/modelslim-schemes
evidence_strength: software_contract
id: unit-vllm-ascend-modelslim-w8a8-w4a16-w4a8-c8-and-quarot-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/modelslim_config.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/methods/registry.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/methods/w4a8.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/methods/w4a4_flatquant.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/methods/kv_c8.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/quantization.md
  source_digest: 69354dadb0aab0069ce252dcb234315d83f19b10fe920b3e09ad393ed33dc8c0
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/modelslim_config.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Ascend ModelSlim quantization schemes (W8A8 static/dynamic/pdmix, W4A16, W4A8,
  W4A4 FlatQuant/QuaRot, KV C8) SOP
triggers:
  error_signatures:
  - quant_model_description.json missing or layer quant_type mismatches registered
    Ascend quantization method
  - W4A8 or W4A4 FlatQuant weight packing fails on process_weights_after_loading
  - KV C8 cache scale tensor missing or misaligned with MLA/SFA latent cache
  keywords:
  - AscendModelSlimConfig
  - modelslim_config
  - quant_model_description
  - W8A8
  - W8A8_DYNAMIC
  - W8A8_PDMIX
  - W4A16
  - W4A8
  - W4A4_FLATQUANT
  - KV_C8
  - QuaRot
---
# Ascend ModelSlim quantization schemes (W8A8 static/dynamic/pdmix, W4A16, W4A8, W4A4 FlatQuant/QuaRot, KV C8) SOP

## Scope and preconditions

Use this SOP when configuring, loading, or debugging `msmodelslim` checkpoints (`--quantization ascend`, `quant_model_description.json`) routed by `AscendModelSlimConfig` (`vllm_ascend/quantization/modelslim_config.py`) and `vllm_ascend/quantization/methods/`.

## Procedure

1. **Inspect per-layer scheme dispatch in `quant_model_description.json` (`vllm_ascend/quantization/methods/registry.py`):**
   - `AscendModelSlimConfig` reads `quant_model_description.json` and maps each layer prefix to a registered scheme:
     - `W8A8` (`w8a8_static.py`): Static per-tensor activation scale (`input_scale`, `input_offset`) + per-channel int8 weight (`deq_scale`, `quant_bias`), suited for dense/attention projections.
     - `W8A8_DYNAMIC` (`w8a8_dynamic.py`): On-the-fly per-token dynamic activation quantization (`npu_dynamic_quant`) + per-channel int8 weight, standard for MoE routed experts.
     - `W8A8_PDMIX` (`w8a8_pdmix.py`): Uses dynamic quantization during prefill and static quantization during decode to balance TTFT and TPOT.
     - `W4A16` (`w4a16.py`) and `W8A16` (`w8a16.py`): Weight-only int4/int8 group-wise quantization via `npu_weight_quant_batchmatmul`.
     - `W4A8_DYNAMIC` (`w4a8.py`): Per-group/per-channel int4 weight packed into int8/int32 storage with per-token dynamic int8 activations (`npu_grouped_matmul` / `w4a8` fused kernels).
     - `W4A4_FLATQUANT_DYNAMIC` / `W4A4_LAOS_DYNAMIC` (`w4a4_flatquant.py`, `w4a4_laos_dynamic.py`): Kronecker / Hadamard / QuaRot rotation matrices (`left_trans`, `right_trans`, `clip_ratio`) fused with int4 activation quantization.
2. **Verify `process_weights_after_loading` layout and NZ casting:**
   - Every quantized linear and MoE method transforms raw checkpoint tensors inside `process_weights_after_loading` (transposing `[N, K]` to `[K, N]`, packing int4 pairs, converting `deq_scale` to `int64`/`float32` as required by CANN, and optionally casting weights to `ACL_FORMAT_FRACTAL_NZ` when `enable_weight_nz_layout` / `weight_nz_mode` is active).
   - Never capture `ACLGraph` or export `RFork` / `Netloader` post-processed pointers before `process_weights_after_loading` completes.
3. **Wire `KV_C8` (`kv_c8.py`) and FA3 attention scales:**
   - When `kv_cache_type == "C8"` (or `enable_sparse_c8` / `fa3_quant` entries exist in `quant_model_description.json`), verify that `k_proj`/`v_proj` (or MLA `kv_a_proj` / indexer) load their `kv_cache_scale` / `antiquant_scale` tensors and register them on the attention layer before warmup.

## Verification

- Confirm that `quant_model_description.json` loads without fallback to unquantized linear on intended quantized layers, and validate greedy logit cosine similarity (`> 0.99`) against the BF16 baseline before throughput benchmarking.
