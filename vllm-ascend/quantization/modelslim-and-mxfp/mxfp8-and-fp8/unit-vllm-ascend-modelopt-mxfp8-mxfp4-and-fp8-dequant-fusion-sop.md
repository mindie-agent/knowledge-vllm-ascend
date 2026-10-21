---
category: sop
conditions:
  component: quantization_mxfp_fp8
  framework: vllm-ascend
domain_path: vllm-ascend/quantization/modelslim-and-mxfp/mxfp8-and-fp8
evidence_strength: software_contract
id: unit-vllm-ascend-modelopt-mxfp8-mxfp4-and-fp8-dequant-fusion-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-soc-generation-feature-matrix-and-abi-axioms
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/modelopt_mxfp8_config.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/fp8_config.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/compressed_tensors_config.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/methods/w8a8_mxfp8.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/methods/w4a8_mxfp4.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/methods/fp8.py
  source_digest: 9c664e19b56191246eb1103b2288db66b7ff82156d68c3483456724cf71c0a4b
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/modelopt_mxfp8_config.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: ModelOpt MXFP8/MXFP4 microscaling, CompressedTensors, and FP8 dynamic quantization
  on Ascend SOP
triggers:
  error_signatures:
  - ModelOpt mxfp8 or fp8 checkpoint fails to map weight_scale or block_size on Ascend
  - MXFP8 or MXFP4 E8M0 block scale shape mismatches npu_dynamic_mx_quant group size
    32
  - compressed-tensors checkpoint raises unsupported scheme error in AscendCompressedTensorsConfig
  keywords:
  - AscendModelOptMXFP8Config
  - modelopt_mxfp8_config
  - fp8_config
  - compressed_tensors_config
  - w8a8_mxfp8
  - w4a8_mxfp4
  - w4a4_mxfp4
  - npu_dynamic_mx_quant
  - float8_e4m3fn
---
# ModelOpt MXFP8/MXFP4 microscaling, CompressedTensors, and FP8 dynamic quantization on Ascend SOP

## Scope and preconditions

Use this SOP when loading or executing `modelopt` MXFP8 (`vllm_ascend/quantization/modelopt_mxfp8_config.py`), native `fp8` (`fp8_config.py`, `methods/fp8.py`, `methods/w8a8fp8_dynamic.py`), `compressed-tensors` (`compressed_tensors_config.py`), or ModelSlim microscaling (`w8a8_mxfp8.py`, `w4a8_mxfp4.py`, `w4a16_mxfp4.py`, `w4a4_mxfp4.py`) checkpoints on Ascend NPUs.

## Procedure

1. **Enforce MXFP8 / MXFP4 microscaling block-size and scale contracts (`w8a8_mxfp8.py`, `w4a8_mxfp4.py`, `modelopt_mxfp8_config.py`):**
   - Microscaling formats (`MXFP8` `e4m3fn` and `MXFP4` `e2m1`) use block size `32` along the reduction dimension $K$, paired with shared exponent scales (`uint8` `E8M0`, shape `[N, K // 32]`).
   - Activation quantization calls `torch_npu.npu_dynamic_mx_quant(x, axis=-1, round_mode="rint", dst_type=...)` to produce the quantized activation tensor and its per-32-element `mxscale` tensor before `npu_quant_matmul` or `npu_grouped_matmul`.
   - Ensure $K$ is divisible by 32 on every TP shard (`input_size_per_partition % 32 == 0`).
2. **Route upstream FP8 checkpoints via `AscendFp8Config` (`vllm_ascend/quantization/fp8_config.py`):**
   - Native FP8 checkpoints (`quant_method: "fp8"`) map to `AscendFp8LinearMethod` (`methods/fp8.py`) and `AscendFp8MoEMethod`.
   - On SoCs with native FP8 / MXFP8 cube support (such as Ascend 950 / `A5` or supported CANN 8.5+ paths), `npu_quant_matmul` consumes `float8_e4m3fn` directly; on paths requiring weight dequantization, weights are dequantized or adapted via `w8a8fp8_dynamic.py` during forward or weight post-processing.
3. **Map `compressed-tensors` configs via `AscendCompressedTensorsConfig` (`compressed_tensors_config.py`):**
   - `AscendCompressedTensorsConfig` translates `compressed-tensors` scheme descriptors (`W8A8Int8`, `W4A16`, `W8A16`, `FP8`) into the corresponding `vllm_ascend/quantization/methods/` implementation through `method_adapters.py`.

## Verification

- Verify that `weight_scale` / `E8M0` scale shapes match `[N_shard, K_shard // 32]` after TP sharding and that a single-step forward produces finite BF16/FP16 outputs with zero `EZ1001` dtype errors.
