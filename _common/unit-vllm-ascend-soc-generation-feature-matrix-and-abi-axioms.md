---
category: axiom
conditions:
  framework: vllm-ascend
  soc_family:
  - Ascend910B
  - Ascend910_93
  - Ascend950
  - Ascend310P
domain_path: _common
evidence_strength: hardware_spec
id: unit-vllm-ascend-soc-generation-feature-matrix-and-abi-axioms
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/device/hardware_profile.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/device/hardware.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/device/device_op.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/platform.py
  source_digest: 206085b82cd763fbe4409fcaf448128ca93a00671292d1cbe8a4e0f629da840c
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/device/hardware_profile.py
  source_type: hardware_profile_specification
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Ascend SoC generation hardware profiles (A2/A3/A5/310P), vector-core scaling,
  and ABI boundaries
triggers:
  error_signatures:
  - Ascend310P rejects bfloat16 model dtype or HCCL collective communication at initialization
  - Triton kernel launch grid exceeds available vector cores on 910B4 or 310P
  - Ascend950 A5 device operator dispatch selects 910B custom kernel or incompatible
    UB workspace
  keywords:
  - AscendDeviceType
  - HardwareProfile
  - Ascend910B
  - Ascend910_93
  - Ascend950
  - Ascend310P
  - vectorcore
  - DeviceOperator
  - A2
  - A3
  - A5
  - 310P
---
# Ascend SoC generation hardware profiles (A2/A3/A5/310P), vector-core scaling, and ABI boundaries

## Condition

Applies to all `vllm-ascend` workloads executing on Atlas A2 (`Ascend910B1`..`Ascend910B4`), Atlas A3 (`Ascend910_93` dual-die HCCS/UB), Ascend 950 (`Ascend950PR`/`Ascend950DT` / `A5`), and Atlas 300I DUO (`Ascend310P`).

## Constraint

1. **SoC capability and dtype matrix (`vllm_ascend/device/hardware_profile.py`, `vllm_ascend/device/hardware.py`):**
   - **Atlas A2 (`Ascend910B`) and Atlas A3 (`Ascend910_93`):** Support `float16` and `bfloat16` compute, `HCCL` collectives, `ACLGraph` (`PIECEWISE` and `FULL_DECODE_ONLY`), `npugraph_ex` TorchAir FX fusion, and Triton Ascend vector-core kernels. A2/A3 vector-core counts are queried dynamically via `get_vectorcore_num()` (typically 48 vector cores on 910B1/B2/93, 40 on 910B3, 32 on 910B4).
   - **Ascend 950 (`A5`):** Uses `DeviceOperatorA5` (`vllm_ascend/device/device_op.py`) with dedicated A5 operator bindings (`torch.ops._C_ascend.indexer_Select_...`, `float8_e4m3fn` indexer cache with `float32` scale, `float32` compressor `RMSNorm` weights, and UB/URMA device mounts `/dev/ummu`, `/dev/uburma`). Never dispatch A2/A3-specific `wkv`/`wgate` quantized compressor paths or TP-sharded index decode on `A5` when the kernel contract requires replicated index-K cache.
   - **Atlas 300I DUO (`Ascend310P`):** Supports `float16` only (`bfloat16` is unsupported in hardware). Inter-card communication uses `Gloo` (`init_device_distributed_environment` sets backend `"gloo"` on 310P) rather than HCCL. `weight_nz_mode` is forced to `FRACTAL_NZ` (`enable_weight_nz_layout`) for linear/MoE/attention weights, and Triton packages (`triton`, `triton-ascend`) must not be active on 310P.
2. **Hardware profile dispatch (`HardwareProfile`):**
   - All SoC-dependent thresholds (default `block_size`, `max_capture_size`, `flashcomm` support, `mla_pa` support, and `DeviceOperator` class selection) must be resolved through `get_ascend_device_type()` and `HardwareProfile` rather than hardcoded chip-name substrings.
3. **Custom C++ ABI and `torch.ops._C_ascend` registration:**
   - Custom operators compiled into `vllm_ascend._C` must match the installed CANN toolkit and `torch_npu` ABI. When `enable_sleep_mode=True` or custom C++ kernels are invoked, `vllm_ascend/meta_registration.py` registers fake/meta implementations on the `"Meta"` dispatch key only for Dynamo shape tracing; actual execution requires the NPU kernel bound on `"PrivateUse1"`.

## Rationale

Assuming uniform `bfloat16`, HCCL, or 48-core vector grids across Ascend generations causes immediate startup crashes on `Ascend310P`, sub-block launch truncation on `Ascend910B4`, or silent precision/layout corruption on `Ascend950` (`A5`).

## Evidence

Verified in `vllm_ascend/device/hardware_profile.py`, `vllm_ascend/device/hardware.py`, `vllm_ascend/device/device_op.py`, and `vllm_ascend/platform.py`.
