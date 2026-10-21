---
category: sop
conditions:
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  - Ascend950
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - vllm-ascend-idle-dp
domain_path: integration/custom-ops
evidence_strength: single_observation
id: unit-vllm-ascend-custom-op-gates-import-retry-and-nz
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - docs/evidence/unit-vllm-ascend-custom-op-gates-import-retry-and-nz-review.json
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/utils.py#L397-L456
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/utils.py#L299-L316
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/utils.py#L656-L786
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/utils.py#L249-L286
  - https://github.com/vllm-project/vllm-ascend/pull/10741
  source_digest: b04fd7d5698c9a1ad9cb1470fe4ce593e1e21bbdd00747c2d797850becc43aa9
  source_locator: docs/evidence/unit-vllm-ascend-custom-op-gates-import-retry-and-nz-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 4
schema: mindie-graded-unit/1
tier: L2
title: Ascend custom-op imports, registration and NZ selection have distinct gates
triggers:
  error_signatures:
  - 'RuntimeError: No such operator _C_ascend::npu_compressor_metadata because camem imported vllm_ascend_C before ASCEND_CUSTOM_OPP_PATH bootstrap'
  keywords:
  - vllm-ascend
  - ascend
  - custom-op
  - imports
  - registration
  - and
  - nz
  - selection
  - have
  - distinct
  - gates
  - Ascend910_93
  - bootstrap_custom_op_env
  - ASCEND_CUSTOM_OPP_PATH
  - CaMemAllocator
---

# Ascend custom-op imports, registration and NZ selection gates

## When to use

Diagnose a `vllm-ascend` custom-op failure at revision `175ef4d49c2072f9233619636f0309f4f88b4e2c`: a missing `_C_ascend` symbol, a failed `libcust_opapi.so` import, a partially applied registration, or an NZ/ND weight mismatch. This is source inspection, not a claim of operator correctness.

## Gates

1. **Import & Early `vllm_ascend_C` Bootstrap Order (`PR #10741`).** `enable_custom_op` lazily memoizes its result. Batch-invariant mode sets it false before extension import; on A5 (`Ascend950`), only direct kernels (`VLLM_ENABLE_ATB_AND_DIRECT_KERNELS`) are skipped in `meta_registration.py` rather than disabling `enable_custom_op` globally. Crucially, `worker.py` top-level imports `CaMemAllocator` (`device_allocator/camem.py`), which imports `vllm_ascend_C` **before** `platform.import_kernels()` runs `bootstrap_custom_op_env()`. If `ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH` are not bootstrapped before that first `vllm_ascend_C` load, custom vendor ops (such as `CompressorMetadata`) fail to register even when `enable_custom_op()` later returns `True`. Always invoke `bootstrap_custom_op_env(include_vendor_lib=True)` prior to the first `vllm_ascend_C` import.
2. **Retry.** Only an `ImportError` containing `libcust_opapi.so` permits one non-compiling retry after adding the vendor library path. That retry imports meta registration before `vllm_ascend_C`, reversing the initial order. Other `ImportError`s set the result false; other exception types propagate.
3. **Registration.** `register_ascend_customop` registers the operation map, adding GateLinear only for a resolved DeepSeek MLA model and substituting selected implementations on 310P. Its completion flag is set after the loop, so an exception can leave earlier registrations applied while the flag remains false. Check for that partial state.
4. **NZ policy.** float32 and meta tensors are excluded; 310P then selects NZ; otherwise `weight_nz_mode` zero disables, two permits only bf16/fp16, and other truthy values permit remaining dtypes. The adjacent `VLLM_ASCEND_ENABLE_NZ` comment does not name the value this function reads.

## Procedure

Identify the gate that actually failed before retrying or editing. Verify via `/proc/<pid>/maps` and `os.environ["ASCEND_CUSTOM_OPP_PATH"]` that `bootstrap_custom_op_env` ran before `vllm_ascend_C` was loaded by `camem.py`. An import success or a registered operation map is at most a metadata/source result; it does not establish kernel correctness, state parity or collective completion. Then take the runtime L1 probe for binding, stream and layout.

## Boundaries

Pinned to the revision above; do not generalize the retry order or the NZ policy to another revision. Source-derived inspection only: no reproduced hardware outcome and no measured performance benefit.
