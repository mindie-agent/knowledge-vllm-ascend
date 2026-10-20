---
category: sop
conditions:
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
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
  source_digest: b04fd7d5698c9a1ad9cb1470fe4ce593e1e21bbdd00747c2d797850becc43aa9
  source_locator: docs/evidence/unit-vllm-ascend-custom-op-gates-import-retry-and-nz-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 3
schema: mindie-graded-unit/1
tier: L2
title: Ascend custom-op imports, registration and NZ selection have distinct gates
triggers:
  error_signatures: []
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
---

# Ascend custom-op imports, registration and NZ selection gates

## When to use

Diagnose a `vllm-ascend` custom-op failure at revision `175ef4d49c2072f9233619636f0309f4f88b4e2c`: a missing `_C_ascend` symbol, a failed `libcust_opapi.so` import, a partially applied registration, or an NZ/ND weight mismatch. This is source inspection, not a claim of operator correctness.

## Gates

1. **Import.** `enable_custom_op` lazily memoizes its result. Batch-invariant mode or an A5 device sets it false before extension import; otherwise extension and meta-registration imports are attempted even while `torch.compiler.is_compiling` is true, and only environment bootstrap is skipped while compiling.
2. **Retry.** Only an `ImportError` containing `libcust_opapi.so` permits one non-compiling retry after adding the vendor library path. That retry imports meta registration before `vllm_ascend_C`, reversing the initial order. Other `ImportError`s set the result false; other exception types propagate.
3. **Registration.** `register_ascend_customop` registers the operation map, adding GateLinear only for a resolved DeepSeek MLA model and substituting selected implementations on 310P. Its completion flag is set after the loop, so an exception can leave earlier registrations applied while the flag remains false. Check for that partial state.
4. **NZ policy.** float32 and meta tensors are excluded; 310P then selects NZ; otherwise `weight_nz_mode` zero disables, two permits only bf16/fp16, and other truthy values permit remaining dtypes. The adjacent `VLLM_ASCEND_ENABLE_NZ` comment does not name the value this function reads.

## Procedure

Identify the gate that actually failed before retrying or editing. An import success or a registered operation map is at most a metadata/source result; it does not establish kernel correctness, state parity or collective completion. Then take the runtime L1 probe for binding, stream and layout.

## Boundaries

Pinned to the revision above; do not generalize the retry order or the NZ policy to another revision. Source-derived inspection only: no reproduced hardware outcome and no measured performance benefit.
