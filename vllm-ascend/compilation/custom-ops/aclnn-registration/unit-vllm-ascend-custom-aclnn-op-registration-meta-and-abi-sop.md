---
category: sop
conditions:
  component: custom_ops
  framework: vllm-ascend
domain_path: vllm-ascend/compilation/custom-ops/aclnn-registration
evidence_strength: software_contract
id: unit-vllm-ascend-custom-aclnn-op-registration-meta-and-abi-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/meta_registration.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/register_custom_ops.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/add_custom_aclnn_op.md
  source_digest: ebb0fe2505497c20ce208473d85f93a62d27a2f4052df0bf450543d5d833f1ac
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/meta_registration.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Custom AscendC / aclnn operator C++ binding, TORCH_LIBRARY_EXPAND, and meta_registration
  SOP
triggers:
  error_signatures:
  - 'NotImplementedError: Could not run ''_C_ascend::...'' with arguments from the
    ''Meta'' backend'
  - 'RuntimeError: No such operator _C_ascend::... when importing vllm_ascend'
  - aclnn two-stage workspace launch fails or mutates unaliased tensor inside torch.compile
  keywords:
  - _C_ascend
  - meta_registration
  - register_custom_ops
  - add_custom_aclnn_op
  - EXEC_NPU_CMD
  - TORCH_LIBRARY_EXPAND
  - direct_register_custom_op
---
# Custom AscendC / aclnn operator C++ binding, TORCH_LIBRARY_EXPAND, and meta_registration SOP

## Scope and preconditions

Use this SOP when registering a new C++ `aclnn` / AscendC operator under `torch.ops._C_ascend` (`csrc/` and `vllm_ascend/meta_registration.py`) or a Python custom op under `torch.ops.vllm` (`vllm_ascend/ops/register_custom_ops.py`).

## Procedure

1. **Follow the three-step C++ `aclnn` binding contract (`docs/source/developer_guide/Design_Documents/add_custom_aclnn_op.md`):**
   - **Host C++ wrapper (`csrc/aclnn_torch_adapter/` or `csrc/ops/`):** Invoke the two-stage `aclnn*GetWorkspaceSize` + `aclnn*` execution macro (`EXEC_NPU_CMD`) on the current NPU stream (`c10_npu::getCurrentNPUStream()`), ensuring workspace allocation uses the PyTorch NPU caching allocator so it is capture-safe under `ACLGraph`.
   - **Schema declaration (`csrc/torch_binding.cpp`):** Declare the exact schema string in `TORCH_LIBRARY_EXPAND(_C_ascend, ops)` with explicit alias annotations (`Tensor(a!)` for in-place mutated tensors) and bind the implementation on `"PrivateUse1"` via `TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m)`.
2. **Register a matching `"Meta"` kernel in `vllm_ascend/meta_registration.py`:**
   - Every `torch.ops._C_ascend.<op>` that can appear inside a `torch.compile` / `npugraph_ex` traced region must have a `@register_meta` function in `vllm_ascend/meta_registration.py` returning `torch.empty_like(...)` or `torch.empty(..., device="meta")` with the exact output shapes and dtypes of the C++ kernel.
   - In-place operators (`Tensor(a!)`) must return the mutated input tensor (or `None` for void schema) without allocating a new tensor that breaks Dynamo alias tracking.
3. **Register Python composite ops via `direct_register_custom_op` (`vllm_ascend/ops/register_custom_ops.py`):**
   - For Python-level ops (`maybe_all_gather_and_maybe_unpad`, `maybe_pad_and_reduce`, `quantize_RmsNorm`), always provide `fake_impl` alongside the op function and declare `mutates_args` accurately.

## Verification

- Test the operator under both eager (`device="npu"`) and `torch.compile(..., backend="eager")` / `Meta` tensor tracing (`device="meta"`) to confirm schema, alias, and shape parity.
