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
  - https://github.com/vllm-project/vllm-ascend/pull/9613
  - https://github.com/vllm-project/vllm-ascend/pull/10378
  - https://github.com/vllm-project/vllm-ascend/pull/10726
  - https://github.com/vllm-project/vllm-ascend/pull/12106
  - https://github.com/vllm-project/vllm-ascend/pull/14516
  source_digest: ebb0fe2505497c20ce208473d85f93a62d27a2f4052df0bf450543d5d833f1ac
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/meta_registration.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 2
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
  - linking full libopapi.so in custom_build.cmake shadows PTA mc2 dispatch or combine symbols
  - torch_npu.npu_kv_quant_sparse_flash_attention segfaults with exit 139 when return_lse is False
  - undefined symbol LLVMInitializeSw64TargetInfo when host /usr/lib64 shadows container libLLVM
  keywords:
  - _C_ascend
  - meta_registration
  - register_custom_ops
  - add_custom_aclnn_op
  - EXEC_NPU_CMD
  - TORCH_LIBRARY_EXPAND
  - direct_register_custom_op
  - opapi_math
  - VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
  - vllm_quant_lightning_indexer
  - npu_kv_quant_sparse_flash_attention
---
# Custom AscendC / aclnn operator C++ binding, TORCH_LIBRARY_EXPAND, and meta_registration SOP

## Scope and preconditions

Use this SOP when registering a new C++ `aclnn` / AscendC operator under `torch.ops._C_ascend` (`csrc/` and `vllm_ascend/meta_registration.py`) or a Python custom op under `torch.ops.vllm` (`vllm_ascend/ops/register_custom_ops.py`).

## Procedure

1. **Follow the three-step C++ `aclnn` binding contract (`docs/source/developer_guide/Design_Documents/add_custom_aclnn_op.md`):**
   - **Host C++ wrapper (`csrc/aclnn_torch_adapter/` or `csrc/ops/`):** Invoke the two-stage `aclnn*GetWorkspaceSize` + `aclnn*` execution macro (`EXEC_NPU_CMD`) on the current NPU stream (`c10_npu::getCurrentNPUStream()`), ensuring workspace allocation uses the PyTorch NPU caching allocator so it is capture-safe under `ACLGraph`.
   - **Schema declaration (`csrc/torch_binding.cpp`):** Declare the exact schema string in `TORCH_LIBRARY_EXPAND(_C_ascend, ops)` with explicit alias annotations (`Tensor(a!)` for in-place mutated tensors) and bind the implementation on `"PrivateUse1"` via `TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m)`.
   - **Link `opapi_math` instead of full `libopapi.so` (`PR #9613`, `PR #14516`):** In `cmake/custom_build.cmake`, never link the monolithic `opapi` (`libopapi.so`) into custom operator targets because it shadows PTA's native `mc2` (`dispatch` / `combine`) symbols at runtime; link `opapi_math` instead.
2. **Register a matching `"Meta"` kernel in `vllm_ascend/meta_registration.py` and respect 310P / A5 direct-kernel gates:**
   - Every `torch.ops._C_ascend.<op>` that can appear inside a `torch.compile` / `npugraph_ex` traced region must have a `@register_meta` function in `vllm_ascend/meta_registration.py` returning `torch.empty_like(...)` or `torch.empty(..., device="meta")` with the exact output shapes and dtypes of the C++ kernel.
   - In-place operators (`Tensor(a!)`) must return the mutated input tensor (or `None` for void schema) without allocating a new tensor that breaks Dynamo alias tracking.
   - **310P & A5 (`Ascend950`) Direct-Kernel Gate (`PR #10378`):** `VLLM_ENABLE_ATB_AND_DIRECT_KERNELS` is disabled on both Ascend310P and Ascend950 (A5). Skip Meta registrations for direct kernels (`get_masked_input_and_mask`, `bgmv_expand`, `sgmv_expand`) on both 310P and A5 rather than globally disabling `enable_custom_op()` on A5.
3. **Register Python composite ops and prevent symbol collisions / broken `torch_npu` bindings (`vllm_ascend/ops/register_custom_ops.py`, `device_op.py`):**
   - For Python-level ops (`maybe_all_gather_and_maybe_unpad`, `maybe_pad_and_reduce`, `quantize_RmsNorm`), always provide `fake_impl` alongside the op function and declare `mutates_args` accurately.
   - **Custom Op vs. CANN Built-in Coexistence (`PR #10726`, `PR #10744`):** Namespace experimental `_C_ascend` operators (e.g., `vllm_quant_lightning_indexer`) so they do not shadow or collide with CANN's built-in `torch_npu.npu_quant_lightning_indexer` required by SFA layers in the same process.
   - **Route through verified `_C_ascend` bindings when `torch_npu` wrapper crashes (`PR #12106`):** Do not split `device_op.py` dispatch between `_C_ascend` (`return_lse=True`) and `torch_npu` (`return_lse=False`) when `torch_npu.npu_kv_quant_sparse_flash_attention` segfaults in eager/graph capture (`exit 139`); route both branches through `torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention(..., return_softmax_lse=return_lse)`.
4. **Enforce CANN B080 header shims, container LLVM isolation, and hybrid `.asc` / Host C++ compiler splits:**
   - When CANN 9.0 B080 removes the bare `OP` macro from `op_common/log/log.h` or conflicts with `toolchain/slog.h` (`CheckLogLevel` / `DlogRecord`), inject a centralized `-include cann_compat.h` in CMake rather than adding per-operator runtime fallbacks.
   - On A5 containers that mount host `/usr/lib64` for driver access, prepend the container's LLVM directory to `LD_LIBRARY_PATH` so host libraries do not shadow `libLLVM-17.so` / `libclang-cpp.so.17` (`undefined symbol: LLVMInitializeSw64TargetInfo`).
   - For hybrid `.asc` + Host C++ extensions (`FlashMLA`, `DeepSelect`, `DeepGEMM`), compile `.asc` sources with `bisheng -U__cpp_concepts` and Host C++ sources with `g++-13` (stripping `--npu-arch`, `-mllvm`, and `--cce-res-usage`), and apply any custom `ASCEND_HOME_PATH` JIT overlay **after** sourcing CANN `set_env.sh`.

## Verification

- Test the operator under eager (`device="npu"`), `torch.compile(dynamic=True)` / `Meta` tensor tracing (`device="meta"`), PyTorch `opcheck`, and `npugraph_ex` capture/replay across both BSND and TND layouts to confirm schema, alias, and numerical parity.
