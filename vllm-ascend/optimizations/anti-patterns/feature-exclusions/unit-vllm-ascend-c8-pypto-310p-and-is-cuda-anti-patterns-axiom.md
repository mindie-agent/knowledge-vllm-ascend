---
category: axiom
conditions:
  framework: vllm-ascend
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  - Ascend310P3
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - vllm-ascend-idle-dp
domain_path: vllm-ascend/optimizations/anti-patterns/feature-exclusions
evidence_strength: software_contract
id: unit-vllm-ascend-c8-pypto-310p-and-is-cuda-anti-patterns-axiom
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/anti_patterns/c8_kvcache_disables_pypto_lightning_indexer.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/anti_patterns/ascend_310p_bringup_unusable_ops_and_nz_layout.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/anti_patterns/npu_tensor_is_cuda_not_cuda_npu_discriminant.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/anti_patterns/multinic_pd_advertise_address_launch_config.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/anti_patterns/ascendc_op_input_validation_pitfalls.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/anti_patterns/ascendc_vector_kernel_silent_correctness_pitfalls.md
  source_digest: 28954fee346ee9a349de355692292e72fafbf04c5498916f16bfda41b83bc7df
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/anti_patterns/c8_kvcache_disables_pypto_lightning_indexer.md
  source_type: cannbot_anti_pattern_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Mutually exclusive optimization paths, Ascend 310P operator boundaries, and
  device-predicate anti-patterns in vLLM-Ascend
triggers:
  error_signatures:
  - enable_pypto=True has no effect when kv_cache_quant_mode is int8 or C8
  - x.is_cuda evaluates to True on torch_npu NPU tensor and dispatches to torch.ops._C
    CUDA symbol
  - Ascend 310P fails on MLA, sparse flash attention, batch_matmul_transpose, or non-NZ
    KV/mask layout
  - AscendC optional input tensor->GetShapeSize() segfaults when nullptr guard is
    missing
  keywords:
  - c8_kvcache_disables_pypto
  - npu_lightning_indexer_pto
  - npu_lightning_indexer_quant
  - is_cuda
  - Ascend310P3
  - FRACTAL_NZ
  - VEC_SCOPE
  - curColNumAlign
  - multinic_pd_advertise
---
# Mutually exclusive optimization paths, Ascend 310P operator boundaries, and device-predicate anti-patterns in vLLM-Ascend

## Condition

Applies when enabling combined inference optimizations, porting CUDA-native model/framework paths to `torch_npu`, deploying onto Ascend 310P (`SOC_VERSION == "ascend310p3"`), or authoring custom AscendC host-tiling and MicroAPI vector kernels in `vllm-ascend`.

## Constraint

1. **C8 KV Cache vs. PyPTO Lightning Indexer Mutual Exclusion (`c8_kvcache_disables_pypto_lightning_indexer.md`):**
   - In DeepSeek Sparse Attention (DSA) decode (`DeepSeek-V3.2-Exp` and structurally isomorphic models), Lightning Indexer dispatches across three mutually exclusive branches gated by `(is_prefill, enable_pypto, kv_cache_c8)`:
     - When `kv_cache_quant_mode == "int8"` (`kv_cache_c8=True`), decode unconditionally routes to `npu_lightning_indexer_quant`.
     - The PyPTO fast path (`npu_lightning_indexer_pto`) requires `not kv_cache_c8`. Never stack `enable_pypto=True` with C8 KV quantization expecting cumulative latency reduction.
2. **Ban `tensor.is_cuda` as a CUDA-vs-NPU Discriminant (`npu_tensor_is_cuda_not_cuda_npu_discriminant.md`):**
   - Under `torch_npu` privateuse1 device mapping, `x.is_cuda` returns `True` for NPU tensors. Branching on `if x.is_cuda:` to invoke `torch.ops._C.*` or Triton CUDA kernels silently enters the CUDA branch and fails with missing symbol or invalid device errors.
   - Always discriminate via `x.device.type == "npu"` or explicit platform capability queries (`current_platform.is_npu()`).
3. **Ascend 310P Capability & `FRACTAL_NZ` Layout Boundaries (`ascend_310p_bringup_unusable_ops_and_nz_layout.md`):**
   - Ascend 310P supports only non-MLA, non-sparse attention (`use_mla=False, use_sparse=False`), eager execution (`--enforce-eager`; ACLGraph capture triggers `ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE`), and `torch.float16` (no native BF16 or Triton kernels).
   - Avoid `MLAPO`, `batch_matmul_transpose`, ATB warmup, KV cache transfer, and shared KV tensors on 310P. Attention KV cache and attention mask tensors on 310P must be explicitly converted to `ACL_FORMAT_FRACTAL_NZ` (`torch_npu.npu_format_cast(..., 29)`).
4. **Multi-NIC PD Advertise Address Pinning (`multinic_pd_advertise_address_launch_config.md`):**
   - In multi-NIC Prefill-Decode (PD) disaggregated deployments, never rely on runtime network-interface probing (`socket.gethostbyname` or default route auto-detection) to advertise ZMQ/HIXL pull endpoints. Explicitly pin `rank_ip` / `advertise_ip` and `GLOO_SOCKET_IFNAME` / `HCCL_IF_IP` in launch configuration.
5. **AscendC Host Validation & MicroAPI Vector Stride Pitfalls (`ascendc_op_input_validation_pitfalls.md`, `ascendc_vector_kernel_silent_correctness_pitfalls.md`):**
   - Guard every optional `gert::TilingContext` input pointer (`tensor != nullptr`) individually before calling `GetShapeSize()`, never chain reversed boolean conditions with `&&` instead of `||` in error checks, and mirror shape/dtype validation in the PyTorch `Meta` implementation.
   - In MicroAPI (`__VEC_SCOPE__`) vector kernels processing multi-row UB buffers (e.g., `kv_compress_epilog`), compute row offsets from the hardware-aligned row stride (`RoundUp(colNum, align)` -> `curColNumAlign`), never by accumulating unaligned logical column counts across loop iterations; keep vector register variables scoped strictly inside `__VEC_SCOPE__`.
