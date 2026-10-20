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
domain_path: vllm-ascend/optimizations/host-scheduling-and-batching/host-overhead
evidence_strength: software_contract
id: unit-vllm-ascend-host-scalar-precompute-and-preprocess-elimination-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/decode_host_scalar_precompute_passthrough.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/inference_preprocess_hotpath_host.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/experience/interventions/graph_break_elimination_host_sync_unsupported_ops.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/aicpu_scheduler.md
  source_digest: 6c35e4543fa36b99cc08682c2826e3abb6a3567d04397545b77ef841ce9d156c
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/decode_host_scalar_precompute_passthrough.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Host scalar precomputation, input preprocessing hot-path optimization, and
  AICPU scheduler offload in vLLM-Ascend
triggers:
  error_signatures:
  - NPU profiler shows large host dispatch gaps and high Free time between decode
    kernels
  - Repeated .item() or .tolist() calls inside forward_decode trigger host-device
    synchronization
  - Large-batch tokenization or per-step Python list comprehensions cause host CPU
    bottleneck or host OOM
  keywords:
  - decode_host_scalar_precompute
  - inference_preprocess_hotpath_host
  - aicpu_scheduler
  - actual_seq_lengths
  - softmax_scale
  - host_bound
  - free_time
---
# Host scalar precomputation, input preprocessing hot-path optimization, and AICPU scheduler offload in vLLM-Ascend

## Applicability and symptom scope

Use this SOP when NPU profiling reveals that decode step latency is host-dispatch bound (`Host to Device` launch gap / NPU `Free` time > `15%` of step time) or when graph capture breaks on host scalar extractions.

## Procedure

1. **Hoist invariant scalars and quantization scales out of the per-layer decode loop (`decode_host_scalar_precompute_passthrough.md`).**
   - Compute `softmax_scale = head_dim ** -0.5`, RoPE `cos/sin` tables, static `dequant_scale` reciprocal views, and layer-invariant attention metadata once during model initialization or once per step in `ModelRunner.prepare_inputs`, passing them through `attn_metadata` to all transformer layers.
   - Never reconstruct scalar tensors (`torch.tensor(scale, device="npu")`) or recompute `1.0 / weight_scale` inside each layer's `forward()`.
   - **Invariant:** Zero per-layer Python float-to-tensor allocations or redundant scale divisions occur during `forward_decode`.

2. **Eliminate `.item()`, `.cpu()`, and Python loops in input preprocessing (`inference_preprocess_hotpath_host.md`, `graph_break_elimination_host_sync_unsupported_ops.md`).**
   - Replace per-request Python loops that build `positions`, `slot_mapping`, and `actual_seq_lengths` with vectorized NumPy / PyTorch CPU tensor operations transferred in a single pinned-memory `non_blocking=True` H2D copy.
   - When `torch.ops.npu` fused attention requires a Python `list[int]` for `actual_seq_lengths_kv` in eager/npugraph_ex mode, extract it once per step in `attn_metadata` before entering the layer loop rather than calling `.tolist()` per layer.
   - Chunk oversized batch tokenization requests on the host to prevent host RAM spikes (`EH0010`).

3. **Offload value-dependent dynamic tiling dispatch to AICPU Scheduler when supported (`aicpu_scheduler.md`).**
   - For operators whose core-split tiling depends on device-resident values (such as MoE `expert_token_num` or dynamic sparse `topk` counts), use the AICPU Scheduler mechanism to sink tiling computation and task dispatch onto device ARM AICPU cores instead of synchronizing counts back to the host CPU.
   - **Verify:** Profile 20 steady-state decode steps with `torch_npu.profiler` and verify that per-layer `.item()` / `aclrtSynchronizeStream` calls are zero and NPU `Free` gap ratio drops below `10%`.
