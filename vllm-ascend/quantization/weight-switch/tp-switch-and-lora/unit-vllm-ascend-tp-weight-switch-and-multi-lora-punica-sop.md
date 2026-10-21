---
category: sop
conditions:
  component: tp_weight_switch_and_lora
  framework: vllm-ascend
domain_path: vllm-ascend/quantization/weight-switch/tp-switch-and-lora
evidence_strength: software_contract
id: unit-vllm-ascend-tp-weight-switch-and-multi-lora-punica-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/tp_weight_switch.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/lora/punica_npu.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/lora/lora_ops.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/lora/fused_moe.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/lora/quant_moe.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/lora.md
  source_digest: fb9ee7eac66320b33b24c8beea2572575a6a4b47f2ffb8c0a437cdeeca05d641
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/tp_weight_switch.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Dynamic TP-to-DP weight switch (TPWeightSwitchManager) and PunicaNPU Multi-LoRA
  serving SOP
triggers:
  error_signatures:
  - TPWeightSwitchManager fails when switching attention or linear weights between
    prefill TP and decode DP
  - PunicaNPU bgmv_shrink or bgmv_expand fails on non-contiguous LoRA weight slices
    or quantized MoE LoRA
  keywords:
  - tp_weight_switch
  - TPWeightSwitchManager
  - PunicaNPU
  - punica_npu
  - lora_ops
  - quant_moe
  - bgmv_shrink
  - bgmv_expand
---
# Dynamic TP-to-DP weight switch (TPWeightSwitchManager) and PunicaNPU Multi-LoRA serving SOP

## Scope and preconditions

Use this SOP when operating dynamic Tensor-Parallel weight resharding (`vllm_ascend/quantization/tp_weight_switch.py`) or serving multi-LoRA adapters on dense, MoE, and quantized models via `PunicaNPU` (`vllm_ascend/lora/punica_npu.py`, `lora_ops.py`, `fused_moe.py`, `quant_moe.py`).

## Procedure

1. **Coordinate dynamic TP weight switching (`vllm_ascend/quantization/tp_weight_switch.py`):**
   - `TPWeightSwitchManager` enables runtime switching of linear/attention weight shards between different TP degrees (for example, high-TP prefill vs low-TP / high-DP decode or RL rollout resharding) using pre-registered weight descriptors (`ColumnParallel` vs `RowParallel` vs `MergedColumnParallel`) and quantized scale tensors (`deq_scale`, `weight_scale`, `weight_offset`).
   - Always synchronize the NPU stream and invalidate or switch the corresponding `ACLGraph` bucket before toggling weight views so captured graphs never read freed or resharded addresses.
2. **Execute Multi-LoRA via `PunicaNPU` (`vllm_ascend/lora/punica_npu.py`, `lora_ops.py`):**
   - `PunicaNPU` implements batched GQA/Linear/MoE low-rank adapter execution on Ascend NPU using `bgmv_shrink`, `bgmv_expand`, `bgmv_expand_slice`, `sgmv_shrink`, and `sgmv_expand` custom ops (`torch.ops._C_ascend`).
   - For quantized base models (`w8a8_static`, `w8a8_dynamic`, `w4a8`) with LoRA (`vllm_ascend/lora/quant_moe.py`, `fused_moe.py`), the base quantized GEMM/GMM executes in int8/int4 and dequantizes to FP16/BF16 before adding the FP16/BF16 LoRA delta `(x @ A) @ B * scale`.
3. **Configure Multi-LoRA capacity flags (`docs/source/user_guide/feature_guide/lora.md`):**
   - Bound `--max-loras`, `--max-lora-rank`, and `--max-cpu-loras` explicitly; when `ACLGraph` is enabled with LoRA, Punica token-to-adapter index buffers (`lora_indices`) are updated in place inside persistent device buffers before graph replay.

## Verification

- Run concurrent requests targeting `base_model`, `lora_adapter_1`, and `lora_adapter_2` in the same batch and verify that `base_model` requests match the `--no-enable-lora` greedy output bit-for-bit.
