---
category: sop
conditions:
  model_family:
  - qwen3_5
  - Qwen3.5
  - gated_deltanet
  model_repository: Qwen/Qwen3.5-0.8B
  model_revision: 2fc06364715b967f1860aea9cf38778875588b17
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - vllm-ascend-idle-dp
  source0_repository: https://github.com/vllm-project/vllm-ascend
  source0_revision: a78a00e0b1b618f26a3bfce2519055b40cf4bf1e
domain_path: vllm-ascend/runtime/qwen35-gated-deltanet-decode
evidence_strength: single_observation
id: unit-vllm-ascend-qwen35-gated-deltanet-decode-adaptation
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/0f289fa2a8a381a644a5f329af61715a73ec2aec/vllm_ascend/patch/worker/patch_qwen3_5.py
  - https://github.com/vllm-project/vllm-ascend/blob/0f289fa2a8a381a644a5f329af61715a73ec2aec/vllm_ascend/patch/worker/__init__.py
  - https://github.com/vllm-project/vllm-ascend/blob/0f289fa2a8a381a644a5f329af61715a73ec2aec/vllm_ascend/patch/__init__.py
  - https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/models/qwen3_5/modeling_qwen3_5.py
  experiment_status: not_run_source_only
  source_digest: d48fbf9acefff670b66fc9941de53b725607d8239153bad1ee7b79fdacd56bdd
  source_locator: vllm-project/vllm-ascend@0f289fa2a8a381a644a5f329af61715a73ec2aec:vllm_ascend/patch/worker/patch_qwen3_5.py
  source_type: public_source
  supersedes: []
revision: 3
schema: mindie-graded-unit/1
tier: L2
title: Qwen3.5 GatedDeltaNet decode adaptation in vLLM-Ascend (historical PR7109)
triggers:
  error_signatures:
  - torch_npu.npu_recurrent_gated_delta_rule does not support ssm_state inputs in
    float32 format
  - Qwen3_5GatedDeltaNet reuses Qwen3NextGatedDeltaNet._forward_core but AscendC ops
    lack float32 ssm_state support
  - worker import is version-gated by not vllm_version_is("v0.16.0")
  keywords:
  - qwen3.5
  - qwen3_5
  - gated deltanet
  - linear attention
  - ssm_state
  - vllm-ascend
  - text-only prefill
  - recurrent decode
  - chunk_gated_delta_rule
  - Ascend910_93
---

# Qwen3.5 GatedDeltaNet decode adaptation (historical PR7109)

## When to use

Qwen3.5 hybrid GDN + full-attention checkpoint on `vllm-ascend` where `torch_npu.npu_recurrent_gated_delta_rule` does not accept a float32 `ssm_state`, so decode must run GDN through fused FLA/Triton operators. This is source-derived, not an execution result; the conditions name the model and pinned revisions.

## Adaptation

At revision `0f289fa2a8a381a644a5f329af61715a73ec2aec`, `vllm_ascend/patch/worker/patch_qwen3_5.py` is imported from the worker `__init__` under `if not vllm_version_is("v0.16.0")`. It assigns `AscendQwen3_5GatedDeltaNet._forward_core` onto `Qwen3_5GatedDeltaNet._forward_core` — a runtime monkeypatch, not a registered model class. The replacement keeps `conv_state`/`ssm_state` indexing and combines vLLM chunk/fused-recurrent delta rules with Ascend fused gating.

## SOP

1. Verify the guard actually imported the patch. A different task-start revision can fail at import or attribute access before any model runs.
2. Verify `torch.ops._C_ascend.causal_conv1d_fn` and the Triton modules exist; the worker import only guards `patch_triton` with `HAS_TRITON`, not this patch.
3. Metadata: read `GDNAttentionMetadata` by `self.prefix`; take query start locations, state indices, `num_actual_tokens`, `num_accepted_tokens` and prefill/decode counts; truncate to `num_actual_tokens` unless `enable_sp()`.
4. Prefill: gather cached `ssm_state` for rows with `has_initial_state` and zero rows without it; call the chunk rule with `output_final_state=True`; write the final state back at the state indices.
5. Decode: split speculative and non-speculative streams by `spec_sequence_masks`. Non-spec uses normal state indices and one-token/accepted length; spec uses `spec_state_indices_tensor` and `num_accepted_tokens`. Never reuse a padded attention `query_start_loc` for GDN.
6. Validate with the L1 gates: loader, eager prefill/decode against CPU `fp32`, conv/ssm continuity, single-token vs multi-token first-position equality, both `has_initial_state` branches, SP on/off.

## Boundaries

The patch changes only `_forward_core`; it does not implement checkpoint loading, config/tokenizer, scheduler or server. The text lane uses `Qwen3_5TextConfig`; the conditional-generation class and eagle model-name list are separate. Read the checkpoint's actual `layer_types`; the 3:1 linear:full pattern is only the default when `layer_types` is absent. The rationale is float32 state compatibility, not measured NPU precision. Quantized cache is incompatible with this hybrid cache.
