---
category: workflow
conditions:
  model_family:
  - qwen3_5
  - Qwen3.5
  - gated_deltanet
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
domain_path: vllm-ascend/runtime/qwen35-gated-deltanet-decode
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-0day-hybrid-model-bringup-hermes-workflow
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/0f289fa2a8a381a644a5f329af61715a73ec2aec/vllm_ascend/patch/worker/patch_qwen3_5.py
  - https://github.com/vllm-project/vllm-ascend/blob/0f289fa2a8a381a644a5f329af61715a73ec2aec/vllm_ascend/patch/worker/__init__.py
  - https://github.com/vllm-project/vllm-ascend/blob/0f289fa2a8a381a644a5f329af61715a73ec2aec/vllm_ascend/patch/__init__.py
  - https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/models/qwen3_5/modeling_qwen3_5.py
  - https://huggingface.co/Qwen/Qwen3.5-0.8B
  source_digest: d48fbf9acefff670b66fc9941de53b725607d8239153bad1ee7b79fdacd56bdd
  source_locator: vllm-project/vllm-ascend@0f289fa2a8a381a644a5f329af61715a73ec2aec:vllm_ascend/patch/worker/patch_qwen3_5.py
  source_type: hermes_workflow_distillation
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
tier: L3
title: '0-day Qwen3.5 GatedDeltaNet + Attention bring-up workflow on vllm-ascend'
triggers:
  error_signatures:
  - torch_npu.npu_recurrent_gated_delta_rule does not support ssm_state inputs in
    float32 format
  - Qwen3.5 GDN prefill logits diverge from the CPU float32 reference
  - GDN decode uses a stale query_start_loc or num_accepted_tokens
  - single-token decode and multi-token continuation disagree at the first position
  keywords:
  - Qwen3.5
  - qwen3_5
  - GatedDeltaNet
  - decode
  - prefill
  - query_start_loc
  - num_accepted_tokens
  - speculative decode
  - ssm_state
  - conv_state
  - workflow
  - reviewed_change
  - Ascend910_93
---

# 0-day Qwen3.5 GatedDeltaNet + Attention bring-up workflow

## Trigger

A hybrid checkpoint alternates GDN linear attention and full attention — `Qwen3.5-0.8B` is the reference case — on Ascend 910B/910_93. The goal is staged: load weights, verify eager prefill numerical parity, adapt decode metadata, verify multi-step stability, and then distill the verified findings.

## First discriminating observation

Before any NPU run, check loader and patch binding: does the text config resolve, do the expected index keys load, does `AscendQwen3_5GatedDeltaNet._forward_core` actually replace the model core, and do `_C_ascend.causal_conv1d_fn` and Triton import? A failure here is an environment or loader issue rather than a model-parity bug.

## Branches

- **A. Loader or import failure** (missing/unexpected key, skipped guard, missing op): fix the environment or revision first. Do not interpret later logits; exit after the loader gate is clean.
- **B. Loader clean, eager prefill diverges:** compare per-layer hidden states after the GDN mixer and attention, `conv_state` after the causal conv, `ssm_state` before/after the chunk rule, then final logits against CPU `fp32`. A correct final logit with a wrong intermediate state is a failure.
- **C. Prefill clean, decode diverges:** check the unpadded GDN `query_start_loc`, `num_actual_tokens`/padding sentinels, `num_accepted_tokens`/`spec_state_indices_tensor`, and single-token vs multi-token first-position equality. Fix the builder and re-run prefill.
- **D. Prefill and decode clean but multi-step or SP unstable:** run repeated greedy decode and SP on/off with determinism and NaN checks before treating multi-step inference as verified.

## Prerequisites

Record revision, NPU/die count, CANN/torch stack, `dp`/`tp`/`pp`, graph mode, checkpoint repo/revision, `layer_types`, `full_attention_interval`, conv kernel, head counts, `partial_rotary_factor`, dtype, cache and speculative config. Freeze tolerance before seeing outputs.

## Exploration exit

Stop when loader, layer-by-layer eager prefill, decode metadata and the multi-step gate pass for the same code state. Then distill verified causes and refuted hypotheses into a `reviewed_change`; a probe pass is not a reviewed change. If a gate fails, record the first divergent layer/token and do not promote.

## Pitfalls

Ingesting the multimodal wrapper for the text lane; comparing only the final logit; reusing a padded `query_start_loc`; treating single-token decode as speculative proof; copying the 488-key/weight-size record to another revision; publishing from a source diff.
