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
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: 'Hermes workflow: 0-day Qwen3.5 GatedDeltaNet + Attention bring-up on vllm-ascend'
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
  - Hermes
  - reviewed_change
  - Ascend910_93
---

# Hermes workflow: 0-day Qwen3.5 GatedDeltaNet + Attention bring-up on `vllm-ascend`

## When to use

Use this workflow for a new hybrid checkpoint whose layers alternate GatedDeltaNet (GDN) linear attention and full attention — the `Qwen3.5-0.8B` family is the reference case. The goal is a staged bring-up on Ascend 910B/910_93: load the checkpoint, prove eager prefill parity against a CPU `fp32` reference, adapt decode metadata for GDN, then close the loop with a multi-step prefill+decode parity gate and a Hermes `knowledge_distill` into a `reviewed_change`.

This is not a performance-tuning workflow and not a substitute for the model owner's acceptance. It produces an evidence trail; a passing gate at one model revision does not generalize to another.

## Preconditions

Record the repository and exact revision, NPU model/die count, CANN/driver and torch/torch_npu versions, `dp`/`tp`/`pp`, graph mode, checkpoint repository and revision, config (`layer_types`, `full_attention_interval`, conv kernel size, head counts, `partial_rotary_factor`), dtype, cache mode and speculative configuration. Freeze the comparison tolerance before looking at target outputs.

For Qwen3.5-0.8B, the source record notes 488 weight-index keys and a 1,746,942,600-byte full-weight file. Verify those against the checkpoint actually loaded; do not copy the number into a different revision.

## Step 1: Mount the L1 runtime and parity gates, plus the L2 adaptation

Load:

1. `unit-vllm-ascend-npu-runtime-and-graph-invariants` — `execute_model` / `_dummy_run` / `_build_attn_group_metadata`, stateful GDN versus paged attention, decode/prefill indexing and `torch.ops._C_ascend` / NZ invariants.
2. `unit-vllm-ascend-verification-and-state-parity-gates` — Probe 1/2/3 and the Hermes distillation gate.
3. `unit-vllm-ascend-qwen35-gated-deltanet-decode-adaptation` — the historical `_forward_core` adaptation, its registration guard, and the source-loss boundaries.

Then state the bring-up hypothesis in falsifiable form: the checkpoint loads with no unexpected keys; the eager GDN prefill matches the CPU `fp32` reference layer by layer; and the decode metadata produces the same first token for a single-token decode and a multi-token continuation with the same prefix. Each clause is a separate gate.

## Step 2: Weight/config loader and layer-by-layer eager prefill parity

### 2a. Loader gate

- Resolve the text config, not the composite multimodal wrapper, for the text-only lane. Confirm `AutoModelForCausalLM` unwraps to `Qwen3_5TextConfig` where the installed stack expects it.
- Verify the full-weight file hash and load all 488 index keys. Fail on unexpected text-model keys, missing GDN keys, or silently skipped `conv_state`/`ssm_state` parameters.
- Read the actual `layer_types`; do not assume the 3:1 linear:full default when the checkpoint config supplies its own pattern.
- Confirm the GDN patch is active: `AscendQwen3_5GatedDeltaNet._forward_core` is assigned onto `Qwen3_5GatedDeltaNet._forward_core`, and the version guard did not skip the import.
- Confirm `torch.ops._C_ascend.causal_conv1d_fn` and the Triton modules are available before the first forward. A missing binding is a runtime precondition failure, not a model bug.

### 2b. Eager prefill parity

Run one short text-only prompt with batch size 1, eager attention, no graph capture and no speculative decoding. Compare against a CPU `fp32` reference:

- per-layer hidden state after the GDN mixer and after the full-attention block;
- `conv_state` after the causal conv, including the `kernel_size - 1` left context;
- `ssm_state` before and after the chunk delta rule;
- final logits for every prompt position.

Apply the pre-registered tolerance. A common failure is a correct final logit with a wrong intermediate state, so do not skip the layer checks. Check the `has_initial_state` branch: rows without an initial state must be zeroed before the chunk rule, and rows with one must gather the cached state.

**Exit:** a loader miss, a skipped layer, or a prefill logit outside tolerance stops the run. Do not proceed to decode with a broken prefill.

## Step 3: Decode metadata adaptation for GDN

Run a single-token decode and a speculative/accepted-token decode from the same prefill prefix. Assert the metadata contract directly:

- `query_start_loc` is the cumulative token-to-request map; GDN reads the unpadded `gdn_query_start_loc` prefix, not the padded attention prefix.
- `num_actual_tokens` selects the active rows; padded tail rows use `PAD_SLOT_ID`/`-1` and never alias a real request's state row.
- For speculative decode, `num_accepted_tokens` matches the verified draft count, and `spec_state_indices_tensor` plus the spec query-start prefix index the state. For non-speculative decode, the normal state indices and one-token query length are used.
- Prefill and decode do not share a stale accepted-token count or a synthetic one-token schedule.
- The first position of a multi-token cached continuation equals the same token fed as a single-token cached decode. Both paths must update `conv_state` and `ssm_state`; neither may leave the other's state behind.
- Padding-free and padded logits match, as the upstream tests require.

If a metadata field is wrong, fix the builder first and re-run Step 2; a decode-only patch on top of a bad prefill is not evidence.

## Step 4: Multi-step prefill+decode parity gate and Hermes `knowledge_distill`

### 4a. Multi-step gate

Run at least one prefill followed by several greedy decode steps (and, if enabled, one speculative round). Compare:

- teacher-forced logits and per-step greedy token IDs against the CPU `fp32` reference;
- `conv_state`/`ssm_state` continuity across the prefill/decode boundary;
- single-token versus multi-token equivalence at the first continuation position;
- repeat-run determinism and the absence of NaN/Inf;
- the same behavior with SP on and off, or a recorded reason why one path is out of scope.

**Exit:** only a full prefill+decode sequence that passes every check is a Probe 3-level result for this model. A first-token match alone is not.

### 4b. Hermes distillation

After the gate passes, run `knowledge_distill` with:

- the verified loader, prefill and decode root causes or confirmations;
- refuted hypotheses, each tied to the failing or passing probe;
- the exact checkpoint/revision, layer pattern, dtype, graph mode and topology;
- the source-loss and environment boundaries from the L2 unit;
- the evidence references and the `content_sha256` of the units to change.

Render a `mindie-reviewed-change/1` plan with `base_generation`, complete unit bytes and expected content hashes, then validate it with `scripts/apply_reviewed_change.py` before writing. If the gate fails, distill the failure instead: record the first divergent layer/token, mark the hypothesis unresolved or refuted, and do not promote the unit to `reproduced_empirical`.

## Anti-patterns

- Ingesting the multimodal wrapper for a text-only run and then blaming the GDN kernel.
- Comparing only the final logit while `ssm_state` is wrong.
- Reusing `query_start_loc` from a padded attention batch for GDN state indexing.
- Treating a single-token decode pass as proof of speculative decode.
- Copying the 488-key/weight-size record to a different checkpoint revision.
- Publishing a `reviewed_change` from a source diff or a Probe 1 metadata test.

## Evidence scope and limits

The procedure and its gates are reproduced from the pinned `vllm-ascend` patch, the official Transformers reference and the upstream parity tests. They define what must be measured; the numerical receipt for a specific Qwen3.5-0.8B run belongs to the executing lane and must record the exact stack. This workflow does not assert a universal NPU tolerance, throughput, or serving acceptance, and it does not cover vision inputs, quantized cache, MTP acceptance or a different model family.
