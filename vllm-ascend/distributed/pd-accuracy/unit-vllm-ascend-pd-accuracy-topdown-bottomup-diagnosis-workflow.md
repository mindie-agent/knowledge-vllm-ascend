---
category: workflow
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
domain_path: vllm-ascend/distributed/pd-accuracy
evidence_strength: software_contract
id: unit-vllm-ascend-pd-accuracy-topdown-bottomup-diagnosis-workflow
inherits_from:
- unit-vllm-ascend-pd-kv-connector-layout-and-handoff-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/b61c9b207f72e7765bf096a5dff2fc928a19b03f/vllm_ascend/worker/model_runner_v1.py#L2661
  - https://github.com/vllm-project/vllm-ascend/blob/b61c9b207f72e7765bf096a5dff2fc928a19b03f/vllm_ascend/ops/triton/mamba/mamba_hybrid.py#L102-L197
  - https://github.com/vllm-project/vllm-ascend/blob/b61c9b207f72e7765bf096a5dff2fc928a19b03f/vllm_ascend/ascend_config.py#L657
  - https://github.com/vllm-project/vllm-ascend/blob/b61c9b207f72e7765bf096a5dff2fc928a19b03f/vllm_ascend/batch_invariant.py#L102
  - https://github.com/vllm-project/vllm/blob/7913c96894cacc348397a377411acde96b63e68a/vllm/v1/core/sched/scheduler.py#L2170-L3218
  source_locator: vllm-project/vllm-ascend:b61c9b207f72e7765bf096a5dff2fc928a19b03f:vllm_ascend/worker/model_runner_v1.py
  source_type: upstream_code_audit_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: 'Hermes workflow: vllm-ascend PD disaggregation accuracy top-down and bottom-up diagnosis'
triggers:
  error_signatures:
  - PD disaggregated decode outputs gibberish, repeated n-grams, punctuation loops, or immediate EOS while colocated serving is normal
  - First decode token on D diverges from colocated baseline after remote KV transfer
  - Enabling ACLGraph, prefix caching, chunked prefill, MTP, or DCP breaks output accuracy only under PD disaggregation
  keywords:
  - PD accuracy debug
  - gibberish
  - repeated_ngram
  - punctuation_run
  - immediate_eos
  - sentinel fill
  - wait_for_layer_load
  - batch_invariant
  - enable_fused_mc2
  - 2x2 ablation matrix
  - step-0 logits
  - workflow
---

# Hermes workflow: vllm-ascend PD disaggregation accuracy top-down and bottom-up diagnosis

## Objective & Entry Criteria

Use this workflow when `vllm-ascend` PD disaggregated serving produces garbled text, repetitive loops (`repeated_ngram`, `punctuation_run`), early EOS, or logit drift compared to a colocated single-instance baseline.

## Stage 1 — Top-Down Triage: Split Step-0 Handoff vs Later Decode Steps

1. Fix greedy decoding (`temperature=0`, `top_k=1`, fixed `seed`, `concurrency=1`) and confirm non-zero remote KV transfer with zero local recompute fallback (`scheduler.py:2170`).
2. Compare D's **step-0 logits** (from the `N-1` handoff token) against the colocated baseline:
   - **Branch A (Step-0 logits diverge):** P-to-D KV/state transfer, block-stride addressing, TP/PCP/DCP mapping, or proxy forwarding is broken. Proceed to Stage 2 and Stage 3.
   - **Branch B (Step-0 matches, step $\ge 1$ diverges):** Initial transfer succeeded; inspect D-side decode KV append, recurrent state updates (`mamba_hybrid.py:102,197`), or ACLGraph replay (`unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop`).

## Stage 2 — 2×2 Feature Isolation Matrix (`(Colocated vs PD) × (Feature Off vs On)`)

For suspect features (`batch_invariant`, `enable_fused_mc2`, prefix caching, chunked prefill, ACLGraph, MTP, PCP/DCP), run the 4-cell matrix (`Colocated Off/On`, `PD Off/On`) and verify effective post-init config (`ascend_config.py:657`, `batch_invariant.py:102`):
- **`Colocated On` fails:** Single-instance kernel/config bug; stop debugging PD transfer.
- **`PD Off` fails while `Colocated Off` passes:** Baseline PD connector/layout bug; debug with Feature Off.
- **Only `PD On` fails:** Feature-specific PD interaction (prefix-cache reuse race, chunked-prefill partial save, or DCP interleave).

## Stage 3 — Bottom-Up Layer & Tensor Localization

1. **1-block vs multi-block probe:** If a 1-block prompt (`len < block_size`) passes and $\ge 2$ blocks fail, audit stride coalescing (`(N - 1) * block_stride + valid_bytes`) and DCP/PCP block indexing (`unit-vllm-ascend-pd-connector-layout-compatibility-audit-sop`).
2. **Consumption-point dumps:** Dump P-saved and D-loaded tensors at layer 0 **after** layer wait and state relocation finish (`model_runner_v1.py:2661`, `mamba_hybrid.py:102,197`), right before the consuming attention/GDN op—never at `execute_model` entry.
3. **Dtype-aware sentinel fill:** Pre-fill **exclusively owned** D destination blocks with a dtype-safe sentinel (`NaN` for `fp16`/`bf16`, extreme finite value for `int8`) before `start_load_kv`. Surviving sentinels in the active token/state slice prove unwritten holes. Never poison shared prefix-cache blocks or page-shared sibling views (`layer_stride == 0`).
