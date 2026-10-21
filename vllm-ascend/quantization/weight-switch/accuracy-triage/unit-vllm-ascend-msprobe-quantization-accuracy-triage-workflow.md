---
category: workflow
conditions:
  component: msprobe_accuracy_triage
  framework: vllm-ascend
domain_path: vllm-ascend/quantization/weight-switch/accuracy-triage
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-msprobe-quantization-accuracy-triage-workflow
inherits_from:
- unit-vllm-ascend-verification-and-state-parity-gates
- unit-vllm-ascend-deterministic-batch-invariance-and-eval-axioms
- unit-vllm-ascend-modelslim-w8a8-w4a16-w4a8-c8-and-quarot-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/performance_and_debug/msprobe_guide.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/quantization.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/quantization/modelslim_config.py
  source_digest: 90b6290a295511211947d005cba3b59b94aeeb737c654f7aa660ea5e7715f0a2
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/performance_and_debug/msprobe_guide.md
  source_type: verification_and_debug_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: msprobe layerwise tensor dump and BF16-vs-quantized accuracy bisection workflow
triggers:
  error_signatures:
  - quantized model emits gibberish, repeated punctuation, or >2% AISBench accuracy
    drop vs BF16
  - layer output cosine similarity drops below 0.98 after W8A8, W4A8, or KV C8 quantization
  - first token logits match in eager mode but diverge when fused MoE or npugraph_ex
    is enabled
  keywords:
  - msprobe
  - PrecisionDebugger
  - config.json
  - cosine_similarity
  - max_abs_err
  - quantization accuracy
  - layerwise dump
  - outlier
---
# msprobe layerwise tensor dump and BF16-vs-quantized accuracy bisection workflow

## Trigger and scope

Use this workflow when a quantized (`W8A8`, `W4A8`, `W4A16`, `MXFP8`, `MXFP4`, `KV_C8`) or fused-kernel `vllm-ascend` deployment exhibits accuracy degradation, NaN/Inf, or greedy token divergence compared to the BF16/FP16 reference (`docs/source/developer_guide/performance_and_debug/msprobe_guide.md`).

## Stage 1 — Freeze deterministic single-request reproduction

1. **Eliminate runtime non-determinism before dumping tensors:**
   - Set `--enforce-eager`, `--no-enable-prefix-caching`, `--seed 1024`, `temperature=0`, `max_tokens=1` (prefill logit check), and `export VLLM_BATCH_INVARIANT=1 PYTHONHASHSEED=0`.
   - Select one minimal failing prompt (and one long-context prompt if the failure only appears above `compress_ratio * topk` or chunked-prefill boundaries).

## Stage 2 — Capture layerwise tensor dumps with `msprobe` (`PrecisionDebugger`)

1. **Configure `msprobe` `config.json` (`docs/source/developer_guide/performance_and_debug/msprobe_guide.md`):**
   - Use `task: "statistics"` (fast MD5/min/max/norm/NaN summary) first to locate the exact layer where range explosion or NaN begins, then switch to `task: "tensor"` scoped to the suspect layer range (`scope: ["model.layers.0", "model.layers.1", ...]`).
   - Instrument `NPUModelRunner.execute_model` with `PrecisionDebugger(config_path=...)`: call `debugger.start(model=self.model)` before `self.model(...)`, `debugger.stop()`, and `debugger.step()` after forward.
2. **Compare BF16 vs quantized / candidate dumps (`msprobe compare`):**
   - Run `msprobe` comparison across the BF16 oracle dump and the quantized candidate dump. Inspect `Cosine`, `MaxAbsErr`, and `MaxRelativeErr` for every module (`input_layernorm`, `qkv_proj`, `rotary_emb`, `attn`, `o_proj`, `post_attention_layernorm`, `mlp.gate`, `mlp.experts`).

## Stage 3 — Remediate the localized quantization or fusion root cause

1. **Apply targeted scheme fallback or outlier suppression:**
   - **Attention / `o_proj` / `down_proj` outlier collapse (`Cosine < 0.98`):** Exclude sensitive layers (such as the first/last dense layers, `mlp.gate`, or `kv_a_proj_with_mqa`) to `FLOAT` in `msmodelslim` anti-outlier calibration (`M3`/`M4`/`M6` SmoothQuant or QuaRot/FlatQuant rotation), or switch sensitive projections from static `W8A8` to `W8A8_DYNAMIC`.
   - **Fused-pass divergence (eager quantized passes, graph/fusion fails):** Toggle `fuse_norm_quant: false`, `fuse_qknorm_rope: false`, or `VLLM_ASCEND_ENABLE_FUSED_MC2=0` to isolate the faulty fusion pass.
