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
  - zhaochuang001/awesome-agent-skills@836e239f80cb76c45a5dd667839e40f336789472:vllm-ascend-accuracy/SKILL.md
  - djh95/runtime-guard-docs@45bc2603f48f883a448ed7c2bbeed38f9a2398e1:06_agent_skills/msprobe-dump/SKILL.md
  source_digest: 929c5e7cc4f3893b98274ed56f63bc65dfaaeaaf6f80aad80bcdad0b25eb975e
  source_locator: zhaochuang001/awesome-agent-skills@836e239f80cb76c45a5dd667839e40f336789472:vllm-ascend-accuracy/SKILL.md
  source_type: verification_and_debug_guide
  supersedes: []
revision: 3
schema: mindie-graded-unit/1
tier: L3
title: vllm-ascend 4-symptom accuracy triage, 11-dimension ablation, runtime_guard/mindie-debugger
  handoff, and eager msprobe bisection workflow
triggers:
  error_signatures:
  - quantized model emits gibberish, repeated punctuation, or >2% AISBench accuracy
    drop vs BF16
  - layer output cosine similarity drops below 0.98 after W8A8, W4A8, or KV C8 quantization
  - first token logits match in eager mode but diverge when fused MoE or npugraph_ex
    is enabled
  - MTP speculative decoding or high-concurrency W8A8 serving produces step-2+ token
    collapse or language drift
  keywords:
  - msprobe
  - PrecisionDebugger
  - config.json
  - cosine_similarity
  - max_abs_err
  - quantization accuracy
  - layerwise dump
  - outlier
  - runtime_guard
  - enforce-eager
  - HCCL_DETERMINISTIC
---
# vllm-ascend 4-symptom accuracy triage, 11-dimension ablation, runtime_guard/mindie-debugger handoff, and eager msprobe bisection workflow

## Trigger and 4-symptom entry classification

Use this workflow when a `vllm-ascend` deployment exhibits accuracy regressions across any of the four canonical symptom classes (`zhaochuang001/awesome-agent-skills:vllm-ascend-accuracy`):

1. **Benchmark metric drop**: Evaluation accuracy (AISBench, GSM8K, CEval, LongBench) drops `> 2%` against the BF16/GPU baseline while outputs remain grammatically readable.
2. **Greedy output divergence**: `temperature=0, top_k=1` token sequence diverges from the reference implementation at a specific prefill or decode step.
3. **Severe degeneration**: Token repetition (`token_repeat`), gibberish, mid-sentence language switching, or infinite decode without EOS.
4. **Numerical collapse**: `NaN`/`Inf` in logits or attention outputs (`logits_finite`), or all-zero KV cache norm collapse (`target_zero_norm`).

## Stage 1 — Environment parity, live process audit, and deterministic freeze

1. **Audit the live serving process environment (not just the interactive shell):**
   - Run `mindie-debugger triage --pid <PID>` (or inspect `/proc/<PID>/environ` and `/proc/<PID>/maps`) to confirm the target worker process actually inherited `ASCEND_TOOLKIT_HOME`, `ATB_HOME_PATH`, `HCCL_DETERMINISTIC=true`, `LCCL_DETERMINISTIC=1`, and `VLLM_BATCH_INVARIANT=1`.
   - Verify checkpoint `.safetensors` MD5 hashes, `quant_model_description.json`, and `vllm` / `vllm-ascend` Git commits match across compared environments.
2. **Freeze a deterministic single-request reproduction:**
   - Fix `--seed 1024`, `temperature=0`, `top_k=1`, `--no-enable-prefix-caching`, and `export PYTHONHASHSEED=0 HCCL_DETERMINISTIC=true LCCL_DETERMINISTIC=1 VLLM_BATCH_INVARIANT=1`.
   - Select one minimal failing prompt (plus one long-context prompt when the failure only appears past `compress_ratio * topk` or chunked-prefill boundaries).

## Stage 2 — 11-dimension single-variable ablation matrix

Before launching heavy tensor dumps, toggle **one execution axis at a time** to bound the faulty subsystem:

| Dimension | Single-Variable Toggle | Subsystem / Historical Issue Anchor |
| --- | --- | --- |
| 1. Graph capture | Add `--enforce-eager` (`cudagraph_mode: "NONE"`) | Isolates `ACLGraph` capture/replay hazards (e.g., Qwen3-Next `FULL_DECODE_ONLY` accuracy drop `#3444`; Qwen2.5-VL `torch.compile` multimodal break `#2767`). |
| 2. Operator fusion | Set `fuse_qknorm_rope: false`, `fuse_norm_quant: false`, or `VLLM_ASCEND_ENABLE_FUSED_MC2=0` | Isolates custom fused pass or MC2 communication-computation fusion bugs. |
| 3. Speculative decoding | Remove `--speculative-config` (disable MTP / EAGLE) | Isolates draft-head KV slot mapping or step-2+ acceptance collapse (`#13863`). |
| 4. Quantization scheme | Compare `BF16` vs `W8A8` vs `W8A8_DYNAMIC` / `KV_C8` | Isolates static activation clipping or missing scale bindings (`#8359` Qwen3-235B W8A8 concurrency saturation; `#8989` DeepSeek-V3.2-Exp conversion). |
| 5. Attention backend | Toggle `MLA` / `SFA` / `FIA` path or `sparse_mode` | Isolates Sparse Flash Attention (`#8863`) or rotary layout (`#13646` Qwen3.5 `apply_eagle_rotary_emb`). |
| 6. Prefix caching | Add `--no-enable-prefix-caching` | Isolates prefix block reuse or hash collision corruption. |
| 7. Chunked prefill | Adjust `--max-num-batched-tokens` / chunked prefill | Isolates cross-chunk KV merge and `slot_mapping` boundary offsets. |
| 8. Concurrency & batching | Reduce `--max-num-seqs 1` vs high concurrency | Isolates batch padding tail leakage (`unmasked_padded_slot_mapping`) and workspace reuse. |
| 9. Parallelism (TP/DP/EP/CP) | Compare smaller TP/EP/DP or disable `enable_eplb` | Isolates collective reduction order, EP AllToAll routing, or CP online-softmax LSE merge. |
| 10. Context length | Sweep short (`< 512`) vs long (`> 4096`) prompt | Isolates `block_table` capacity (`len(block_ids) * block_size < n_tokens`) and RoPE scaling. |
| 11. Sampling vs prefill logits | Compare `max_tokens=1` prefill logit vs multi-step decode | Separates prefill forward corruption from decode KV write or sampler faults. |

## Stage 3 — Progressive 3-stage diagnostic funnel (`runtime_guard` -> `mindie-debugger` -> `msprobe`)

1. **Stage 3A — Online KV cache delimitation (`runtime_guard` + `force-feed`):**
   - When the anomaly occurs in `ACLGraph` or multi-rank serving, enable `additional_config.runtime_guard` (`token_repeat` + `logits_finite`, `dump_kv=true`) and run a same-mode, same-topology `force-feed` clean reference (`history = n_prompt + n_output - 1`).
   - Run numeric-first KV comparison (`unused_slot_status` -> `numeric_issue` before `cosine_sim` -> Table 1 token evolution & Table 2 layer chain at the first divergent token `T0`) to determine whether KV cache writes (`Q1-Q7`) or post-KV layers (`Q8-Q10`) caused the divergence.
2. **Stage 3B — In-graph & sub-layer isolation (`mindie-debugger`):**
   - **Graph vs. Eager divergence**: Run `mindie-debugger aclgraph inspect --compare-eager --script <repro.py>` to execute `ACLGraphCaptureGuard`, `ACLGraphAddressTracker` (checking `rebound_pointer`, `active_slot_out_of_bounds`, `unmasked_padded_slot_mapping`, and `block_table_capacity_insufficient`), and `ACLGraphDumper`.
   - **Sub-layer error amplification & MoE repro**: Use `LockstepDiffer(heal=True)` to separate true operator defects from upstream jitter, `MemoryWatchpoint` for stride-exact alias corruption, and `mindie-debugger precision slim-moe <in> <out> <k>` for single-card MoE repros.
3. **Stage 3C — Eager-only layerwise `msprobe` (`PrecisionDebugger`) dump (`docs/source/developer_guide/performance_and_debug/msprobe_guide.md`):**
   - **Mandatory `--enforce-eager` gate**: Always set `--enforce-eager` (`compilation_config: {"cudagraph_mode": "NONE"}`) before enabling `msprobe`. Registering `PrecisionDebugger` hooks under `ACLGraph` either breaks graph capture or records only the static warmup pass.
   - **Two-step `config.json` scope**: Start with `task: "statistics"`, `level: "L1"`, and `summary_mode: "statistics"` across the model to pinpoint the first layer with `NaN`/`Inf` or `MaxAbsErr` jump, then narrow `scope` (`["model.layers.K"]`) with `task: "tensor"` and run `msprobe compare` (`Cosine`, `MaxAbsErr`, `MaxRelativeErr`).
   - Instrument `NPUModelRunner.execute_model`: call `debugger.start(model=self.model)` before forward, `debugger.stop()`, and `debugger.step()` immediately after forward.

## Stage 4 — Remediate and report with 4-state honesty

1. **Apply targeted remediation:**
   - **Quantization outlier collapse (`Cosine < 0.98` in `o_proj` / `down_proj` / `mlp.gate`)**: Keep sensitive layers in `FLOAT` during `msmodelslim` calibration (`M3`/`M4`/`M6` SmoothQuant or QuaRot/FlatQuant rotation), or switch static `W8A8` to `W8A8_DYNAMIC`.
   - **Graph / fusion / MTP defect**: Fix the underlying unmasked `slot_mapping` tail, static buffer rebinding, or rotary/fusion predicate.
2. **Enforce the 4-state closure classification (never report a workaround as a root-cause fix):**
   - `候选修复` (*Candidate Fix*): Root-cause operator/config/weight patch implemented; awaiting full accuracy and throughput regression gates.
   - `已规避` (*Mitigated via Workaround*): Accuracy restored by disabling an optimization (`--enforce-eager`, disabling MTP/fusion, or falling back a layer to `FLOAT`), while the underlying defect remains open.
   - `已解决` (*Resolved*): Root cause fixed and verified under the original target serving mode and topology without accuracy or performance regression.
   - `受阻` (*Blocked*): Missing reference checkpoint, unaligned runtime environment, or hardware/container blocker explicitly documented with reproduction evidence.
