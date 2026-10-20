---
category: workflow
conditions:
  analysis_scope:
  - compressed_tensors_quantization
  - stream_sample_perf_breakdown
  npu_model:
  - Ascend910B
  - Ascend910B2
  - Ascend910B3
  - Ascend910B4
  - Ascend910_93
  quant_method:
  - compressed-tensors
  - W8A8
  - W8A8C8
  - W4A8C8
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - cann/cann-recipes-infer
domain_path: vllm-ascend/performance/quantization-breakdown
evidence_strength: software_contract
id: unit-vllm-ascend-compressed-tensors-quantization-and-step-breakdown-workflow
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-quantization/SKILL.md#L108-L540
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-perf-breakdown/SKILL.md#L18-L366
  - https://gitcode.com/cann/cannbot/blob/d0f8c40ebc208fa17323781c89e4ec6ea16e4c52/plugins-official/model-infer-optimize/workflows/optimize-workflow.md#L122-L158
  - https://gitcode.com/cann/cannbot/blob/d0f8c40ebc208fa17323781c89e4ec6ea16e4c52/plugins-official/model-infer-optimize/workflows/references/decision-rules.md#L62-L113
  experiment_status: not_run_source_only
  source_digest: 661c95a4fa72e27243ed0f614b91476b240c223ae2092e410f93fbea6acb6ebe
  source_locator: cann/cannbot-skills:f305e200aa4b1e1fed73cd21ff1f083b8df25a1e:model/model-infer-quantization/SKILL.md
  source_type: cannbot_skill_and_source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Compressed-tensors quantization admission, post-load fusion recovery, and stream-sample performance breakdown workflow
triggers:
  error_signatures:
  - quantized checkpoint splits fused gate_up_proj or qkv_proj into separate small matmuls and regresses decode latency
  - Linear module absent from compressed-tensors ignore list lacks weight_scale in safetensors index
  - auxiliary_stream_temporally_displaced or compute-core unmatched ratio exceeds 5% during kernel_details.csv breakdown
  - quantized model loads without error but falls back to UnquantizedLinearMethod or produces garbled greedy tokens
  keywords:
  - compressed-tensors
  - quantization_config
  - process_weights_after_loading
  - QuantBatchMatmul
  - DequantSwigluQuant
  - DynamicQuant
  - UnquantizedLinearMethod
  - kernel_details.csv
  - stream_sample_ack
  - auxiliary_stream_temporally_displaced
  - bubble_ms
  - operator_jitter
---

# Compressed-tensors quantization admission, post-load fusion recovery, and stream-sample performance breakdown workflow

## Objective & Scope

Use this workflow when integrating `compressed-tensors` (`W8A8`, `W8A8C8`, `W4A8C8`) quantized checkpoints into Ascend inference runners and attributing step-level latency changes across prefill/decode components using `kernel_details.csv` and `trace_view.json`. It enforces strict contract admission, runtime activation proof, and stream-aware component breakdown without fabricating scales or mutating upstream checkpoint configs.

---

## Stage 1 — Audit the `compressed-tensors` Artifact Contract & Readiness Tier

1. **Verify the three artifact files before touching model code**:
   - `config.json`: must declare `quant_method: "compressed-tensors"`, `config_groups`, `targets`, and `ignore`.
   - `model.safetensors.index.json`: must map every targeted module prefix to explicit quantized tensors (`weight`/`qweight`, `weight_scale`, `weight_offset`/`zero_point`, `smooth_scale`).
   - `deploy_quantization.md`: must declare quantized vs float-fallback modules.
2. **Handle incomplete `ignore` lists without mutating `config.json`**:
   - If a `Linear` module is absent from `ignore` and has no `weight_scale` in `model.safetensors.index.json`, check `deploy_quantization.md`:
     - If the module is outside the declared quantization scope (e.g., `lm_head` or MTP head when only `moe, mlp` are targeted), inject its prefix into the runtime `CompressedTensorsConfig.ignore` set so `get_quant_method` selects `UnquantizedLinearMethod`, and log a contract-completion request for the producer.
     - If the module is inside the declared quantization scope or ambiguous, **stop** (`L3` contract block) and request a corrected export. Never synthesize dummy scales or modify `config.json`.
3. **Grade per-module readiness (`L0`–`L3`) by structural fingerprint** (`Dense Decoder`, `MoE Decoder`, `MLA + MoE / KVCache`, `Indexer + MoE`):
   - `L0`: YAML/config alignment only.
   - `L1`: Model prefix mapping, `process_weights_after_loading` transpose/NZ conversion, or post-load fusion recovery required.
   - `L2`: Missing quantized runtime kernel/method for the target structure — stop and escalate as a runtime gap.
   - `L3`: Incomplete artifact contract — stop before code edits.

---

## Stage 2 — Execute Post-Load Fusion Recovery & Three-Probe Activation Verification

1. **Restore runtime fusion in `process_weights_after_loading`**:
   - Quantized exports frequently split fused `gate_up_proj` (`w1`/`w3`) or `qkv_proj` into separate `Linear` weights. Re-concatenate `qweight` and `weight_scale` in `process_weights_after_loading` (when shared input scale/smooth-scale contracts hold), transpose weights as required by `QuantBatchMatmul` / `MoEGMM`, and convert to Fractal NZ format so dynamic quantization (`DynamicQuant`) and `QuantBatchMatmul` are not fragmented into small launches.
   - For mixed-bitwidth MoE, require a dedicated `targets: ["MoEGMM"]` config group.
   - If a fused operator rejects the quantized dtype/scale/layout, do not alter `targets`/`ignore`; revert that module to its non-fused quantized operator path (once approved) and record the fused-quant operator requirement.
2. **Prove quantization is genuinely active with three ordered probes**:
   - **Probe-B (Object probe, zero profiling overhead)**: Inspect loaded layers and assert `type(layer.quant_method).__name__ != "UnquantizedLinearMethod"`.
   - **Probe-C (Weight load reconciliation)**: Reconcile loaded `qweight` / `weight_scale` tensor counts against the expected quantized module count.
   - **Probe-A (Kernel profile probe)**: In the profiling `op_statistic` / `kernel_details.csv`, confirm execution of `QuantBatchMatmul`, `GroupedMatmul`, `DequantSwigluQuant`, or `DynamicQuant` (`count > 0`), and verify greedy output equivalence against the BF16 baseline (no garbled text, early EOS, or runaway repetition).

---

## Stage 3 — Decompose Single-Step Latency & Quantization Overhead (`kernel_details.csv`)

When evaluating prefill/decode performance or diagnosing why quantization or fusion failed to reduce wall-clock time, decompose one steady step from `kernel_details.csv` (skipping step 0 warmup):

1. **Filter noise rows deterministically**:
   - Exclude `accelerator_core in {AI_CPU, AICPU}` when building row-ordered structural samples.
   - When both HCCL collective summary rows and `Name == AivKernel` communication task fragments exist, keep the collective summary row and drop matched fragments (retaining only unmatched fragments).
2. **Partition into component instances with `stream_sample_ack.v1`**:
   - Assign exactly one `primary` stream per component instance (`mla`, `dense`, `moe`, `mtp`) and tag semantically owned secondary streams as `auxiliary`.
   - If an `auxiliary` stream is temporally displaced from its component's primary window (`auxiliary_stream_temporally_displaced`), exclude its displaced ops from the component's local `cluster / bubble / TOTAL` median so cross-layer overlap does not distort local bubble accounting, while keeping its `op_indices` in the partition.
   - Enforce the **unmatched compute-core gate**: compute-core unmatched ops must be $\le 5\%$ of step compute duration before accepting cluster tables.
3. **Classify findings into five actionable insight buckets**:
   - `module_bubble`: Free-time stream gaps (`bubble_ms`) between consecutive kernels inside a component or across layer boundaries.
   - `operator_jitter`: Layer-to-layer duration variance on identical shapes (e.g., MoE expert load imbalance or dynamic quant outliers).
   - `theoretical_deviation`: Operators whose measured median duration diverges most from Roofline theoretical time (`theoretical_operator_time_us`).
   - `vector_sequence_candidates`: Contiguous `AI_VECTOR_CORE` chains (`Cast -> Mul -> Add -> RmsNorm` or `DynamicQuant` prologue/epilogue) where quant/dequant overhead exceeds MatMul savings.
   - `data_movement_ops`: Redundant `TransData`, `Transpose`, `Concat`, or `ViewCopy` inserted around quantized or multi-stream boundaries.
