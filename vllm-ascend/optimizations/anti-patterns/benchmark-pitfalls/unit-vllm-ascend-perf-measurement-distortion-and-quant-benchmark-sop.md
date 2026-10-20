---
category: sop
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
domain_path: vllm-ascend/optimizations/anti-patterns/benchmark-pitfalls
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-perf-measurement-distortion-and-quant-benchmark-sop
inherits_from:
- unit-vllm-ascend-c8-pypto-310p-and-is-cuda-anti-patterns-axiom
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/anti_patterns/perf_measurement_distortion_input_and_metric.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/benchmark_observations/benchmark_deepseek_v32_quant_accuracy.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/benchmark_observations/dsa_indexcache_topk_reuse_throughput.md
  source_digest: 20ac7733e2b849df37efc411e69f64313279e02c92a44543edbae31332f400c8
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/anti_patterns/perf_measurement_distortion_input_and_metric.md
  source_type: cannbot_benchmark_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Eliminating performance measurement distortion and evaluating W8A8/W4A8C8 and
  IndexCache accuracy-throughput tradeoffs
triggers:
  error_signatures:
  - Synthetic microbenchmark shows speedup from repeated L2 cache hits on identical
    block_table or zeroed topk_indices
  - Integer floor division before unit scaling truncates sub-millisecond operator
    latency to zero
  - Long-sequence DSA decode throughput bottlenecked by per-step Lightning Indexer
    top-k selection
  keywords:
  - perf_measurement_distortion
  - benchmark_deepseek_v32_quant_accuracy
  - dsa_indexcache_topk_reuse
  - IndexCache
  - W8A8C16
  - W8A8C8
  - W4A8C8
---
# Eliminating performance measurement distortion and evaluating W8A8/W4A8C8 and IndexCache accuracy-throughput tradeoffs

## Applicability and symptom scope

Use this SOP when benchmarking custom operators or end-to-end serving changes in `vllm-ascend`, or when selecting quantization tiers (`W8A8C16`, `W8A8C8`, `W4A8C8`) and sparse-attention `IndexCache` reuse intervals on DSA/MLA models.

## Procedure

1. **Audit microbenchmark inputs against artificial L2 cache residency and degenerate control flow (`perf_measurement_distortion_input_and_metric.md`).**
   - Never benchmark paged attention, `GatherSelectionKvCache`, or MoE GMM using all-zero `block_tables`, identical `topk_indices` across batches, or a single reused KV page. Identical indices keep the working set inside the NPU L2 cache and overstate memory-bound kernel bandwidth by `2x-5x`.
   - Populate `block_tables` and `topk_indices` with realistic non-overlapping physical block permutations spanning the full HBM KV pool footprint, and match the real token routing histogram across experts.
   - **Invariant:** Benchmark working-set bytes exceed NPU L2 capacity and mirror production page-scatter strides.

2. **Eliminate pre-scaling integer truncation and rank-skewed timers (`perf_measurement_distortion_input_and_metric.md`).**
   - Do not apply integer division (`// 1000`) to microsecond or nanosecond counters before converting to floating-point milliseconds, and always execute `torch.npu.synchronize()` (plus `dist.barrier()` on multi-card runs) before both `start_time` and `end_time`.
   - **Verify:** Confirm that sum of per-operator profiled durations matches wall-clock step time within `5%`.

3. **Select DSA/MLA quantization tier using empirical accuracy baselines (`benchmark_deepseek_v32_quant_accuracy.md`).**
   - Across `DeepSeek-V3.2-Exp` evaluations on GSM8K, MMLU, CEVAL, and LongBench:
     - `W8A8C16` (dynamic INT8 weights/activations, BF16 KV cache) preserves `< 0.3%` accuracy delta from BF16 baseline.
     - `W8A8C8` (adding INT8 C8 KV cache with pre-calibrated scales) retains near-BF16 accuracy while cutting KV HBM footprint by `50%`.
     - `W4A8C8` (INT4 routed MoE experts, INT8 shared experts/attention, INT8 KV cache) requires outlier clipping (`clip/clamp`) and per-group scales on routed experts to stay within `0.5%-1.0%` of BF16.

4. **Evaluate cross-step `IndexCache` top-k index reuse for long-context DSA decode (`dsa_indexcache_topk_reuse_throughput.md`).**
   - In long-sequence DSA/SFA decode (e.g., GLM-5 / DeepSeek-V3.2 W8A8), executing Lightning Indexer every decode step to select top-k KV blocks dominates step latency at long context lengths.
   - Reusing selected `topk_indices` across `K` consecutive decode steps (`IndexCache`) amortizes Lightning Indexer compute overhead and increases decode throughput, provided accuracy is validated on long-context needle/retrieval tasks before production adoption.
