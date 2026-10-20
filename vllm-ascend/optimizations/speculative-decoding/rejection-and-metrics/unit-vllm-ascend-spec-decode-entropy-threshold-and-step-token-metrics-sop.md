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
domain_path: vllm-ascend/optimizations/speculative-decoding/rejection-and-metrics
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-spec-decode-entropy-threshold-and-step-token-metrics-sop
inherits_from:
- unit-vllm-ascend-mtp-draft-kv-allocation-and-recurrent-state-sop
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/spec_decode_entropy_verify_relax_rejection_threshold.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/cases/mtp_benefit_sparse_attention_and_latency_metric.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/mtp_decode_timing_two_metric_split.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/validations/mtp_decode_timing_collection_granularity.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/validations/multicard_mtp_timing_barrier.md
  source_digest: 9d05d0f69491b24959744e7115d85f69d5b64a539b76f56cf1ecb0365380b37d
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/spec_decode_entropy_verify_relax_rejection_threshold.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Entropy-relaxed speculative rejection sampling and dual step/token latency
  accounting in vLLM-Ascend
triggers:
  error_signatures:
  - Speculative decoding acceptance rate drops sharply on high-entropy tokens, causing
    net TPOT regression
  - Reported MTP decode latency omits draft forward steps or conflates per-step latency
    with per-token TPOT
  - Multi-card MTP step timings exhibit high rank-to-rank jitter due to unsynchronized
    H2D post-processing
  keywords:
  - spec_decode_entropy_verify
  - posterior_threshold
  - posterior_alpha
  - mtp_decode_timing_two_metric_split
  - accepted_num
  - effective_tpot
---
# Entropy-relaxed speculative rejection sampling and dual step/token latency accounting in vLLM-Ascend

## Applicability and symptom scope

Use this SOP when tuning MTP / Eagle speculative decoding acceptance rates or auditing MTP latency benchmarks on Ascend NPU.

## Procedure

1. **Relax rejection sampling via target-distribution entropy gating when strict greedy/posterior thresholds under-accept (`spec_decode_entropy_verify_relax_rejection_threshold.md`).**
   - When the target model's next-token probability distribution has high Shannon entropy $H(p) = -\sum_i p_i \log p_i$, multiple candidate tokens are semantically plausible, causing strict token-identity or fixed `posterior_threshold` checks to reject valid draft continuations.
   - Compute the dynamic entropy-relaxed acceptance floor:
     $$\text{threshold} = \min\left(\exp(-H(p) \cdot \alpha),\; \text{posterior\_threshold}\right)$$
     where $\alpha$ (`posterior_alpha`) scales entropy sensitivity, accepting draft tokens whose target probability $p(\text{draft}) \ge \text{threshold}$.
   - **Verify:** Measure both acceptance length ($1 + \mathbb{E}[\text{accepted\_num}]$) and downstream benchmark accuracy (e.g., GSM8K / HumanEval) to confirm zero accuracy degradation.

2. **Accumulate full target + all `next_n` draft forwards into every decode step record and report dual metrics (`mtp_decode_timing_two_metric_split.md`, `mtp_decode_timing_collection_granularity.md`).**
   - One speculative decode step comprises `1` target verification forward (`q_len = 1 + next_n`) plus `next_n` sequential draft forwards, emitting `1 + accepted_num` tokens.
   - Never record target forward and draft forwards as separate entries in the step-latency array (which corrupts IQR outlier filtering and understates step cost).
   - Always report two distinct metrics:
     - `step_latency_ms`: full wall-clock duration of `target + sum(draft_1 .. draft_{next_n}) + verify_postprocess`.
     - `effective_tpot_ms`: `total_decode_time_ms / total_generated_tokens` ($= \text{step\_latency\_ms} / (1 + \overline{\text{accepted\_num}})$).

3. **Evaluate MTP net benefit alongside sparse attention (`mtp_benefit_sparse_attention_and_latency_metric.md`) and align multi-card timers (`multicard_mtp_timing_barrier.md`).**
   - Because target verification processes `1 + next_n` query tokens per request simultaneously, memory-bound weight loading is amortized across `1 + accepted_num` tokens, whereas attention KV bandwidth scales with context length unless paired with DSA/SFA sparse attention.
   - Insert `dist.barrier()` + `torch.npu.synchronize()` before starting the decode timing window in multi-card MTP benchmarks to eliminate rank skew from asynchronous H2D draft token preparation.
