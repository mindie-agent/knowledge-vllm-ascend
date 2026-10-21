---
category: sop
conditions:
  component: schedulers_qos_and_structured_output
  framework: vllm-ascend
domain_path: vllm-ascend/scheduling-and-ep/qos-and-lifecycle/slo-and-guided-decode
evidence_strength: software_contract
id: unit-vllm-ascend-batch-job-srf-ai-qos-and-structured-output-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/core/short_request_first_scheduler.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/core/batch_job_aware_scheduler.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/patch/platform/patch_structured_output.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/short_request_first.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/batch_job_aware_scheduler.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/Ai_QoS_introduction_en.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/structured_output.md
  source_digest: 75d5b6442064e09bf86eb4f438b6e5912d7624d105c5cdf8c61348dc2bcb7ab3
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/core/short_request_first_scheduler.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: ShortRequestFirst (SRF), Batch-Job-Aware Scheduler, UB Switch AI QoS, and Structured
  Output SOP
triggers:
  error_signatures:
  - ShortRequestFirst emits repeated degradation warnings when threshold is too small
    for prompt distribution
  - KV cache prefetch SDMA traffic contends with collective communication on Atlas
    A3 UB switch
  - batch_job_sched_config or short_request_first_config combined with incompatible
    Decode consumer role
  keywords:
  - short_request_first_config
  - ShortRequestFirst
  - batch_job_sched_config
  - BatchJobAwareScheduler
  - ai_qos
  - tools.ai_qos
  - structured_output
  - patch_structured_output
---
# ShortRequestFirst (SRF), Batch-Job-Aware Scheduler, UB Switch AI QoS, and Structured Output SOP

## Scope and preconditions

Use this SOP when configuring specialized waiting-queue schedulers (`short_request_first_config`, `batch_job_sched_config`), Atlas A3 UB-switch traffic isolation (`tools.ai_qos`), or guided decoding (`patch_structured_output.py`, `worker/v2/structured_outputs.py`) on `vllm-ascend`.

## Procedure

1. **Tune `ShortRequestFirst` (`short_request_first_scheduler.py`, `short_request_first.md`) for skewed online prefill:**
   - Splits the FCFS waiting queue into `immediate` (preempted/partially computed), `short` (`num_prompt_tokens <= threshold`, default `256`), and `long` (`> threshold`), dispatching in priority order: `immediate > aged-long > short > long`.
   - Anchor `threshold` at the valley between short and long prompt clusters (`P70–P85` of prompt length), and set `long_max_wait_ms` inside `[W_normal(P90..P95), W_slo]`. If 3 consecutive aged-long promotions trigger the degradation warning, increase `threshold` first before raising `long_max_wait_ms`. Do not enable on PD-disaggregated Decode (`kv_consumer`) nodes.
2. **Use `BatchJobAwareScheduler` (`batch_job_aware_scheduler.py`, `batch_job_aware_scheduler.md`) for offline batch throughput:**
   - Enabled via `{"scheduler_config": {"batch_job_sched_config": {"enabled": true}}}` for `vllm.entrypoints.openai.run_batch`.
   - Extracts `#job_name[<job>]#` tags from `request_id`, tracks per-job decode length via EWMA (cold-start `128` tokens), reserves KV blocks (`reserve_margin_blocks=2`, `reserve_max_blocks=8`), and prioritizes long-decode jobs when available tokens `> low_available_tokens_threshold` (`4096`) and short-decode jobs (`<= short_decode_token_threshold=32`) when KV capacity is tight.
3. **Isolate UB Switch Virtual Lanes with `tools.ai_qos` (`Ai_QoS_introduction_en.md`):**
   - On Atlas 800T A3 / Atlas 900 A3 SuperPoD (Ascend HDK `>= 25.5.2`, LingQu UB Switch `>= 1.5.1`) when overlapping KV cache H2D/D2H prefetching with collective communication, build `tools/ai_qos` against DSMI (`libdrvdsmi_host.so`) and run `python -m tools.ai_qos` (`AIV: high`, `PCIEDMA: high`, `SDMA: low`) so SDMA KV prefetch traffic is mapped to a lower-priority Virtual Lane and never stalls operator delivery.
4. **Apply guided decoding bitmasks (`patch_structured_output.py`, `worker/v2/structured_outputs.py`):**
   - Structured output (JSON schema, regex, grammar via XGrammar/Guidance) applies vocabulary bitmasks on NPU logits prior to sampling; ensure any custom sampler path preserves the bitmask application hook.

## Verification

- Verify `ShortRequestFirst` 5-second aggregate queue logs show zero consecutive aged-long degradation warnings under target traffic, and confirm structured output requests return 100% schema-valid JSON.
