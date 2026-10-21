---
category: workflow
conditions:
  component: hybrid_mtp_rollback
  framework: vllm-ascend
domain_path: vllm-ascend/speculative-decoding/proposers/hybrid-mtp-rollback
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-step3p5-and-hybrid-mtp-cache-rollback-workflow
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-bailing-linear-and-gdn-metadata-builder-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/spec_decode/step3p5.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/patch/worker/patch_step3p5.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/deepseek_mtp.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/deepseek_v4/mtp.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/gdn_attn_builder.py
  source_digest: 53f5f4f3c7e5430058f5bc20a2329b6880fc343dbf513f4b2b7ef52fc68819ac
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/spec_decode/step3p5.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Hybrid recurrent/sliding-window MTP speculative state rollback (Step3p5, Qwen3.5,
  Kimi K3, DSV4) workflow
triggers:
  error_signatures:
  - MTP speculative rejection corrupts recurrent GDN/KDA/conv state or compressor
    state on subsequent steps
  - Step3p5 or hybrid sliding-window MTP draft head reads stale rejected tokens after
    partial acceptance
  - chunked prefill followed by MTP decode mismatches seq_lens and state slot indices
  keywords:
  - step3p5
  - patch_step3p5
  - deepseek_mtp
  - qwen3_5_mtp
  - MambaStateCopyFunc
  - spec_state_indices
  - num_rejected_tokens
  - rollback
---
# Hybrid recurrent/sliding-window MTP speculative state rollback (Step3p5, Qwen3.5, Kimi K3, DSV4) workflow

## Trigger and scope

Use this workflow when enabling Multi-Token Prediction (`mtp`, `deepseek_mtp`, `qwen3_5_mtp`, `step3p5`) on models whose layers maintain non-trivial state beyond standard full-attention paged KV caches—specifically GatedDeltaNet/KDA recurrent + causal-conv states (Qwen3.5/3.8, Kimi K3), sliding-window / hybrid attention (`step3p5.py`, `patch_step3p5.py`), or DeepSeek-V4 Compress-4/Compress-128 compressor state caches (`models/deepseek_v4/mtp.py`).

## Stage 1 — Provision speculative state slots for target verification ($1 + K$ tokens)

1. **Size state pools for speculative verification (`gdn_attn_builder.py`, `patch_mamba_config.py`):**
   - Unlike paged KV caches—where rejected tokens at positions $> \text{accepted}$ are simply ignored by decrementing `seq_lens`—recurrent SSM/GDN/KDA states and causal-conv ring buffers mutate in place as each token is processed.
   - Allocate `1 + K` state slots per active request (`spec_state_indices_tensor`) so the target verification step records intermediate recurrent and conv states at each candidate token position $0, 1, \dots, K$.

## Stage 2 — Commit the last accepted state slot on rejection (`num_rejected_tokens`)

1. **Select and restore the accepted state on device:**
   - After `AscendRejectionSampler` computes `num_accepted_tokens` (or `num_rejected_tokens_gpu`), copy or re-index the state from candidate offset `num_accepted_tokens - 1` into the request's canonical base state slot before the next step begins.
   - In `step3p5.py` and `patch_step3p5.py`, ensure sliding-window and multi-step MTP draft heads advance `positions` and `slot_mapping` using `num_rejected_tokens_gpu` on device so rejected tail tokens are overwritten cleanly on the next draft pass.
2. **Align DSV4 compressor state cache with MTP (`vllm_ascend/models/deepseek_v4/mtp.py`, `compressor.py`):**
   - In DeepSeek-V4, `AscendCompressorStateCache` accumulates partial window states across `compress_ratio=4` (with overlap) and `compress_ratio=128`. Verify that rejected speculative tokens do not prematurely commit a compressed block boundary past `seq_len_accepted`.

## Stage 3 — Gate multi-step MTP (`num_speculative_tokens >= 1`) parity

1. **Run rejection-stress parity verification:**
   - Test with `num_speculative_tokens = 1` and `3` where draft tokens are intentionally partially rejected (varying prompt styles and long contexts). Confirm zero state drift between `MTP` (`temperature=0`) and non-MTP eager execution over 256 generated tokens.
