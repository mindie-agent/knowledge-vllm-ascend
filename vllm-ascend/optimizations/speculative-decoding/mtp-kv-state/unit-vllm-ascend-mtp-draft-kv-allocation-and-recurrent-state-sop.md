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
domain_path: vllm-ascend/optimizations/speculative-decoding/mtp-kv-state
evidence_strength: software_contract
id: unit-vllm-ascend-mtp-draft-kv-allocation-and-recurrent-state-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/mtp_cache_and_microbatch_prefill_bookkeeping.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/mtp_draft_cache_state_correctness.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/runtime/mtp_speculative_decode_graph_mode_prereqs.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/mtp_speculative_decoding.md
  source_digest: 13fede29d523e5910bfb499ed1cebba6b757f6dc44b9125a304bd5ec03d6e80f
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/mtp_cache_and_microbatch_prefill_bookkeeping.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: MTP speculative decoding KV-cache layer accounting, recurrent state rollback,
  and graph-mode isolation in vLLM-Ascend
triggers:
  error_signatures:
  - Single-layer MTP draft model overwrites KV cache across next_n speculative steps
    when only 1 KV layer is allocated
  - Hybrid SSM/GDN model with MTP diverges after token rejection due to unrewound
    recurrent state
  - Enabling TorchAir or ACLGraph with MTP causes repeated graph recompilation between
    target and draft forwards
  keywords:
  - mtp_cache_and_microbatch_prefill
  - mtp_draft_cache_state_correctness
  - mtp_speculative_decode_graph_mode
  - num_nextn_predict_layers
  - next_n
  - speculative_config
---
# MTP speculative decoding KV-cache layer accounting, recurrent state rollback, and graph-mode isolation in vLLM-Ascend

## Applicability and symptom scope

Use this SOP when enabling Multi-Token Prediction (MTP) or Eagle speculative decoding in `vllm-ascend`, particularly with multi-step draft chains (`next_n > 1`), chunked/micro-batch prefill, hybrid recurrent-attention architectures, or NPU graph mode.

## Procedure

1. **Allocate MTP KV cache layers by `next_n` prediction depth when a single physical draft layer is reused (`mtp_cache_and_microbatch_prefill_bookkeeping.md`, `mtp_draft_cache_state_correctness.md`).**
   - Distinguish between *physical MTP weight layers* (`num_mtp_layers`) and *speculative draft steps* (`next_n`):
     - When `num_mtp_layers == 1` is autoregressively invoked `next_n` times per decode step, allocate `next_n` distinct KV cache layer slots (or address distinct `slot_mapping` offsets per draft step) so draft step `k` does not overwrite the KV cache state needed by draft step `k-1`.
     - When `num_mtp_layers == next_n` (distinct MTP heads), allocate `1` KV cache layer per physical MTP head.
   - **Invariant:** Every speculative draft step `0 .. next_n - 1` writes to a non-overlapping KV slot range.

2. **Advance or rewind recurrent/SSM state strictly by `accepted_num` tokens (`mtp_draft_cache_state_correctness.md`).**
   - For hybrid models combining full attention with Mamba2 / Gated DeltaNet (GDN), paged KV cache entries for rejected draft tokens are naturally ignored by advancing `seq_lens` by `1 + accepted_num`.
   - However, in-place recurrent states (`conv_state`, `ssm_state`) are mutated by every verified draft token. Maintain `1 + next_n` state snapshots (or dual draft/committed state slots) and commit the snapshot at index `accepted_num` after rejection sampling.

3. **Defer MTP draft prefill until the final chunk when chunked or micro-batch prefill is active (`mtp_cache_and_microbatch_prefill_bookkeeping.md`).**
   - During chunked prefill, intermediate chunks do not emit a verified final token. Execute the MTP draft prefill pass only after the final prompt chunk completes target-model prefill and produces the first sampled token, feeding the shifted token sequence `[t_1 .. t_L, sampled_t_{L+1}]`.

4. **Isolate target and MTP draft graph-compile entrypoints (`mtp_speculative_decode_graph_mode_prereqs.md`).**
   - Target verify (`q_len = 1 + next_n` per request) and single-token MTP draft forward (`q_len = 1` per request) have distinct query lengths and operator graphs.
   - Compile target and draft runners as separate `torchair` / `ACLGraph` callables with independent static buffer pools, and verify that FIA meets GQA head-divisibility requirements for `q_len = 1 + next_n`.
