---
category: sop
conditions:
  component: runtime_guard
  framework: vllm-ascend
domain_path: vllm-ascend/runbooks/precision-and-validation/runtime-guard
evidence_strength: software_contract
id: unit-vllm-ascend-runtime-guard-kv-delimitation-and-force-feed-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
- unit-vllm-ascend-runtime-guard-topology-and-config-axioms
provenance:
  evidence_refs:
  - djh95/runtime-guard-docs@45bc2603f48f883a448ed7c2bbeed38f9a2398e1:02_problem_localization/README.md
  - djh95/runtime-guard-docs@45bc2603f48f883a448ed7c2bbeed38f9a2398e1:06_agent_skills/investigation/SKILL.md
  - djh95/runtime-guard-docs@45bc2603f48f883a448ed7c2bbeed38f9a2398e1:06_agent_skills/ref-kv-dump/SKILL.md
  - djh95/runtime-guard-docs@45bc2603f48f883a448ed7c2bbeed38f9a2398e1:06_agent_skills/analysis/SKILL.md
  source_digest: a69b763c0f8500a1a3d184102acc5fd385aea61833becc3cc28d6ed65c366bd4
  source_locator: djh95/runtime-guard-docs@45bc2603f48f883a448ed7c2bbeed38f9a2398e1:06_agent_skills/analysis/SKILL.md
  source_type: cannbot_runbook_distillation
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
tier: L2
title: runtime_guard KV cache anomaly delimitation, same-mode force-feed reference,
  numeric-first comparison, and Q1-Q10 SOP
triggers:
  error_signatures:
  - runtime_guard triggers token_repeat or logits_finite and writes dump_kv .pt files
    for a degraded request
  - vllm-ascend emits non-repetitive gibberish or language drift missed by token_repeat
    (detector_gap) requiring manual_dump KV inspection
  - cosine similarity on dumped KV cache is misled by NaN/Inf or all-zero norm collapse
    in used_slots or NaN pollution in unused_slots
  - request KV dump has len(block_ids) * block_size < n_tokens causing out-of-bounds
    KV slot reads
  keywords:
  - runtime_guard
  - dump_kv
  - force_feed
  - locate_first_divergence
  - used_slots
  - unused_slots
  - zero_norm
  - detector_gap
  - manual_dump
  - verify_request_kv
  - Q1-Q10
---
# runtime_guard KV cache anomaly delimitation, same-mode force-feed reference, numeric-first comparison, and Q1-Q10 SOP

## Applicability and 3-layer rate accounting

Use this SOP to delimit whether an online serving accuracy anomaly (`token_repeat`, `logits_finite`, `spec_acceptance`, or client-observed gibberish) originates inside the KV cache (`Q1-Q7`) or downstream of KV writes (`Q8-Q10`).

1. **Keep the three detector rates strictly separate (`detector-sweep`):**
   - `trigger_rate = N_triggered / N_total` (fraction of requests firing a detector).
   - `precision = N_true_bad_in_triggered / N_triggered` (low precision indicates over-sensitive thresholds such as `ngram_min` / `min_repeat_count` firing on benign markdown/math tables).
   - `detect_rate = N_triggered_in_bad / N_all_bad` (when client outputs contain non-repetitive gibberish or mid-sentence language switching while `logits_finite` is clean and `token_repeat` does not fire, classify as `detector_gap` / `乱码漏检` and capture KV via hot-reloaded `dump_kv.auto_max_times: 0` + incremented `dump_kv.manual_dump: N`).

## Procedure

1. **Run capacity and parity pre-gates before tensor comparison (`verify_request_kv`):**
   - Inspect `meta.json` (or run `mindie-debugger`'s `ACLGraphAddressTracker.verify_replay_inputs`) to verify `len(valid_block_ids) * block_size >= n_tokens` (`n_prompt + n_output`) and `0 <= block_id < num_blocks` for every layer group. If capacity is insufficient (`block_table_capacity_insufficient`), stop immediately—the scheduler/block manager under-allocated physical KV blocks.
   - When comparing against a second environment, verify checkpoint `.safetensors` MD5 parity, `vllm`/`vllm-ascend` commit parity, and live `/proc/<PID>/environ` parity first.

2. **Generate a same-mode, same-topology `force-feed` clean reference KV dump (`ref-kv-dump`):**
   - Read `prompt_token_ids` (`n_prompt`) and `output_token_ids` (`n_output`) from the anomalous dump's `meta.json`.
   - Construct `force_prompt_ids = prompt_token_ids + output_token_ids[:-1]` with exact length **`n_prompt + n_output - 1`** (excluding the final anomalous token; never feed all `n_prompt + n_output` tokens or the reference tensor length will exceed `used_slots` by 1).
   - Serve `force_prompt_ids` with `max_tokens=1, temperature=0` on the **same execution mode (`ACLGraph` vs Eager) and same TP/PP topology** (or a verified clean baseline), and trigger `manual_dump` (`auto_max_times: 0`, increment `manual_dump`) so prefill materializes the clean reference KV for all `n_prompt + n_output - 1` tokens in one pass.

3. **Stitch TP KV heads (`dim=-2`) and execute the 3-step numeric-first comparison (`analysis`):**
   - When comparing dumps across multi-TP ranks (`tp0..tp{K-1}`), concatenate per-rank `used_slots` along `dim=-2` (`n_kv_heads`) in ascending `tp_rank` order before diffing.
   - Evaluate every layer in strict priority order (matching `mindie_debugger.precision.metrics._tensor_metrics(..., valid_length=n_tokens)`):
     1. **Step 1 — `unused_slots` 3-level hygiene (`unused_slot_status`)**:
        - `clean_zero`: unallocated tail slots in the last KV block are all `0.0` (healthy).
        - `finite_dirty`: finite non-zero residual from previously freed requests (expected when block zeroing on free is disabled; harmless unless `block_table` / `seq_lens` over-reads past `valid_length`).
        - `nonfinite_pollution`: `NaN` or `Inf` in `unused_slots` (high-risk: any unmasked attention read produces `0 * NaN = NaN`).
     2. **Step 2 — `used_slots` numeric-first gate (`numeric_issue` before `cosine_sim`)**:
        - Check `target_nan`, `target_inf`, `baseline_nonfinite`, and `target_zero_norm` (`norm_targ == 0` while `norm_base > 0`, caused by dropped KV writes or `slot_mapping = -1`) **before** computing cosine similarity. Never sort layers by raw `cosine_sim` when `numeric_issue` is present (`cosine_sim` must be `None`).
     3. **Step 3 — Dual-view localization (Table 1 token evolution + Table 2 layer chain)**:
        - **Table 1 (`token_idx 0..N-1 × layer`)**: Locate the first token `T0` where `mean(k_cos, v_cos)` drops or `numeric_issue` appears. Annotate columns `t >= max(n_prompt, T0)` with `out[k]=<token_id>`: once decode outputs diverge at `T0`, subsequent token columns feed different input tokens into the model, so post-`T0` KV divergence is expected causal feedback rather than new independent faults.
        - **Table 2 (`layer 0..L-1` at fixed token `T0`)**: Inspect `k_cos`, `v_cos`, `k_max_diff`, `v_max_diff`, and `numeric_issue` across all layers at `T0` to identify the exact first failing layer `L0`.

4. **Walk the `Q1-Q10` decision tree and emit the 4-tier closure report:**
   - **Step A (`Q1-Q2` Region)**: Does `T0` fall in prefill (`0 .. n_prompt-1`, `Q1`) or decode (`n_prompt .. N-1`, `Q2`), or are all `used_slots` across all layers/tokens clean (`cos > 0.999` and `numeric_issue is None` -> jump to Step D)?
   - **Step B (`Q3-Q5` Temporal Shape)**: Is the drop at `T0` a step-function cliff (`Q3`: check if `T0` is a multiple of `block_size` -> paged block allocation/offload/prefix-cache boundary, or `chunk_size` -> chunked prefill boundary) or a gradual multi-step decay (`Q4-Q5`: quantization rounding accumulation or RoPE position drift)?
   - **Step C (`Q6-Q7` Layer Topology at `T0`)**:
     - Drops starting at a specific layer `L0` and propagating to `L0+1..L-1` (`Q6`): operator, fusion pass, or quantization scale fault inside layer `L0` -> hand off to `mindie-debugger` (`aclgraph inspect` if graph-only, or `--enforce-eager` `msprobe` scoped to `layer.L0`).
     - Drops simultaneously at `layer 0` (`Q7`): input embedding, `positions`, `slot_mapping`, or `block_table` corruption before layer `0`.
   - **Step D (`Q8-Q10` Clean KV, Bad Output)**: When all `used_slots` match (`cos > 0.999`) up to the anomalous step, the KV cache is innocent; isolate downstream post-attention stages: last-layer attention read over `nonfinite_pollution` in `unused_slots` (`Q8`), `final_layernorm` / `lm_head` / TP `AllGather` (`Q9`), or sampler / detokenizer / speculative rejection (`Q10`).
   - **Four-Tier Closure Output**: Report **Tier 1** (`KV Corrupt` vs `KV Clean`), **Tier 2** (`First Divergent Token T0 & Layer L0`), **Tier 3** (`Root-Cause Subsystem`), and **Tier 4** (`Deterministic Next Verification Command`).
