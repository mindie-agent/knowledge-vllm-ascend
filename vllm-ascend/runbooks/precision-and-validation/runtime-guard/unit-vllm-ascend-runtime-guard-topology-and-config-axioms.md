---
category: axiom
conditions:
  component: runtime_guard
  framework: vllm-ascend
domain_path: vllm-ascend/runbooks/precision-and-validation/runtime-guard
evidence_strength: software_contract
id: unit-vllm-ascend-runtime-guard-topology-and-config-axioms
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - djh95/runtime-guard-docs@45bc2603f48f883a448ed7c2bbeed38f9a2398e1:01_enable_and_start/README.md
  - djh95/runtime-guard-docs@45bc2603f48f883a448ed7c2bbeed38f9a2398e1:03_detectors/README.md
  - djh95/runtime-guard-docs@45bc2603f48f883a448ed7c2bbeed38f9a2398e1:06_agent_skills/config-recommender/SKILL.md
  - djh95/runtime-guard-docs@45bc2603f48f883a448ed7c2bbeed38f9a2398e1:06_agent_skills/SKILLS_METADATA.md
  source_digest: 98b7c2981f0fd5d5a4822be8bb3af6385b632f8f3ca22395f9745e44e78bda8f
  source_locator: djh95/runtime-guard-docs@45bc2603f48f883a448ed7c2bbeed38f9a2398e1:06_agent_skills/config-recommender/SKILL.md
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
tier: L1
title: vllm-ascend runtime_guard MRV2 hook boundary, shipped detectors, last-PP TP
  topology, and dump_kv watermark axioms
triggers:
  error_signatures:
  - runtime_guard enabled under VLLM_ASCEND_USE_V2_MODEL_RUNNER=0 produces no log
    or report without MRV1 backport patch
  - runtime_guard runtime_config_hot_reload toggled from false to true at runtime
    never reloads runtime_config_path
  - runtime_guard manual_dump set to 1 repeatedly or alongside auto_max_times > 0
    does not trigger a new KV dump
  - runtime_guard dump_kv across TP ranks yields per-rank KV head slices requiring
    dim=-2 concatenation
  keywords:
  - runtime_guard
  - token_repeat
  - logits_finite
  - spec_acceptance
  - dump_kv
  - manual_dump
  - auto_max_times
  - runtime_config_hot_reload
  - due_bus
  - used_slots
  - unused_slots
---
# vllm-ascend runtime_guard MRV2 hook boundary, shipped detectors, last-PP TP topology, and dump_kv watermark axioms

**Condition.** Applies when configuring or analyzing `additional_config.runtime_guard` in `vllm-ascend` (`v5.20.12` contract) across single-rank or multi-rank (`TP`, `PP`, `DP`, `EP`) deployments.

**Constraint.**

1. **ModelRunner hook boundary (`MRV2` native vs `MRV1`/`v0.3.0` backports).** Native `runtime_guard` hooks `vllm_ascend/worker/v2/model_runner.py` (`ModelRunner V2`) at two points per step: post-forward (`logits` before sampling) and post-sampler (`sampled_token_ids`). Running with `VLLM_ASCEND_USE_V2_MODEL_RUNNER=0` (`ModelRunner V1` in `model_runner_v1.py`) or on `vllm-ascend v0.11.0` (`v0.3.0` branch) silently bypasses `runtime_guard` unless the corresponding `05_mrv1_backport` or `05_v030_backport` patch series is applied.

2. **Shipped detector whitelist (`token_repeat`, `logits_finite`, `spec_acceptance` only).** Only three detectors are implemented in `RuntimeGuardConfig.from_dict()`:
   - `token_repeat`: CPU n-gram repetition check on post-sampler token IDs (`ngram_min`, `ngram_max`, `min_repeat_count`, `check_interval_steps`).
   - `logits_finite`: in-place NPU `isnan`/`isinf` check on post-forward `logits` (`check_interval_steps`, `check_nan`, `check_inf`).
   - `spec_acceptance`: rolling draft-token acceptance rate check under speculative decoding (`window_size`, `min_accept_rate`, `check_interval_steps`).
   Unimplemented keys (`output_substring`, `token_logprob`, `slot_*`, `kv_state`) are ignored by `from_dict()` and must never be emitted in configuration files.

3. **Last-PP `TP0` detection authority and `due bus` all-TP `dump_kv` contract.**
   - Under `TP` and `PP`, anomaly detection and `report/rank*_pid*.jsonl` emission run **exclusively on last-PP `TP0`** (`is_last_pp_rank() and tp_rank == 0`).
   - When `dump_kv.enable=true`, last-PP `TP0` broadcasts due request IDs over the `due bus` (`<= 32` requests per step) so **every TP rank in the last PP stage** writes `tp{tp_rank}_rank{global_rank}_step{step}.pt` containing its local KV head shard (`used_slots` shape `[n_tokens, n_kv_heads_per_tp, head_dim]`, `unused_slots` shape `[n_unused, n_kv_heads_per_tp, head_dim]`). Non-last PP stages (`pp_rank < pp_size - 1`) never run detectors or dump KV.

4. **Startup-only hot-reload and `manual_dump` watermark semantics.**
   - `runtime_config_hot_reload: true` and `runtime_config_path` must be set at process startup; the `file_mtime` poller is created only during initialization, so changing `runtime_config_hot_reload` from `false` to `true` at runtime has no effect.
   - `report.max_per_req` controls per-request JSONL emission (`0` = final summary on request completion only; `1` = first anomaly trigger + final summary; `N > 1` = up to `N` triggers + final summary).
   - `dump_kv.manual_dump` is an integer **cumulative watermark** (`manual_dump > _manual_dumped_count`), not a boolean toggle, and is evaluated **only when `auto_max_times == 0`**. If `auto_max_times > 0`, `manual_dump` is ignored; to trigger successive manual dumps via hot-reload, set `auto_max_times: 0` and increment `manual_dump` (`0 -> 1 -> 2 -> ...`).

5. **Attention backend coverage.** `dump_kv` extracts paged KV blocks for standard `PagedAttention` (`[2, num_blocks, block_size, n_kv_heads, head_dim]`), split `(k_cache, v_cache)` tuples, and DeepSeek `MLA` (`kv_lora_rank=512` + `qk_rope_head_dim=64` `[num_blocks, block_size, 1, dim]`), while skipping linear/recurrent state layers (`GDN` / `Mamba` conv+SSM states).

**Rationale.** Running detection on last-PP `TP0` avoids redundant cross-rank CPU checks while the `due bus` guarantees all TP shards of the last PP stage dump synchronously at the exact anomaly step without breaking `ACLGraph` replay. Watermark semantics prevent hot-reload polling from re-dumping the same manual request on every `file_mtime` check.
