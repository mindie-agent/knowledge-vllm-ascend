---
category: sop
conditions:
  component: hybrid_linear_attention
  framework: vllm-ascend
domain_path: vllm-ascend/attention/backends/hybrid-linear-attention
evidence_strength: software_contract
id: unit-vllm-ascend-bailing-linear-and-gdn-metadata-builder-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-hccl-and-dp-collective-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/gdn_attn_builder.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/gdn.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/kimi_kda.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/bailing_moe_linear_attn.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/triton/fla/chunk_gated_delta_rule.py
  - https://github.com/vllm-project/vllm-ascend/pull/12027
  - https://github.com/vllm-project/vllm-ascend/pull/12255
  - https://github.com/vllm-project/vllm/pull/35219
  source_digest: ba8b93c8191c6f1d97f5af3704c749468ed86f042837481003826983f07f8de7
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/gdn_attn_builder.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
tier: L2
title: Hybrid linear attention (GatedDeltaNet / KDA / Bailing MoE linear attention)
  metadata builder and Triton FLA SOP
triggers:
  error_signatures:
  - GatedDeltaNet or KDA layer corrupts recurrent state when batch mixes prefill and
    decode requests
  - AscendGDNMetadataBuilder non_spec_state_indices_tensor points to padding slot
    -1 without mask guard
  - Bailing linear attention slope tensor or chunk_gated_delta_rule Triton launch
    fails on Ascend
  - Qwen3.5 PCP or PD accuracy degrades in FULL_DECODE_ONLY graph mode after moving GDN conv1d metadata to device tensors
  - MTP step >= 1 full-graph decode fails with KvQuantSparseAttnSharedkv AICORE errcode 264 when query_start_loc[-1] exceeds num_actual_tokens
  - Qwen3_5MoeForCausalLM fails at assert mamba_block_size is not None when missing from MODELS_CONFIG_MAP
  keywords:
  - gdn_attn_builder
  - AscendGDNMetadataBuilder
  - bailing_moe_linear_attn
  - chunk_gated_delta_rule
  - fused_recurrent_gated_delta_rule
  - kimi_kda
  - mamba_utils
  - num_decode_tokens
  - KVBlockZeroer
  - MODELS_CONFIG_MAP
---
# Hybrid linear attention (GatedDeltaNet / KDA / Bailing MoE linear attention) metadata builder and Triton FLA SOP

## Scope and preconditions

Use this SOP when working with hybrid recurrent/linear attention layers in `vllm-ascend`, including Qwen3.5/Qwen3.8 `GatedDeltaNet` (`vllm_ascend/ops/gdn.py`, `gdn_attn_builder.py`), Kimi K3 `KDA` (`kimi_kda.py`, `kimi_kda_state.py`), Bailing-MoE linear attention (`bailing_moe_linear_attn.py`), and Triton Flash-Linear-Attention kernels (`vllm_ascend/ops/triton/fla/`).

## Procedure

1. **Build separated spec vs non-spec metadata and separate the four counting domains in `AscendGDNMetadataBuilder` (`vllm_ascend/ops/gdn_attn_builder.py`, `PR #12027`, `PR #12255`):**
   - `AscendGDNMetadataBuilder` partitions requests into speculative-verify tokens and non-speculative (prefill + standard 1-token decode) tokens, constructing `spec_query_start_loc`, `non_spec_query_start_loc`, `spec_state_indices_tensor`, and `non_spec_state_indices_tensor`. On Ascend310P (`vllm_ascend/_310p/ops/gdn_attn_builder_310.py`), use the dedicated 310P builder.
   - **Four Counting Domains under `FULL_DECODE_ONLY` & MTP:** Strictly distinguish `num_reqs` (padded graph request slots), `num_actual_tokens` (true tokens), `num_decodes` (which in vLLM `split_decodes_and_prefills()` fast-path `max_query_len <= 1` equals padded `num_reqs`), and `num_decode_tokens` (true decode tokens). When MTP is enabled globally but the current batch falls back to non-spec decode (all draft masks `-1`), the only trustworthy real request count is `actual_num_decodes = num_decode_tokens` (`attn_metadata.num_decode_tokens` on 310P).
   - **Persistent Graph Buffer Hygiene & MTP `step >= 1` Token Alignment:** Because `model_runner_v1.py` only zeroes padded block-table rows when `pcp_size == 1`, copying `num_decodes` rows into persistent graph buffers under PCP copies stale valid state slots into padding rows (and slicing the Python view to `num_actual_tokens` does not alter the captured ACLGraph buffer address). Copy only `actual_num_decodes` rows and explicitly clear padded tail entries to null/`PAD_SLOT_ID`. In MTP `step >= 1` full-graph steps, always enforce `query_start_loc[-1] == q.shape[0] == slot_mapping.shape[0]` (`num_actual_tokens`) so padded request slots never inflate `cu_seqlens_q[-1]` and trigger AICore `errcode 264`.
2. **Route prefill chunks to `chunk_gated_delta_rule` and decode steps to `fused_recurrent_gated_delta_rule`:**
   - Prefill sequences (`query_len > 1` without speculative tree, and 1-token initial prefill without prior state in PD disaggregation) must preserve prefill semantics (`chunk_gated_delta_rule` in `vllm_ascend/ops/triton/fla/`).
   - Single-token continuation decode and speculative token verification execute `fused_recurrent_gated_delta_rule` updating the per-request recurrent state in place.
3. **Preserve TP-sharded ALiBi-style slopes in `AscendBailingMoELinearAttention` (`vllm_ascend/ops/bailing_moe_linear_attn.py`):**
   - `AscendBailingMoELinearAttention` precomputes power-of-two head slopes sliced to the local TP rank `[tp_rank * local_heads : (tp_rank + 1) * local_heads]` and scales decay factors by `layer_decay_ratio = 1 - (layer_idx / (num_layers - 1)) + 1e-5`.
4. **Guard idle-DP dummy steps (`PAD_SLOT_ID = -1`), `KVBlockZeroer` semantics, and async `global_stream` state updates:**
   - When a DP rank runs a dummy forward to participate in MoE EP collectives, its `state_indices_tensor` is `-1`; ensure recurrent state kernels skip `state_idx < 0` or write to a scratch zero-pad slot so dummy steps never clobber slot `0`.
   - **Upstream `KVBlockZeroer` (`vllm#35219`) vs. Mamba State:** `needs_kv_cache_zeroing=True` in hybrid models zeroes **only newly allocated `FullAttentionSpec` blocks** (so FP32 Mamba bit patterns are not reinterpreted as `NaN/Inf` by FP16/FP8 attention); it never zeroes Mamba/SSM blocks. Moreover, in vLLM 0.27+, `async_scheduling` defaults to `None` and auto-enables to `True` unless `--no-async-scheduling` is passed; any state update offloaded to an auxiliary `global_stream` in `_update_states_after_model_execute()` must record reverse stream event dependencies before the next step reuses physical blocks or block tables, and overlapping state copies must obey `memmove` semantics.
5. **Enforce `MODELS_CONFIG_MAP` dual registration and PD speculative parity:**
   - Every hybrid Mamba/Attention model class (e.g., `Qwen3_5MoeForCausalLM`) must be registered in **both** the model registry and `MODELS_CONFIG_MAP` so hybrid `mamba_block_size` initialization runs. In PD disaggregation, P and D instances of hybrid Mamba models must configure identical `num_speculative_tokens` so cache groups match.

## Verification

- Run a batch combining a 4096-token chunked prefill and a 1-token decode request, plus `FULL_DECODE_ONLY` graph replay under `PCP > 1` and `MTP` with non-spec fallback (`draft_mask == -1`), and verify that recurrent states and greedy outputs match `BS=1` eager execution.
