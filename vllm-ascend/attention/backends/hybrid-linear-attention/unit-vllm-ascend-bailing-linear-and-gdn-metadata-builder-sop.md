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
  source_digest: ba8b93c8191c6f1d97f5af3704c749468ed86f042837481003826983f07f8de7
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/gdn_attn_builder.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
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
  keywords:
  - gdn_attn_builder
  - AscendGDNMetadataBuilder
  - bailing_moe_linear_attn
  - chunk_gated_delta_rule
  - fused_recurrent_gated_delta_rule
  - kimi_kda
  - mamba_utils
---
# Hybrid linear attention (GatedDeltaNet / KDA / Bailing MoE linear attention) metadata builder and Triton FLA SOP

## Scope and preconditions

Use this SOP when working with hybrid recurrent/linear attention layers in `vllm-ascend`, including Qwen3.5/Qwen3.8 `GatedDeltaNet` (`vllm_ascend/ops/gdn.py`, `gdn_attn_builder.py`), Kimi K3 `KDA` (`kimi_kda.py`, `kimi_kda_state.py`), Bailing-MoE linear attention (`bailing_moe_linear_attn.py`), and Triton Flash-Linear-Attention kernels (`vllm_ascend/ops/triton/fla/`).

## Procedure

1. **Build separated spec vs non-spec metadata in `AscendGDNMetadataBuilder` (`vllm_ascend/ops/gdn_attn_builder.py`):**
   - `AscendGDNMetadataBuilder` partitions requests into speculative-verify tokens and non-speculative (prefill + standard 1-token decode) tokens, constructing `spec_query_start_loc`, `non_spec_query_start_loc`, `spec_state_indices_tensor`, and `non_spec_state_indices_tensor`.
   - On Ascend310P (`vllm_ascend/_310p/ops/gdn_attn_builder_310.py`), use the dedicated 310P builder and `gdn_310.py` custom ops (`cumpute_causal_conv1d_metadata_310.py`).
2. **Route prefill chunks to `chunk_gated_delta_rule` and decode steps to `fused_recurrent_gated_delta_rule`:**
   - Prefill sequences (`query_len > 1` without speculative tree) execute the chunked WY representation (`chunk_scaled_dot_kkt`, `solve_tril`, `wy_fast`, `chunk_delta_hupdate`, `chunk_o` in `vllm_ascend/ops/triton/fla/`).
   - Single-token decode and speculative token verification execute `fused_recurrent_gated_delta_rule` updating the per-request recurrent state in place.
3. **Preserve TP-sharded ALiBi-style slopes in `AscendBailingMoELinearAttention` (`vllm_ascend/ops/bailing_moe_linear_attn.py`):**
   - `AscendBailingMoELinearAttention` precomputes power-of-two head slopes sliced to the local TP rank `[tp_rank * local_heads : (tp_rank + 1) * local_heads]` and scales decay factors by `layer_decay_ratio = 1 - (layer_idx / (num_layers - 1)) + 1e-5`.
4. **Guard idle-DP dummy steps (`PAD_SLOT_ID = -1`):**
   - When a DP rank runs a dummy forward to participate in MoE EP collectives, its `state_indices_tensor` is `-1`. Ensure recurrent state kernels either skip `state_idx < 0` or write to a scratch zero-pad slot so dummy steps never clobber slot `0`.

## Verification

- Run a batch combining a 4096-token chunked prefill and a 1-token decode request, then verify that the decode request's recurrent state and greedy output are bit-identical to running it isolated at `BS=1`.
