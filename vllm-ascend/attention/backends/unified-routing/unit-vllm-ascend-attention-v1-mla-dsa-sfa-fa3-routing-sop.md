---
category: sop
conditions:
  component: attention
  framework: vllm-ascend
domain_path: vllm-ascend/attention/backends/unified-routing
evidence_strength: software_contract
id: unit-vllm-ascend-attention-v1-mla-dsa-sfa-fa3-routing-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/attention/attention_v1.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/attention/mla_v1.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/attention/dsa_v1.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/attention/sfa_v1.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/attention/fa3_v1.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/platform.py
  source_digest: d9c6536f6903589fe83a6e4c8acb3934f88de60abd7768190a4a9edcf67e50ab
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/attention/attention_v1.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Unified attention backend routing (Standard V1, MLA, DSA V1, SFA V1, FA3) and
  AscendAttentionState SOP
triggers:
  error_signatures:
  - get_attn_backend_cls selects incompatible attention backend for MLA, DSA, or SFA
    layers
  - AscendAttentionState chooses ChunkedPrefill instead of DecodeOnly and disables
    FULL_DECODE_ONLY graph replay
  - FA3 quantization scale missing on Qwen3 or DeepSeek attention layer
  keywords:
  - AscendAttentionBackend
  - AscendMLABackend
  - AscendDSABackend
  - AscendSFABackend
  - AscendFA3Backend
  - AscendAttentionState
  - get_attn_backend_cls
  - MLAPO
---
# Unified attention backend routing (Standard V1, MLA, DSA V1, SFA V1, FA3) and AscendAttentionState SOP

## Scope and preconditions

Use this SOP when diagnosing attention backend selection in `NPUPlatform.get_attn_backend_cls()` (`vllm_ascend/platform.py`) or execution-phase transitions across `AscendAttentionBackend` (`attention_v1.py`), `AscendMLABackend` (`mla_v1.py`), `AscendDSABackend` (`dsa_v1.py`), `AscendSFABackend` (`sfa_v1.py`), and `AscendFA3Backend` (`fa3_v1.py`).

## Procedure

1. **Trace backend dispatch in `NPUPlatform.get_attn_backend_cls()` (`vllm_ascend/platform.py`):**
   - **Ascend310P:** Routes exclusively to `vllm_ascend._310p.attention.attention_v1.AscendAttentionBackend310` (paged `NZ` KV cache `[2, num_blocks, num_kv_heads * head_size // 16, block_size, 16]`).
   - **DeepSeek Sparse Attention (DSA / DSV4 Compress-4 & Compress-128):** Routes to `vllm_ascend.attention.dsa_v1.AscendDSABackend` (and `AscendDSAC4Backend` / `AscendDSAC128Backend` for indexer/compressor caches).
   - **Sparse Flash Attention (SFA / DSV3.2 / GLM-5 Lightning Indexer):** Routes to `vllm_ascend.attention.sfa_v1.AscendSFABackend` when sparse top-k indexers are configured.
   - **Multi-Head Latent Attention (MLA / DSV3 / R1 / Kimi K2):** Routes to `vllm_ascend.attention.mla_v1.AscendMLABackend` (`use_mla=True`), splitting KV cache into `nope` (`kv_lora_rank=512`) and `rope` (`qk_rope_head_dim=64`) planes and enabling `VLLM_ASCEND_ENABLE_MLAPO=1` fused decode preprocessing when supported.
   - **FA3 quantized attention (`fa3_v1.py`):** Selected when `fa3_quant` per-head FP8/INT8 KV scales (`q_scale`, `k_scale`, `v_scale`) are present in the quantization config.
   - **Standard GQA/MHA (`attention_v1.py`):** Routes to `AscendAttentionBackend` using `torch_npu.npu_fused_infer_attention_score` (FIA) or `npu_paged_attention`.
2. **Verify `AscendAttentionState` transitions (`vllm_ascend/attention/attention_v1.py`):**
   - `AscendAttentionMetadataBuilder` classifies each step into `PrefillNoCache`, `PrefillCacheHit`, `DecodeOnly`, or `ChunkedPrefill` (which also covers mixed prefill+decode and speculative multi-token verification).
   - `FULL_DECODE_ONLY` ACLGraph capture requires `DecodeOnly` (or uniform query length `1 + num_spec_tokens` per request). If a metadata builder misclassifies a uniform decode step, the runner falls back to eager execution.

## Verification

- Check startup logs for the resolved attention backend class per layer group and confirm that KV cache tensor shapes/strides match `get_kv_cache_shape()` of the selected backend.
