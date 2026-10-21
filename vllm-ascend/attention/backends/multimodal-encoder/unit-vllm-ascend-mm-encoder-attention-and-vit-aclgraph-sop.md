---
category: sop
conditions:
  component: mm_encoder
  framework: vllm-ascend
domain_path: vllm-ascend/attention/backends/multimodal-encoder
evidence_strength: software_contract
id: unit-vllm-ascend-mm-encoder-attention-and-vit-aclgraph-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/mm_encoder_attention.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/worker/encoder_acl_graph.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/patch/platform/patch_vision.py
  source_digest: 09c6edef638109ffae8d4563bd911ef6620dc26cd77a238ea1a2f815cff13ae7
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/mm_encoder_attention.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Multimodal vision encoder attention (AscendMMEncoderAttention), ViT DP/TP sharding,
  and Encoder ACLGraph SOP
triggers:
  error_signatures:
  - AscendMMEncoderAttention fails when vision sequence lengths vary across images
    or video frames
  - vision encoder TP fails when num_vision_heads is not divisible by tensor_parallel_size
  - EncoderACLGraphManager recapture storm on variable-resolution image tiles
  keywords:
  - AscendMMEncoderAttention
  - mm_encoder_attention
  - encoder_acl_graph
  - EncoderACLGraphManager
  - mm-encoder-tp-mode
  - npu_fusion_attention
  - cu_seqlens
---
# Multimodal vision encoder attention (AscendMMEncoderAttention), ViT DP/TP sharding, and Encoder ACLGraph SOP

## Scope and preconditions

Use this SOP when deploying or optimizing multimodal Vision-Language Models (Qwen2.5-VL, Qwen3-VL, InternVL3.5, Kimi-K2.5/K2.6/K3, MiniMax-M3, PaddleOCR-VL, DeepSeek-OCR-2) that execute vision encoders through `AscendMMEncoderAttention` (`vllm_ascend/ops/mm_encoder_attention.py`) and optional `EncoderACLGraphManager` (`vllm_ascend/worker/encoder_acl_graph.py`).

## Procedure

1. **Verify varlen `TND` packing and `npu_fusion_attention` constraints (`vllm_ascend/ops/mm_encoder_attention.py`):**
   - `AscendMMEncoderAttention.forward_oot` reshapes variable-length image/video patch sequences into packed `TND` layout `[total_tokens, num_heads, head_dim]` using cumulative sequence lengths `cu_seqlens` (`actual_seq_qlen` / `actual_seq_kvlen` as `list[int]`).
   - When `head_dim` is not a multiple of 16 (or exceeds the `npu_fusion_attention` hardware limit of `512`), `AscendMMEncoderAttention` pads `Q`, `K`, `V` along `head_dim` to the next multiple of 16 before calling `torch_npu.npu_fusion_attention(..., input_layout="TND", sparse_mode=0)` and slices `[:, :, :ori_head_dim]` afterward.
2. **Select `--mm-encoder-tp-mode data` when vision heads do not divide `tp_size`:**
   - Many ViT backbones have `num_heads` (e.g., 16, 25, or 27) that are not divisible by large language-model `tensor_parallel_size` (e.g., `TP=8` or `TP=16`).
   - Pass `--mm-encoder-tp-mode data` (or rely on `_is_vit_use_data_parallel` fallback in `patch_vision.py` / `kimi_k3.py`) so the vision encoder shards image batches across TP ranks along the data dimension instead of splitting attention heads.
3. **Configure `EncoderACLGraphManager` (`vllm_ascend/worker/encoder_acl_graph.py`) for fixed-bucket ViT graph replay:**
   - `EncoderACLGraphManager` captures ACLGraphs for ViT forward passes keyed by padded token count buckets (`cu_seqlens` padded with a trailing dummy segment to match the bucket size). Keep `--mm-processor-cache-gb 0` when benchmarking cold ViT throughput or using cross-process EPD encoder caching.

## Verification

- Send single-image and multi-image requests of varying resolutions and verify that `AscendMMEncoderAttention` completes without `EZ1001` tiling errors and matches eager ViT embeddings.
