---
category: sop
conditions:
  framework: vllm-ascend
  model_family:
  - qwen3_vl
  - qwen3_omni
  - qwen3_asr
  - qwen3_embedding
  - qwen3_reranker
domain_path: vllm-ascend/models/qwen/vl-omni-audio-rag
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-qwen3-vl-omni-asr-embedding-reranker-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-mm-encoder-attention-and-vit-aclgraph-sop
- unit-vllm-ascend-310p-dedicated-ops-fp16-cast-and-nz-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3-ASR-1.7B.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3-VL-Embedding.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3-VL-Reranker.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3-VL-235B-A22B-Instruct.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/patch/worker/patch_qwen3vl.py
  source_digest: 46d74236810092b920b0f15da755304c4b4c0c81a05d3e7137246ca72811e974
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md
  source_type: model_deployment_tutorial
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Qwen3-VL (235B/30B), Qwen3-Omni-30B-A3B-Thinking, Qwen3-ASR-1.7B, and Qwen3-VL-Embedding/Reranker
  SOP
triggers:
  error_signatures:
  - Qwen3-VL-Reranker returns invalid scores when sequence classification hf_overrides
    or jinja template is missing
  - Qwen3-ASR-1.7B or Qwen3-VL-Embedding on Atlas 300I DUO fails when triton-ascend
    is installed or max-model-len is unset
  - Qwen3-Omni-30B-A3B-W8A8 accuracy drops when attention projections use dynamic
    instead of static W8A8
  keywords:
  - Qwen3-VL
  - Qwen3-Omni-30B-A3B
  - Qwen3-ASR-1.7B
  - Qwen3-VL-Embedding
  - Qwen3-VL-Reranker
  - classifier_from_token
  - is_original_qwen3_reranker
  - patch_qwen3vl
---
# Qwen3-VL (235B/30B), Qwen3-Omni-30B-A3B-Thinking, Qwen3-ASR-1.7B, and Qwen3-VL-Embedding/Reranker SOP

## Scope and preconditions

Use this SOP when deploying Qwen multimodal vision, omni-modal, speech, and retrieval models on Atlas A2, Atlas A3, and Atlas 300I DUO (`Ascend310P`): `Qwen3-VL-235B-A22B` / `30B-A3B`, `Qwen3-Omni-30B-A3B-Thinking`, `Qwen3-ASR-1.7B`, `Qwen3-Embedding` / `Qwen3-VL-Embedding`, and `Qwen3-Reranker` / `Qwen3-VL-Reranker`.

## Procedure

1. **Deploy `Qwen3-Omni-30B-A3B-Thinking` and `Qwen3-VL` (`Qwen3-Omni-30B-A3B-Thinking.md`, `patch_qwen3vl.py`):**
   - `Qwen3-Omni-30B-A3B` processes audio, video, image, and text inputs via the thinker component.
   - When quantizing `Qwen3-Omni-30B-A3B` to `W8A8` with `msmodelslim`, use the hybrid scheme: keep `embed_tokens`, `q_norm`/`k_norm`, and `mlp.gate` in `BF16`; quantize attention `q/k/v/o_proj` with **static** `W8A8`; and quantize MoE routed experts (`gate/up/down_proj`) with **dynamic** `W8A8`.
   - On Atlas 300I DUO (`310P`), `vllm_ascend/_310p/ops/qwen3vl_310.py` casts 3D vision patch conv weights to `ACL_FORMAT_FRACTAL_Z`.
2. **Deploy `Qwen3-ASR-1.7B` on A2 and 310P (`Qwen3-ASR-1.7B.md`):**
   - On Atlas A2: run with `--tensor-parallel-size 1 --max-model-len 4096 --enforce-eager`.
   - On Atlas 300I DUO (`310P`): uninstall `triton` and `triton-ascend`, and launch with `--dtype float16 --max-model-len 4096 --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false, "enable_npugraph_ex": false}}' --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 4]}'`.
3. **Configure `--runner pooling` for `Qwen3-VL-Embedding` and `Qwen3-VL-Reranker` (`Qwen3-VL-Embedding.md`, `Qwen3-VL-Reranker.md`):**
   - **Embedding:** Launch with `--runner pooling --max-model-len 1024` (and on 310P add `--dtype float16 --compilation-config '{"cudagraph_capture_sizes": [1024, 512]}' --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}'`).
   - **Reranker:** Pass `--runner pooling --chat-template ./qwen3_vl_reranker.jinja` and `--hf_overrides '{"classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'` with the `Qwen3VL` sequence-classification architecture override so yes/no token logits are converted into normalized relevance scores.

## Verification

- Verify `/v1/chat/completions` with `audio_url` / `image_url` payloads on `Qwen3-Omni` / `Qwen3-ASR`, and `/v1/embeddings` or `/v1/rerank` (`/v1/score`) on `Qwen3-VL-Embedding` / `Reranker`.
