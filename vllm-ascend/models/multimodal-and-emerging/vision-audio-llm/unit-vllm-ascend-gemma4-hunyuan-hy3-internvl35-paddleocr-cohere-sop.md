---
category: sop
conditions:
  framework: vllm-ascend
  model_family:
  - gemma4
  - hy3-preview
  - hunyuan-a13b
  - internvl3.5
  - paddleocr-vl
  - cohere-transcribe
domain_path: vllm-ascend/models/multimodal-and-emerging/vision-audio-llm
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-gemma4-hunyuan-hy3-internvl35-paddleocr-cohere-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-mm-encoder-attention-and-vit-aclgraph-sop
- unit-vllm-ascend-310p-dedicated-ops-fp16-cast-and-nz-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Hy3-preview.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Hunyuan-A13B-Instruct.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Gemma4.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/InternVL3.5.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/PaddleOCR-VL.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Cohere-Transcribe.md
  source_digest: ca93aa49be3615db8848edd6dbfcf64853847d48df9e93f28332e9202502f67b
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Hy3-preview.md
  source_type: model_deployment_tutorial
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Gemma4, Tencent Hy3-preview (295B-A21B) / Hunyuan-A13B, InternVL3.5, PaddleOCR-VL,
  and Cohere Transcribe SOP
triggers:
  error_signatures:
  - Hy3-preview 295B MoE checkpoint exhausts host RAM during startup when --enable-ep-weight-filter
    is omitted
  - Cohere Transcribe ASR fails at startup when --block-size is below 128 or librosa
    is missing
  - PaddleOCR-VL on Atlas 300I DUO fails on bfloat16 or PP-DocLayoutV2 environment
    conflict
  keywords:
  - Gemma4
  - Hy3-preview
  - Hunyuan-A13B-Instruct
  - InternVL3.5
  - PaddleOCR-VL
  - Cohere-Transcribe
  - cohere_asr
  - enable-ep-weight-filter
  - hy_v3
---
# Gemma4, Tencent Hy3-preview (295B-A21B) / Hunyuan-A13B, InternVL3.5, PaddleOCR-VL, and Cohere Transcribe SOP

## Scope and preconditions

Use this SOP when deploying `Gemma4` (dense and MoE on A2/A3/A5), Tencent `Hy3-preview` (`295B-A21B` + `3.8B` MTP) and `Hunyuan-A13B-Instruct`, `InternVL3.5` (`38B` and `241B-A28B-w8a8`), `PaddleOCR-VL-0.9B`, and `Cohere Transcribe` (`cohere_asr` 2B Conformer) on `vllm-ascend`.

## Procedure

1. **Deploy `Hy3-preview` (`295B-A21B`) and `Hunyuan-A13B-Instruct` (`Hy3-preview.md`, `Hunyuan-A13B-Instruct.md`):**
   - **`Hy3-preview` on 1 x Atlas A3 (`64G x 16`, ~58 GB/card):** Launch with `HCCL_OP_EXPANSION_MODE=AIV`, `--tensor-parallel-size 16 --enable-expert-parallel --enable-ep-weight-filter` (essential to skip non-local expert weights during loading and prevent host RAM exhaustion), `--speculative-config.method mtp --speculative-config.num_speculative_tokens 1`, and `--tool-call-parser hy_v3 --reasoning-parser hy_v3 --enable-auto-tool-choice`.
   - **`Hunyuan-A13B-Instruct` (`80B-A13B`):** Runs on 4 A2/A3 NPUs (`--tensor-parallel-size 4 --max-model-len 32768 --gpu-memory-utilization 0.90`) with `PIECEWISE` ACLGraph capture (~18s warmup).
2. **Deploy `Gemma4` (A2/A3/A5) and `InternVL3.5` (`38B` / `241B-A28B-w8a8`):**
   - **`Gemma4` (`Gemma4.md`):** Validated across Atlas A2, Atlas A3, and Ascend 950 (`A5`) in both eager and `FULL_DECODE_ONLY` / `PIECEWISE` ACLGraph modes (`--tensor-parallel-size 4 --max-model-len 32768`, plus `--enable-expert-parallel` for MoE variants).
   - **`InternVL3.5` (`InternVL3.5.md`):** For `InternVL3_5-241B-A28B-w8a8` on A3, export `VLLM_ASCEND_ENABLE_FUSED_MC2=1`, `HCCL_BUFFSIZE=1536`, and launch with `--tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel --enable-chunked-prefill --mm-processor-cache-gb 0 --safetensors-load-strategy 'prefetch'` and `--additional-config '{"enable_weight_nz_layout": true, "enable_cpu_binding": true}'`.
3. **Deploy `PaddleOCR-VL-0.9B` and `Cohere Transcribe` (`PaddleOCR-VL.md`, `Cohere-Transcribe.md`):**
   - **`PaddleOCR-VL-0.9B` (NaViT + ERNIE-4.5-0.3B):** Runs on 1 A2 or Atlas 300I DUO (`310P`) card (`--mm-processor-cache-gb 0 --no-enable-prefix-caching --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, plus `--dtype float16` and CANN `>= 9.0.0` for graph mode on 310P). Keep `PP-DocLayoutV2` in a separate Python virtualenv to avoid dependency conflicts.
   - **`Cohere Transcribe` (`cohere-transcribe-03-2026` / `arabic-07-2026`):** Ensure `librosa` is installed, and launch on 1 A2 NPU with `--trust-remote-code --tensor-parallel-size 1 --dtype bfloat16 --enforce-eager --block-size 128` (`block_size >= 128` is mandatory).

## Verification

- Confirm HTTP 200 responses and benchmark parity across text/chat (`Gemma4`, `Hy3-preview`, `Hunyuan-A13B`), vision/OCR (`InternVL3.5`, `PaddleOCR-VL`), and `/v1/chat/completions` `audio_url` transcription (`Cohere Transcribe`).
