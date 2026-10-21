---
category: sop
conditions:
  framework: vllm-ascend
  model_family:
  - qwen3
  - qwen3_moe
  - qwen3_next
  - qwen3_dspark
domain_path: vllm-ascend/models/qwen/qwen3-moe-next
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-qwen3-235b-30b-coder-next-and-dspark-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-llm-base-dspark-and-dflash-proposer-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/qwen3_dspark.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3-235B-A22B.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3-30B-A3B.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3-Coder-30B-A3B.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3-Next.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3-Dense.md
  source_digest: 7bf1ed2ebabc1597de4187888da473ea9ff4fa58fcb4f5f5b3b436d183fbc5e2
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/qwen3_dspark.py
  source_type: model_deployment_tutorial
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Qwen3 Dense, Qwen3-MoE (235B-A22B / 30B-A3B / Coder), Qwen3-Next (80B-A3B),
  and Qwen3-DSpark SOP
triggers:
  error_signatures:
  - Qwen3-Next-80B-A3B fails when Triton Ascend is missing or out of date
  - Qwen3-235B-A22B or Qwen3-30B-A3B with DSpark block7 checkpoint fails confidence_head
    weight loading
  - Qwen3 long-context YaRN scaling exceeds 40960 without VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  keywords:
  - Qwen3-235B-A22B
  - Qwen3-30B-A3B
  - Qwen3-Coder-30B-A3B
  - Qwen3-Next-80B-A3B
  - AscendQwen3DSparkForCausalLM
  - qwen3_dspark
  - yarn
---
# Qwen3 Dense, Qwen3-MoE (235B-A22B / 30B-A3B / Coder), Qwen3-Next (80B-A3B), and Qwen3-DSpark SOP

## Scope and preconditions

Use this SOP when deploying or tuning `Qwen3-Dense` (`0.6B`..`32B`), `Qwen3-30B-A3B` / `Qwen3-Coder-30B-A3B`, `Qwen3-235B-A22B`, `Qwen3-Next-80B-A3B-Instruct`, and `Qwen3-DSpark` speculative decoding on Atlas A2/A3 (`Qwen3-235B-A22B.md`, `Qwen3-Next.md`, `vllm_ascend/models/qwen3_dspark.py`).

## Procedure

1. **Deploy `Qwen3-235B-A22B` and `Qwen3-30B-A3B` / `Coder` with EP and `FULL_DECODE_ONLY`:**
   - **`Qwen3-30B-A3B` / `Qwen3-Coder-30B-A3B`:** Fits on 2–4 A2/A3 cards (BF16) or 1–2 cards (`W8A8`). For 128K–256K long context with YaRN, set `export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and `--hf-overrides '{"rope_theta": 1000000, "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 40960}, "max_model_len": 262144}'`.
   - **`Qwen3-235B-A22B`:** Deploy BF16 on 1 x Atlas 800 A3 (`TP16` or `DP2+TP8+EP`) or 2 x Atlas 800 A2 nodes; deploy `W8A8` on a single 8-card A2 node (`TP8 + --enable-expert-parallel --quantization ascend`).
2. **Deploy `Qwen3-Next-80B-A3B-Instruct` (`Qwen3-Next.md`):**
   - `Qwen3-Next` combines high-sparsity MoE (`80B` total, `3B` active), hybrid GatedDeltaNet linear + full attention, and native MTP using `triton-ascend` kernels (`vllm_ascend/ops/triton/fla/`).
   - Launch on 4 or 8 A2/A3 cards: `vllm serve Qwen3-Next-80B-A3B-Instruct --tensor-parallel-size 4 --max-model-len 32768 --max-num-batched-tokens 4096 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`.
3. **Enable `Qwen3-DSpark` block speculative decoding (`vllm_ascend/models/qwen3_dspark.py`):**
   - `AscendQwen3DSparkForCausalLM` supports `dspark_qwen3_*_block7` checkpoints with `DSparkConfidenceHead` (`proj.bias=True`, `params_dtype=torch.float32`, `markov_rank=256`).
   - When the target Qwen3 model uses rotary quantization (`rotation_path`), `load_weights` automatically right-multiplies `fc.` weights by the FP32 `rotation_weight` matrix.

## Verification

- Verify `Qwen3-Next` and `Qwen3-235B-A22B` pass `FULL_DECODE_ONLY` warmup and achieve expected GSM8K/C-Eval accuracy in `ais_bench`.
