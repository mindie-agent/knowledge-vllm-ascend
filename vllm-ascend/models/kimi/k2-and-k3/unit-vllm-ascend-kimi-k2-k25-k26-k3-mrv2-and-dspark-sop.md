---
category: sop
conditions:
  framework: vllm-ascend
  model_family:
  - kimi-k2
  - kimi-k2.5
  - kimi-k2.6
  - kimi-k3
domain_path: vllm-ascend/models/kimi/k2-and-k3
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-kimi-k2-k25-k26-k3-mrv2-and-dspark-sop
inherits_from:
- unit-vllm-ascend-kimi-k3-kda-mamba-hybrid-attention-axioms
- unit-vllm-ascend-kimi-k3-model-bringup-and-hybrid-cache-workflow
- unit-vllm-ascend-llm-base-dspark-and-dflash-proposer-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/kimi_k3.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/kimi_kda.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/kimi_kda_state.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/patch/platform/patch_kimi_k3_parsers.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Kimi-K2.6.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Kimi-K2.5.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Kimi-K2-Thinking.md
  source_digest: 498839ee000fd1d0731915613de1b98b562474c9da5c5357c686b5ae2cc0922d
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/kimi_k3.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Kimi-K2-Thinking, Kimi-K2.5, Kimi-K2.6 (Eagle3/DFlash), and native multimodal
  Kimi-K3 (KDA/MoE/MRV2) SOP
triggers:
  error_signatures:
  - Kimi-K2.5 or Kimi-K2.6 vision encoder fails when num_heads is not divisible by
    tensor_parallel_size
  - Kimi-K3 packed expert weight or latent MoE quantization mismatches ModelSlim ascend
    scheme
  - Kimi-K3 structural tag or reasoning/tool parser fails on streaming thinking responses
  keywords:
  - Kimi-K2-Thinking
  - Kimi-K2.5
  - Kimi-K2.6
  - Kimi-K3
  - kimi_k3
  - KimiK3Processor
  - kimi_kda
  - Kimi-K2.5-DFlash
  - kimi-k2.6-eagle3
---
# Kimi-K2-Thinking, Kimi-K2.5, Kimi-K2.6 (Eagle3/DFlash), and native multimodal Kimi-K3 (KDA/MoE/MRV2) SOP

## Scope and preconditions

Use this SOP when deploying or debugging `Kimi-K2-Thinking`, `Kimi-K2.5`, `Kimi-K2.6` (`Eco-Tech/Kimi-K2.6-W4A8` with `Kimi-K2.5-DFlash` or `kimi-k2.6-eagle3`), and native multimodal `Kimi-K3` (`vllm_ascend/models/kimi_k3.py`, `ops/kimi_kda.py`, `patch/platform/kimi_k3_*.py`) on Atlas A2/A3.

## Procedure

1. **Deploy `Kimi-K2.5` and `Kimi-K2.6-W4A8` on Single-Node A3 (`DP4 + TP4 + EP`) or Multi-Node A2 (`Kimi-K2.6.md`):**
   - Export `HCCL_OP_EXPANSION_MODE="AIV"`, `VLLM_ASCEND_ENABLE_MLAPO=1`, `HCCL_BUFFSIZE=600`, `OMP_NUM_THREADS=1`, and `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`.
   - Pass `--quantization ascend --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --tensor-parallel-size 4 --data-parallel-size 4 --enable-expert-parallel --mm-processor-cache-gb 0 --mm-encoder-tp-mode data --additional-config '{"enable_balance_scheduling": true}'`.
   - For speculative decoding on `Kimi-K2.6`, use either **DFlash** (`--speculative-config '{"method": "dflash", "model": "z-lab/Kimi-K2.5-DFlash", "num_speculative_tokens": 15}'`) or **EAGLE3** (`kimi-k2.6-eagle3`). Set `--max-model-len >= 35000` during accuracy evaluation so long thinking traces do not truncate.
2. **Operate native multimodal `Kimi-K3` (`vllm_ascend/models/kimi_k3.py`):**
   - `Kimi-K3` integrates a 2D-RoPE / patch-merger vision tower (`KimiK3VisionConfig`, `KimiK3Processor`) with a hybrid language backbone combining `KimiGatedDeltaNetAttention` (`kimi_kda.py`, `kimi_kda_state.py`), `MLA` (`MultiHeadLatentAttentionWrapper`), and latent/routed `FusedMoE`.
   - `_resolve_packed_expert_weight_name` maps `_weight` to `_weight_packed` for quantized schemes, and `_routed_latent_quant_config` restricts latent MoE projection quantization to native `"ascend"` (`ModelSlim`) checkpoints while preserving lazy `meta` parameters during `_move_module_to_device`.
   - Use the registered `kimi_k3` reasoning, tool, renderer, and structural-tag parsers (`vllm_ascend/patch/platform/patch_kimi_k3_parsers.py`, `patch_kimi_k3_renderer.py`) and pair with `VLLM_ASCEND_USE_V2_MODEL_RUNNER=1` (`MambaHybridModelState`) or DSpark when configured.

## Verification

- Verify multimodal image+text and thinking/tool-call requests on `Kimi-K2.6` and `Kimi-K3`, confirming clean parser extraction of `reasoning` and `tool_calls` fields and zero NaN/Inf across cold and warm prefix runs.
