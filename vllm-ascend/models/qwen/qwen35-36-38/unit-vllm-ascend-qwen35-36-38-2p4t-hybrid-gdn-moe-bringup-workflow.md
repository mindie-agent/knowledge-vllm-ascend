---
category: workflow
conditions:
  framework: vllm-ascend
  model_family:
  - qwen3.5
  - qwen3.6
  - qwen3.8
  - qwen3.8-2.4t-a95b
domain_path: vllm-ascend/models/qwen/qwen35-36-38
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-qwen35-36-38-2p4t-hybrid-gdn-moe-bringup-workflow
inherits_from:
- unit-vllm-ascend-hccl-and-dp-collective-invariants
- unit-vllm-ascend-bailing-linear-and-gdn-metadata-builder-sop
- unit-vllm-ascend-step3p5-and-hybrid-mtp-cache-rollback-workflow
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3.8-2.4T-A95B.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3.5-397B-A17B.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3.6-35B-A3B.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/patch/worker/patch_qwen3_5.py
  source_digest: 26e6555b45538b26328d1cf9f742970708ad9157bfd984dc14a28d8f3e9c3111
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/Qwen3.8-2.4T-A95B.md
  source_type: model_deployment_tutorial
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Qwen3.5 (Dense/397B-A17B), Qwen3.6 (35B-A3B), and Qwen3.8-2.4T-A95B ultra-scale
  hybrid GDN+MoE workflow
triggers:
  error_signatures:
  - Qwen3.8-2.4T-A95B 2.33TiB checkpoint exhausts host memory or times out during
    multi-node safetensors load
  - Qwen3.5 or Qwen3.8 hybrid GDN + MoE layer hangs in EP collective when one DP rank
    is idle
  - qwen3_5_mtp speculative decoding fails on GDN recurrent state slot indexing
  keywords:
  - Qwen3.8-2.4T-A95B
  - Qwen3.5-397B-A17B
  - Qwen3.6-35B-A3B
  - Qwen3.5-27B
  - qwen3_5_mtp
  - patch_qwen3_5
  - safetensors-load-strategy
  - HCCL_BUFFSIZE_EP
---
# Qwen3.5 (Dense/397B-A17B), Qwen3.6 (35B-A3B), and Qwen3.8-2.4T-A95B ultra-scale hybrid GDN+MoE workflow

## Trigger and scope

Use this workflow when bringing up and deploying the `Qwen3.5` (`Qwen3.5-27B`, `Qwen3.5-397B-A17B`), `Qwen3.6` (`Qwen3.6-27B`, `Qwen3.6-35B-A3B`), and `Qwen3.8` (`Qwen3.8-27B`, `Qwen3.8-2.4T-A95B` with 2.4 trillion total / 95B active parameters) hybrid GatedDeltaNet + full-attention MoE models on Atlas A2/A3 (`Qwen3.8-2.4T-A95B.md`, `Qwen3.5-397B-A17B.md`, `vllm_ascend/patch/worker/patch_qwen3_5.py`).

## Stage 1 — Lazy Safetensors loading and HCCL EP buffer sizing for multi-TiB checkpoints

1. **Prevent host I/O and RAM exhaustion on `Qwen3.8-2.4T-A95B-w8a8` (`2.33 TiB`) / `BF16` (`4.89 TB`):**
   - Deploy `Qwen3.8-2.4T-A95B-w8a8` across **4 x Atlas 800 A3 (`64G x 16`) nodes** with topology `DP4 / TP16 / EP64` (`--data-parallel-size 4 -- data-parallel-size-local 1 --tensor-parallel-size 16 --enable-expert-parallel`).
   - Always pass `--safetensors-load-strategy lazy` so each rank mmaps and materializes only its own TP/EP shard instead of prefetching 2.33 TiB across shared storage.
   - Set extended startup and execution timeouts: `VLLM_ENGINE_READY_TIMEOUT_S=7200`, `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000`, `HCCL_BUFFSIZE=1024`, `HCCL_BUFFSIZE_EP=2048`, `HCCL_INTRA_PCIE_ENABLE=1`, `HCCL_INTRA_ROCE_ENABLE=0`, `HCCL_OP_EXPANSION_MODE="AIV"`, and `VLLM_ASCEND_ENABLE_FUSED_MC2=1`.

## Stage 2 — Wire hybrid GatedDeltaNet + Full-Attention caches and `patch_qwen3_5.py`

1. **Verify hybrid Mamba/GDN + Attention cache coordination (`patch_qwen3_5.py`, `patch_mamba_config.py`):**
   - `Qwen3.5` / `Qwen3.6` / `Qwen3.8` interleave `GatedDeltaNet` linear-recurrent layers with grouped-query full-attention layers.
   - Ensure `AscendGDNMetadataBuilder` (`gdn_attn_builder.py`) and the hybrid KV cache coordinator allocate both paged KV blocks and recurrent/conv state slots, with safe dummy-step padding (`PAD_SLOT_ID = -1`) when one DP rank is idle during EP all-to-all waves.

## Stage 3 — Enable `qwen3_5_mtp`, Prefix Caching, and `FULL_DECODE_ONLY` ACLGraph

1. **Configure speculative decoding and fused MC2:**
   - Pass `--enable-prefix-caching`, `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'`, `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, and `--additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":1}'`.
   - Verify all 4 A3 nodes complete `FULL_DECODE_ONLY` capture and pass multi-turn long-context (`131072`) requests without HCCL EP timeout or recurrent state corruption.
