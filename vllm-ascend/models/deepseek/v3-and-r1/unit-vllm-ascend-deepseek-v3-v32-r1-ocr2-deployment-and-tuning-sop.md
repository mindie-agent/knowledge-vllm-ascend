---
category: sop
conditions:
  framework: vllm-ascend
  model_family:
  - deepseek-v3
  - deepseek-v3.1
  - deepseek-v3.2
  - deepseek-r1
  - deepseek-ocr-2
domain_path: vllm-ascend/models/deepseek/v3-and-r1
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-deepseek-v3-v32-r1-ocr2-deployment-and-tuning-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-hccl-and-dp-collective-invariants
- unit-vllm-ascend-attention-v1-mla-dsa-sfa-fa3-routing-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/DeepSeek-V3.2.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/DeepSeek-V3.1.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/DeepSeek-R1.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/DeepSeekOCR2.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/deepseek_mtp.py
  source_digest: 0550535399ae7e2c29a8c0b9f8c841dfacfa60a82a30098494c065ef42ef9d10
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/DeepSeek-V3.2.md
  source_type: model_deployment_tutorial
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: DeepSeek-V3 / V3.1 / V3.2 (Sparse SFA) / R1 / DeepSeek-OCR-2 deployment and
  tuning SOP
triggers:
  error_signatures:
  - DeepSeek-V3.2 or R1 decode TPOT regresses when VLLM_ASCEND_ENABLE_MLAPO is disabled
    or HCCL_OP_EXPANSION_MODE is not AIV
  - DeepSeek-OCR-2 outputs corrupted tokens when VLLM_ASCEND_ENABLE_NZ=1 or fuse_qknorm_rope
    is enabled
  - multi-node DeepSeek-V3.2 W8A8 fails on mismatched data-parallel-address or insufficient
    HCCL_BUFFSIZE
  keywords:
  - DeepSeek-V3
  - DeepSeek-V3.1
  - DeepSeek-V3.2
  - DeepSeek-R1
  - DeepSeek-OCR-2
  - VLLM_ASCEND_ENABLE_MLAPO
  - deepseek_mtp
  - HCCL_OP_EXPANSION_MODE
  - fuse_qknorm_rope
---
# DeepSeek-V3 / V3.1 / V3.2 (Sparse SFA) / R1 / DeepSeek-OCR-2 deployment and tuning SOP

## Scope and preconditions

Use this SOP when deploying or tuning DeepSeek-V3, DeepSeek-V3.1, DeepSeek-V3.2 (`DeepSeek-V3.2-W8A8` with Sparse Flash Attention), DeepSeek-R1 (`DeepSeek-R1-W8A8` / `W4A8`), and `DeepSeek-OCR-2` on Atlas 800 A3 (`64G x 16`) or Atlas 800 A2 (`64G x 8`) nodes (`DeepSeek-V3.2.md`, `DeepSeek-R1.md`, `DeepSeekOCR2.md`).

## Procedure

1. **Configure DeepSeek-V3 / V3.1 / V3.2 / R1 single-node A3 (`DP2 + TP8 + EP`) and two-node A2 topologies:**
   - **Single Atlas 800 A3 (`64G x 16`) for `DeepSeek-V3.2-W8A8` / `DeepSeek-R1-W8A8`:**
     - Export `HCCL_OP_EXPANSION_MODE="AIV"`, `VLLM_ASCEND_ENABLE_MLAPO=1` (fuses MLA decode preprocessing), `HCCL_BUFFSIZE=200` (or `256`–`512` for large EP), `OMP_PROC_BIND=false`, `OMP_NUM_THREADS=10`, and `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`.
     - Launch with `--data-parallel-size 2 --tensor-parallel-size 8 --enable-expert-parallel --quantization ascend --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` and `--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'`.
   - **Two-node Atlas 800 A2 (`2 x 8 x 64G`):**
     - Set `--data-parallel-size 2 --data-parallel-size-local 1 --tensor-parallel-size 8 --enable-expert-parallel`, bind `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME` to the local NIC, and point `--data-parallel-address` on both ranks to the master rank's IP.
2. **Apply `DeepSeek-OCR-2` vision-language invariants (`DeepSeekOCR2.md`):**
   - `DeepSeek-OCR-2` runs on 1 Atlas 800 A2/A3 card (`--tensor-parallel-size 1`).
   - Explicitly set `export VLLM_ASCEND_ENABLE_NZ=0`, `--mm-processor-cache-gb 0`, `--no-enable-prefix-caching`, and disable QKNorm+RoPE fusion via `--additional-config '{"enable_cpu_binding": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"fuse_qknorm_rope": false}}'` to prevent vision/decoder RoPE layout corruption.

## Verification

- Verify `DeepSeek-V3.2` / `R1` online service startup, confirm `FULL_DECODE_ONLY` graph capture and `deepseek_mtp` speculative acceptance, and validate GSM8K / C-Eval accuracy via `ais_bench`.
