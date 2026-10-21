---
category: sop
conditions:
  framework: vllm-ascend
  model_family:
  - glm-4
  - glm-5
  - glm-5.2
  - minimax-m2
  - minimax-m3
domain_path: vllm-ascend/models/glm-and-minimax/glm-minimax
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-glm4-glm5-glm52-and-minimax-m2-m3-adaptation-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-hccl-and-dp-collective-invariants
- unit-vllm-ascend-attention-v1-mla-dsa-sfa-fa3-routing-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/minimax_m3/msa_m3.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/minimax_m3/minimax_m3.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/minimax_m3/minimax_m3_vl.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/patch/worker/patch_minimax_m2.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/GLM5.2.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/GLM5.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/MiniMax-M3.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/MiniMax-M2.md
  source_digest: 4ac781e5b4605fcaa37f8dece347601068a1ecd5dd43ae2c3af480ccb5e7c22a
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/minimax_m3/msa_m3.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: GLM-4.x / GLM-5 / GLM-5.2 (W4A8C8 + DSA-CP + MTP) and MiniMax-M2 / MiniMax-M3
  (Sparse MSA Indexer) SOP
triggers:
  error_signatures:
  - GLM-5.2 MTP speculative decoding fails when enforce_eager is not set inside speculative-config
  - MiniMax-M3 tool parser fails in container when _rust_tool_parser is not built
    or fuse_norm_quant is enabled
  - MiniMax-M3 sparse indexer ASCEND_MINIMAX_M3_SPARSE_INDEXER miscounts active decode
    or prefill requests under FIA padding
  keywords:
  - GLM-5.2
  - GLM-5
  - GLM4.x
  - enable_sparse_li_c8
  - MiniMax-M3
  - MiniMax-M2
  - ASCEND_MINIMAX_M3_SPARSE_INDEXER
  - minimax_m3_sparse_attn
  - msa_m3
---
# GLM-4.x / GLM-5 / GLM-5.2 (W4A8C8 + DSA-CP + MTP) and MiniMax-M2 / MiniMax-M3 (Sparse MSA Indexer) SOP

## Scope and preconditions

Use this SOP when deploying or debugging `GLM-4.x`, `GLM-5`, `GLM-5.2` (`GLM-5.2-w4a8c8` / `w8a8` up to 1M context) and `MiniMax-M2` / `MiniMax-M3` (`vllm_ascend/models/minimax_m3/`, `patch_minimax_m2.py`) on Atlas A2, Atlas A3, and Ascend 950 (`A5`).

## Procedure

1. **Deploy `GLM-5` and `GLM-5.2-w4a8c8` on Atlas 800 A3 (`GLM5.2.md`):**
   - Set `export VLLM_ASCEND_ENABLE_FUSED_MC2=0`, `HCCL_OP_EXPANSION_MODE="AIV"`, `OMP_NUM_THREADS=1`, and `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`.
   - Launch on 1 x A3 (`64G x 16` or `128G x 8`) with `--data-parallel-size 2 --tensor-parallel-size 8 --enable-expert-parallel --quantization ascend --safetensors-load-strategy prefetch --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice`.
   - Enable Sparse Lightning-Indexer C8 and DSA-CP via `--additional-config '{"enable_dsa_cp": true, "enable_sparse_li_c8": true, "enable_balance_scheduling": true, "multistream_overlap_shared_expert": true}'`.
   - **Critical MTP rule:** GLM-5/5.2 does not support graph-mode speculative draft execution; always pass `"enforce_eager": true` inside `--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'` while keeping target decode in `FULL_DECODE_ONLY`.
2. **Deploy `MiniMax-M3` multimodal sparse attention (`vllm_ascend/models/minimax_m3/msa_m3.py`, `MiniMax-M3.md`):**
   - `MiniMax-M3` uses `AscendMiniMaxM3IndexerBackend` (`ASCEND_MINIMAX_M3_SPARSE_INDEXER`, `head_size=128`, `block_size=128`) and NPU/Triton sparse kernels (`minimax_m3_sparse_attn`, `minimax_m3_index_decode`, with dedicated `msa_m3_triton_a5.py` on Ascend 950 where index-K is replicated across TP ranks).
   - `_active_decode_num_reqs` and `_active_prefill_num_reqs` strip FIA/SP graph tail padding segments before launching the sparse indexer.
   - Build the Rust tool parser (`./build_rust.sh` in `/vllm-workspace/vllm` for `_rust_tool_parser`) when using `MiniMax-M3` tool calling, set `--reasoning-parser minimax_m3`, and pass `"fuse_norm_quant": false` inside `ascend_compilation_config` (plus `--speculative-config '{"model": "...", "method": "eagle3", "num_speculative_tokens": 3}'` for W8A8).

## Verification

- Verify `GLM-5.2` 135K-context requests with `enable_sparse_li_c8` and `MiniMax-M3` multimodal image/text requests complete with zero FIA padding index faults.
