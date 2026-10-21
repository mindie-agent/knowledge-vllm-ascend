---
category: workflow
conditions:
  framework: vllm-ascend
  model_family:
  - deepseek-v4
  - deepseek-v4-flash
  - deepseek-v4-pro
domain_path: vllm-ascend/models/deepseek/v4-pro-flash
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-deepseek-v4-flash-pro-hash-topk-and-rope-workflow
inherits_from:
- unit-vllm-ascend-dsv4-v41-qli-v2-compressor-and-engram-axioms
- unit-vllm-ascend-dsv4-v41-sparse-mla-lightning-indexer-and-mtp-sop
- unit-vllm-ascend-llm-base-dspark-and-dflash-proposer-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/deepseek_v4/model.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/deepseek_v4/compressor.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/deepseek_v4/indexer.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/deepseek_v4/dspark.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/DeepSeek-V4-Flash.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/models/DeepSeek-V4-Pro.md
  source_digest: 257c415f820f6d203b78b3e1000b5e30c475281db1d99a270c65cba52a3f62ab
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/deepseek_v4/model.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: DeepSeek-V4-Flash & DeepSeek-V4-Pro (mHC, Compress-4/128 DSA, Hadamard Indexer,
  DSpark/MTP) workflow
triggers:
  error_signatures:
  - DeepSeek-V4-Pro multi-node W4A8-MTP times out during weight load or HCCL collective
    init
  - DeepSeek-V4-Flash-0731 DSpark draft model fails when rope_dsv4 inverse RoPE or
    Markov head is miswired
  - AscendDSAC4Backend or AscendDSAC128Backend block size mismatch on DSV4_BLOCK_SIZES
  keywords:
  - DeepSeek-V4-Flash
  - DeepSeek-V4-Pro
  - DeepSeek-V4-Flash-0731
  - AscendCompressorStateCache
  - AscendDeepseekV4IndexerCache
  - DeepseekV4DSparkModel
  - enable_dsa_cp
  - rope_dsv4
  - mHC
---
# DeepSeek-V4-Flash & DeepSeek-V4-Pro (mHC, Compress-4/128 DSA, Hadamard Indexer, DSpark/MTP) workflow

## Trigger and scope

Use this workflow when deploying or debugging `DeepSeek-V4-Flash` (`DeepSeek-V4-Flash-w8a8-mtp` and `DeepSeek-V4-Flash-0731-w8a8` with DSpark) and `DeepSeek-V4-Pro` (`DeepSeek-V4-Pro-w4a8-mtp`) on Atlas A2/A3/A5 (`vllm_ascend/models/deepseek_v4/`, `DeepSeek-V4-Flash.md`, `DeepSeek-V4-Pro.md`).

## Stage 1 — Configure Manifold-Constrained Hyper-Connections (mHC) and Compress-4 / Compress-128 caches

1. **Verify `DSV4_BLOCK_SIZES` and multi-group cache specs (`compressor.py`, `indexer.py`):**
   - DeepSeek-V4 combines `Compress-4` (`compress_ratio=4`, `overlap=True`, `coff=2`, `AscendDSAC4Backend` / `AscendDSAC4StateBackend`) and `Compress-128` (`compress_ratio=128`, `overlap=False`, `coff=1`, `AscendDSAC128Backend` / `AscendDSAC128StateBackend`) with Manifold-Constrained Hyper-Connections (`hc_mult`).
   - Always launch with `--block-size 128` and `--no-enable-prefix-caching` (unless hybrid compressor state prefix caching is explicitly validated for the target revision).
   - On Ascend 950 (`A5`), `AscendDeepseekV4IndexerCache` automatically switches indexer cache dtype to `float8_e4m3fn` with `float32` scale and forces `Compressor.norm` to `float32`.

## Stage 2 — Wire Hadamard activation rotation, `IndexerOverlapPlan`, and `rope_dsv4`

1. **Verify Hadamard rotation and compressor/indexer stream overlap (`vllm_ascend/models/deepseek_v4/indexer.py`):**
   - `rotate_activation` pads hidden dimension to $2^{\lceil \log_2 D ceil}$ and applies the Hadamard projection scaled by $D^{-0.5}$ before `quantize_query` / `quantize_key_and_update_cache`.
   - `IndexerOverlapPlan` overlaps main-attention compressed KV computation and scatter on `aux_stream` with Indexer top-$k$ selection on the primary stream.
2. **Select serial `mtp` vs block `DSpark` (`DeepSeek-V4-Flash-0731`):**
   - Standard `DeepSeek-V4-Flash-w8a8-mtp` and `DeepSeek-V4-Pro-w4a8-mtp` use `--speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}'`.
   - `DeepSeek-V4-Flash-0731-w8a8` uses `DeepseekV4DSparkModel` (`vllm_ascend/models/deepseek_v4/dspark.py`), which applies layer-specific forward and inverse DSA RoPE (`_apply_dsv4_rope(..., inverse=True)`) across `dspark_target_layer_ids`, `DSparkMarkovHead`, and `DSparkConfidenceHead`.

## Stage 3 — Deploy Single-Node `V4-Flash` and Multi-Node `V4-Pro` with production timeouts

1. **Single-node `DeepSeek-V4-Flash` (1 x A2 8-card or 1 x A3 8/16-card, `--shm-size=512g`):**
   - Use `--tensor-parallel-size 8 --enable-expert-parallel --quantization ascend --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4` and `--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "enable_cpu_binding": true, "enable_dsa_cp": true, "multistream_overlap_shared_expert": true}'`.
2. **Multi-node `DeepSeek-V4-Pro-w4a8-mtp` (4 x A2 or 2 x A3 nodes, `DP4 + TP8 + EP32`):**
   - Set extended initialization timeouts: `VLLM_ENGINE_READY_TIMEOUT_S=3600`, `HCCL_CONNECT_TIMEOUT=7200`, `ASCEND_CONNECT_TIMEOUT=10000`, `ASCEND_TRANSFER_TIMEOUT=10000`, `VLLM_RPC_TIMEOUT=1800000`, `HCCL_BUFFSIZE=512`, `ACL_OP_INIT_MODE=1`, and `--safetensors-load-strategy 'prefetch'`.
