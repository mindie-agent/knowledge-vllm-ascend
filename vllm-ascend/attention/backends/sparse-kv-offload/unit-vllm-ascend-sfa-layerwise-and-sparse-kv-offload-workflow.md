---
category: workflow
conditions:
  component: sparse_kv_offload
  framework: vllm-ascend
domain_path: vllm-ascend/attention/backends/sparse-kv-offload
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-sfa-layerwise-and-sparse-kv-offload-workflow
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-attention-v1-mla-dsa-sfa-fa3-routing-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/attention/sfa_kv_offload.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/connector.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md
  source_digest: 7b796d915eb5ce3215c0c4d45a4ae2fd20dc68b5a14a98e51aa934229ecb7de9
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/attention/sfa_kv_offload.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: SFA layerwise KV buffer reuse and sparse C8 KV cache CPU offloading workflow
triggers:
  error_signatures:
  - Prefill OOM on long-context SFA or MLA model when allocating full-layer HBM KV
    caches
  - sparse_kv_offload H2D gather reads stale host blocks or collides with Mooncake
    P2P send
  - enable_sparse_c8 layerwise buffer reuse corrupts MTP draft layer KV cache
  keywords:
  - sfa_kv_offload
  - sparse_kv_offload_manager
  - layerwise_and_sparse_kv_cache_offloading
  - enable_sparse_c8
  - sfa_pd_rd2h
  - SparseKVOffloadManager
---
# SFA layerwise KV buffer reuse and sparse C8 KV cache CPU offloading workflow

## Trigger and scope

Use this workflow when deploying long-context Sparse Flash Attention (SFA) models (such as DeepSeek-V3.2 or GLM-5) where HBM capacity cannot hold all layers' full KV caches during prefill or where only top-$k$ selected KV blocks should reside in HBM while cold KV blocks remain in host DRAM (`vllm_ascend/attention/sfa_kv_offload.py`, `sparse_kv_offload_manager.py`, `sfa_pd_rd2h/`).

## Stage 1 — Configure Prefill layerwise KV buffer reuse and Sparse C8 layout

1. **Understand the memory hierarchy (`layerwise_and_sparse_kv_cache_offloading.md`):**
   - In SFA models, the Lightning Indexer (`k_index`) requires a lightweight full-sequence index cache in HBM, whereas the heavy MLA latent KV cache (`kv_nope` + `k_rope`, or quantized `C8` int8 latent + scale) is only accessed at the top-$k$ selected token/block positions.
   - On Prefill nodes, enable layerwise KV buffer reuse so that full-sequence HBM KV buffers are allocated for a small sliding window of layers (plus MTP layers) and offloaded asynchronously to host pinned DRAM (`SparseKVOffloadManager`) or pushed directly to Decode via `sfa_pd_rd2h`.
2. **Validate C8 quantization alignment:**
   - When `enable_sparse_c8` / `enable_sparse_li_c8` is enabled, verify that the per-token int8 KV latent cache and its FP16/BF16 scale tensor share identical block-table indexing and that MTP draft layers either maintain dedicated HBM KV buffers or match the target layer's compression spec.

## Stage 2 — Pipeline top-k Indexer selection with asynchronous H2D sparse KV gather

1. **Overlap Indexer top-$k$ selection and H2D DMA (`vllm_ascend/attention/sfa_kv_offload.py`):**
   - At layer $l$, the Lightning Indexer computes top-$k$ block indices on the NPU compute stream while the `SparseKVOffloadManager` prefetches or gathers selected KV blocks from pinned host DRAM onto an auxiliary H2D stream.
   - Enforce stream event synchronization (`torch.npu.Event`) before `npu_sparse_flash_attention` consumes the gathered HBM staging buffer, and before the staging buffer is overwritten by layer $l+1$.
2. **Guard invalid `-1` top-$k$ padding indices before H2D gather:**
   - When `seq_len < topk`, unselected entries in `topk_indices` are `-1`. Clamp or mask `-1` indices before indexing host/device block tables so DMA descriptors never read out-of-bounds addresses.

## Stage 3 — Coordinate PD disaggregation via `sfa_pd_rd2h`

1. **Configure remote Device-to-Host / Host-to-Device transfer (`vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/`):**
   - In disaggregated SFA deployments, `sfa_pd_rd2h` (`connector.py`, `send_thread.py`, `read_thread.py`) streams newly computed KV blocks from the Prefill node directly into the Decode node's host DRAM pool and HBM indexer cache.
   - Verify that `--shm-size` in Docker is sized to hold the full host DRAM KV pool (`>= 512g` on A3 long-context nodes) and that `done_sending` / `done_receiving` acknowledgments complete before freeing prefill blocks.
