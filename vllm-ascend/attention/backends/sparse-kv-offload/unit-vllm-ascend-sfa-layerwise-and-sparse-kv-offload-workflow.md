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
  - https://github.com/vllm-project/vllm-ascend/pull/10441
  - https://github.com/vllm-project/vllm-ascend/blob/23d1d9055f2161b3c15613c7fc309f7d5d2ef629/vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py#L320
  source_digest: 7b796d915eb5ce3215c0c4d45a4ae2fd20dc68b5a14a98e51aa934229ecb7de9
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/attention/sfa_kv_offload.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 2
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
  - SFA layer with has_indexer=False or skip_topk=True corrupts KV cache when reshape_cache_event is not recorded
  keywords:
  - sfa_kv_offload
  - sparse_kv_offload_manager
  - layerwise_and_sparse_kv_cache_offloading
  - enable_sparse_c8
  - sfa_pd_rd2h
  - SparseKVOffloadManager
  - keep_device_kv_cache
  - reshape_cache_event
  - skip_topk
---
# SFA layerwise KV buffer reuse and sparse C8 KV cache CPU offloading workflow

## Trigger and scope

Use this workflow when deploying long-context Sparse Flash Attention (SFA) models (such as DeepSeek-V3.2 or GLM-5) where HBM capacity cannot hold all layers' full KV caches during prefill or where only top-$k$ selected KV blocks should reside in HBM while cold KV blocks remain in host DRAM (`vllm_ascend/attention/sfa_kv_offload.py`, `sparse_kv_offload_manager.py`, `sfa_pd_rd2h/`).

## Stage 1 — Configure Prefill layerwise KV buffer reuse and Sparse C8 layout

1. **Understand the three-tier memory hierarchy (`layerwise_and_sparse_kv_cache_offloading.md`, `sparse_kv_offload_manager.py`):**
   - Under `sparse_kv_offload_config.enabled=true` and `keep_device_kv_cache=false` (`LIM + Copy-SFA`), full-history main MLA KV (`cKV / k_nope` and `k_pe`, `store_on_host=True`) is allocated in Decode host DRAM via MemFabric (shared across TP ranks within a DP group by TP0), whereas two structures remain in NPU HBM: (a) full-history Lightning Indexer `k_li` cache (plus quantization scale), and (b) finite-capacity main-KV TopK hot/recent-tail staging buffers. Setting `keep_device_kv_cache=true` allocates full-history main KV in NPU HBM as well (debug-only, zero HBM capacity savings).
   - On Prefill nodes, enable layerwise KV buffer reuse so that full-sequence HBM KV buffers are allocated for a small sliding window of layers (plus MTP layers) and offloaded asynchronously to host pinned DRAM (`SparseKVOffloadManager`) or pushed directly to Decode via `sfa_pd_rd2h`.
2. **Validate C8 quantization alignment:**
   - When `enable_sparse_c8` / `enable_sparse_li_c8` is enabled, verify that the per-token int8 KV latent cache and its FP16/BF16 scale tensor share identical block-table indexing and that MTP draft layers either maintain dedicated HBM KV buffers or match the target layer's compression spec.

## Stage 2 — Pipeline top-k Indexer selection with asynchronous H2D sparse KV gather

1. **Overlap Indexer top-$k$ selection, `skip_topk=True` index sharing, and H2D DMA (`vllm_ascend/attention/sfa_kv_offload.py`, `PR #10441`):**
   - At layer $l$, the Lightning Indexer computes top-$k$ block indices on the NPU compute stream while `SparseKVOffloadManager` prefetches or gathers selected KV blocks from pinned host DRAM onto an auxiliary H2D stream.
   - **Record `reshape_cache_event` even when `has_indexer=False` (`skip_topk=True`, `PR #10441`):** Layers without their own Indexer (`indexer=None`) or MTP draft steps (`step >= 1`) set `skip_topk=True` and reuse an upstream layer's or step-0's `topk_indices_buffer`. Even when TopK computation is skipped, the layer's KV producer must still record `reshape_cache_event` so downstream consumers wait for KV cache writes to finish.
   - Enforce stream event synchronization (`torch.npu.Event`) before `npu_sparse_flash_attention` consumes the gathered HBM staging buffer, and before the staging buffer is overwritten by layer $l+1$.
2. **Guard invalid `-1` top-$k$ padding indices before H2D gather:**
   - When `seq_len < topk`, unselected entries in `topk_indices` are `-1`. Clamp or mask `-1` indices before indexing host/device block tables so DMA descriptors never read out-of-bounds addresses.

## Stage 3 — Coordinate PD disaggregation via `sfa_pd_rd2h`

1. **Configure remote Device-to-Host / Host-to-Device transfer (`vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/`):**
   - In disaggregated SFA deployments, `sfa_pd_rd2h` (`connector.py`, `send_thread.py`, `read_thread.py`) streams newly computed main KV blocks from the Prefill node into the Decode node's host DRAM pool while streaming Indexer `k_li` directly into Decode HBM.
   - Verify that `--shm-size` in Docker is sized to hold the full host DRAM KV pool (`>= 512g` on A3 long-context nodes) and that `done_sending` / `done_receiving` acknowledgments complete before freeing prefill blocks.
