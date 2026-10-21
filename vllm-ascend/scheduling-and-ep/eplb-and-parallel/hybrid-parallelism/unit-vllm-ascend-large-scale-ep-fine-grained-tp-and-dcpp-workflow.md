---
category: workflow
conditions:
  component: hybrid_parallelism
  framework: vllm-ascend
domain_path: vllm-ascend/scheduling-and-ep/eplb-and-parallel/hybrid-parallelism
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-large-scale-ep-fine-grained-tp-and-dcpp-workflow
inherits_from:
- unit-vllm-ascend-hccl-and-dp-collective-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/core/scheduler_profiling_chunk.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/core/profiling_chunk_predictor.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/large_scale_ep.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/Fine_grained_TP.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md
  source_digest: 63f9e0e3efabd8f64b9ab934ea86c72734051864d96e07b41c837a714c73bf22
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/core/scheduler_profiling_chunk.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Large-scale EP distributed DP server, Fine-Grained TP (lmhead/oproj/embedding/mlp),
  and Dynamic Chunked PP (CPP) workflow
triggers:
  error_signatures:
  - finegrained_tp_config oproj_tensor_parallel_size fails in eager mode or when tensor_parallel_size
    > 1
  - Dynamic Chunked Pipeline Parallel (profiling_chunk_config) degrades prefill when
    async-scheduling is enabled
  - multi-node Large-Scale EP fails HCCS or RoCE ping check across A2 or A3 nodes
  keywords:
  - large_scale_ep
  - VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED
  - finegrained_tp_config
  - oproj_tensor_parallel_size
  - lmhead_tensor_parallel_size
  - profiling_chunk_config
  - dynamic_chunk_pipeline_parallel
  - hccn_tool
---
# Large-scale EP distributed DP server, Fine-Grained TP (lmhead/oproj/embedding/mlp), and Dynamic Chunked PP (CPP) workflow

## Trigger and scope

Use this workflow when designing or troubleshooting hybrid parallelism topologies on `vllm-ascend`:
1. **Multi-node Large-Scale EP with Distributed DP Server (`large_scale_ep.md`)**;
2. **Fine-Grained Tensor Parallelism across the all-DP axis (`Fine_grained_TP.md`)**;
3. **Dynamic Chunked Pipeline Parallelism (`CPP` / `profiling_chunk_config`) on PD-disaggregated Prefill nodes (`dynamic_chunk_pipeline_parallel.md`, `scheduler_profiling_chunk.py`)**.

## Stage 1 — Pre-flight physical link verification (`hccn_tool`) and Large-Scale EP DP server setup

1. **Verify all-card HCCS / RoCE connectivity (`docs/source/user_guide/feature_guide/large_scale_ep.md`):**
   - On Atlas A3 (`i in {0..15}`) or Atlas A2 (`i in {0..7}`), verify `hccn_tool -i $i -link -g` (`UP`), `-net_health -g` (`Success`), and cross-node ping (`-hccs_ping` on A3 SuperPoD or `-ping` on A2 RoCE).
   - Export matching `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`, and `VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1` when driving multi-node DP+EP instances via the distributed DP server.

## Stage 2 — Apply Fine-Grained TP (`finegrained_tp_config`) on all-DP (`tp_size == 1`) Decode fleets

1. **Enforce `tensor_parallel_size == 1` and `dp_size % finegrained_tp == 0` (`Fine_grained_TP.md`):**
   - `finegrained_tp_config` (`embedding_tensor_parallel_size`, `lmhead_tensor_parallel_size`, `oproj_tensor_parallel_size`, `mlp_tensor_parallel_size`) shards heavy weight matrices **along the DP dimension**, not the standard TP dimension.
   - Every configured size must be `<= data_parallel_size` and evenly divide `data_parallel_size`.
   - `oproj_tensor_parallel_size` is **only supported in Graph mode during Decode** (`FULL_DECODE_ONLY` / `PIECEWISE`) and strictly requires `tensor_parallel_size == 1` (because dummy eager runs do not trigger `o_proj` OTP reshapes and standard `tp_size > 1` conflicts with OTP weight sharding).
   - On DeepSeek-R1 W8A8 Decode (`DP32+EP32`, fine-grained TP=8), sharding `o_proj`, `lm_head`, dense `mlp`, and `embedding` saves `9.72 GB` HBM per card while `lm_head`, `mlp`, and `embedding` TP=8 each reduce TPOT by `1.0–1.2 ms`.

## Stage 3 — Enable Dynamic Chunked Pipeline Parallelism (`CPP`) on P/D Prefill (`P`) nodes

1. **Configure `profiling_chunk_config` on Prefill nodes only (`dynamic_chunk_pipeline_parallel.md`):**
   - When a Prefill instance uses Pipeline Parallelism (`--pipeline-parallel-size > 1`), enable CPP via `--additional-config '{"scheduler_config": {"profiling_chunk_config": {"enabled": true}}}'` and pass `--no-async-scheduling` (async scheduling degrades PP prefill).
   - `ProfilingChunkPredictor` (`vllm_ascend/core/profiling_chunk_predictor.py`) profiles quadratic attention vs linear layer latency during startup and dynamically sizes prefill chunks per step to minimize PP stage bubbles on long sequences (`32K–128K+`).
