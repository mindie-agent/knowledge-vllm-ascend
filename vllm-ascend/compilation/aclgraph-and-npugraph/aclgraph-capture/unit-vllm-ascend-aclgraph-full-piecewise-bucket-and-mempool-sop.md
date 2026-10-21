---
category: sop
conditions:
  component: aclgraph
  framework: vllm-ascend
domain_path: vllm-ascend/compilation/aclgraph-and-npugraph/aclgraph-capture
evidence_strength: software_contract
id: unit-vllm-ascend-aclgraph-full-piecewise-bucket-and-mempool-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/compilation/acl_graph.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/ACL_Graph.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/graph_mode.md
  source_digest: b13492d8f117f7c7bcf3721f849f273c7c3ea890ae4cd022640b8280af9538a2
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/compilation/acl_graph.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: ACLGraph PIECEWISE vs FULL_DECODE_ONLY capture buckets, shared mempool, and
  workspace sizing SOP
triggers:
  error_signatures:
  - ACLGraph capture runs out of NPU hardware streams or triggers OOM during warmup
  - FULL_DECODE_ONLY graph replay crashes when attention metadata is not updated via
    update_attn_params
  - capturing ACLGraph with speculative decoding exceeds max_capture_size
  keywords:
  - ACLGraphWrapper
  - acl_graph
  - FULL_DECODE_ONLY
  - PIECEWISE
  - cudagraph_capture_sizes
  - update_attn_params
  - update_mla_attn_params
  - graph_pool_handle
---
# ACLGraph PIECEWISE vs FULL_DECODE_ONLY capture buckets, shared mempool, and workspace sizing SOP

## Scope and preconditions

Use this SOP when configuring, tuning, or debugging `ACLGraph` execution (`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` or `"PIECEWISE"`, `vllm_ascend/compilation/acl_graph.py`) on Ascend NPUs.

## Procedure

1. **Choose `PIECEWISE` vs `FULL_DECODE_ONLY` (`docs/source/developer_guide/Design_Documents/ACL_Graph.md`):**
   - **`PIECEWISE`:** Splits the TorchFX graph at `splitting_ops` (attention kernels) so attention runs eagerly on the host while linear/MoE/norm subgraphs run as captured `torch.npu.NPUGraph` segments. Works for both prefill and decode, and requires no in-graph attention parameter mutation.
   - **`FULL_DECODE_ONLY`:** Captures the entire decode forward pass (including attention) into a single `NPUGraph` per batch bucket. Before `graph.replay()`, `ACLGraphWrapper` invokes `update_attn_params` / `update_mla_attn_params` / `update_dsa_attn_params` / `update_sfa_attn_params` on the capture stream to refresh host-computed tiling/sequence lengths inside the captured task descriptors.
2. **Bound `cudagraph_capture_sizes` against NPU stream limits (`vllm_ascend/platform.py`):**
   - Each captured graph consumes NPU hardware streams proportional to the number of multi-stream switches inside the model (`multistream_overlap_shared_expert`, EP dispatch/combine).
   - `platform.py` caps `max_capture_size` based on the SoC stream budget (`1800` streams on A2/A3, stricter on 310P). When `num_speculative_tokens = K > 0`, every request contributes `1 + K` tokens, so capture buckets must be multiples of `1 + K` (`[1+K, 2*(1+K), ...]`).
3. **Share the NPU graph memory pool (`graph_pool_handle`):**
   - All captured bucket sizes share a single `torch.npu.graph_pool_handle()` so intermediate activation buffers are reused across mutually exclusive batch sizes. Never allocate separate pool handles per bucket size, or warmup will exhaust HBM.

## Verification

- Confirm in startup logs that `Graph capturing finished in ... secs, took ... GiB` completes for all configured `cudagraph_capture_sizes` without stream-exhaustion fallback, and verify zero graph recaptures during steady-state decode.
