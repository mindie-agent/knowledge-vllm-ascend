---
category: sop
conditions:
  graph_mode:
  - aclgraph
  - piecewise
  - full_decode_only
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
domain_path: vllm-ascend/runtime/graph-capture
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py
  source_digest: 61ee8bc56d4ac0e1a7f43d1804bfe41c1a6699181774938907c97174f2348343
  source_locator: vllm-project/vllm-ascend@d8c9e3a673af918c94a7794ef6f1f69c024d9c62:vllm_ascend/worker/model_runner_v1.py
  source_type: runtime_triage_sop
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: ACLGraph capture invalidation, dynamic shape bucketing, and stream sync triage SOP
triggers:
  error_signatures:
  - ACLGraph replay falls back to eager execution on decode steps
  - RuntimeError during npu_graph capture due to host-device synchronization or dynamic allocation
  - decode output diverges after batch size changes across padded graph buckets
  keywords:
  - ACLGraph
  - graph capture
  - invalidation
  - padded batch
  - bucket
  - eager fallback
  - stream synchronization
  - persistent buffer
  - model_runner_v1
---

# ACLGraph capture invalidation, dynamic shape bucketing, and stream sync triage SOP

## Applicability & Symptom Scope

Applies when `vllm-ascend` decode steps running under `aclgraph` / `piecewise` / `full_decode_only` mode exhibit:
- unexpected fallback to eager execution or repeated graph re-capture across decode steps,
- capture failure caused by host scalar extraction (`.item()`, `.cpu()`) or stream-switching inside custom operators, or
- numerical divergence when transitioning from a larger padded batch bucket to a smaller decode batch.

## Step-by-Step Diagnostic Procedure

1. **Verify Bucket Coverage vs Scheduled Token Count**
   - Record the scheduled `num_scheduled_tokens`, `num_reqs`, and `padded_num_tokens` at the step boundary in `model_runner_v1.py`.
   - Confirm whether `uniform_decode` holds (`max_query_len == 1` across active requests) and whether `padded_num_tokens` matches a pre-captured graph key. If mixed prefill-decode chunks enter a `full_decode_only` graph path, the runner must bypass full-graph replay rather than corrupting fixed-shape input buffers.

2. **Isolate Host Synchronization & Dynamic Buffer Allocation Inside Capture**
   - Run a single warm capture step with `ASCEND_LAUNCH_BLOCKING=0` while tracing host-device sync calls (`torch.npu.synchronize()`, `.item()`, `.to("cpu")`, or dynamic `torch.empty` allocations inside the captured region).
   - Verify that all input metadata tensors (`positions`, `slot_mapping`, `block_table`, `seq_lens`, `query_start_loc`) are copied into persistent static device buffers via in-place `.copy_()` on the capture stream before graph replay.

3. **Check Stale Tail Padding Across Padded Buckets**
   - Run a two-step transition test: Step A at bucket size `B_large` followed by Step B at `B_small < B_large` reusing the same persistent buffer.
   - Verify that padded tail slots `[num_actual_tokens : padded_num_tokens]` have `slot_mapping = -1` (or masked attention lengths `seq_lens = 0`) so padded tokens never overwrite KV cache blocks or recurrent state slots belonging to active requests.
