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
revision: 4
schema: mindie-graded-unit/1
tier: L2
title: ACLGraph capture invalidation, dynamic shape bucketing, and stream sync triage SOP
triggers:
  error_signatures:
  - ACLGraph replay falls back to eager execution on decode steps
  - RuntimeError during npu_graph capture due to host-device synchronization or dynamic allocation
  - decode output diverges after batch size changes across padded graph buckets
  - eager execution passes on the same decode input while aclgraph or piecewise replay diverges
  - NPUGraph replay corrupts state slot 0 or reads stale warmup buffers on padded decode rows
  - AutoRegressiveAclGraphManager _updatable_graph_replay or PCP draft block_table diverges on step > 0
  - RoutedExpertsCapturer returns zeros or stale warmup expert IDs in upper batch rows under TP > 1
  keywords:
  - ACLGraph
  - mindie-debugger
  - debugger aclgraph
  - debugger precision
  - ACLGraphVllmInspector
  - ACLGraphAddressTracker
  - ACLGraphDumper
  - MemoryWatchpoint
  - StorageGraph
  - LockstepDiffer
  - graph capture
  - invalidation
  - padded batch
  - PAD_SLOT_ID
  - null_block_id
  - causal_conv1d
  - conv_state
  - AutoRegressiveAclGraphManager
  - _updatable_graph_replay
  - build_fia_params
  - RoutedExpertsCapturer
  - patch_routed_experts_capture
  - bucket
  - eager fallback
  - stream synchronization
  - persistent buffer
  - model_runner_v1
  - stale tail padding
---

# ACLGraph capture invalidation, dynamic shape bucketing, and stream sync triage SOP

## 1. Applicable Problem & When to Select

Select this SOP when `vllm-ascend` decode steps under `aclgraph`, `piecewise`, or `full_decode_only` mode exhibit graph-specific failure while the underlying operator math is valid in eager mode:
- **Observable Triggers**:
  - Eager execution (`--enforce-eager`) produces correct reference outputs on the identical decode step, whereas `aclgraph` / `piecewise` / `NPUGraph` replay diverges numerically, returns stale warmup buffers, or corrupts subsequent steps;
  - `npu_graph` capture raises `RuntimeError` due to host scalar extraction (`.item()`, `.cpu()`) or stream switching inside custom operators;
  - Decode steps unexpectedly fall back to eager execution or repeatedly re-capture across padded batch buckets (`B_large -> B_small`).
- **Automated Inspection via `debugger` MCP Tool (`mindie-debugger`)**:
  - Whenever an `ACLGraph` / `NPUGraph` / `UpdatableGraph` capture or replay bug, persistent-buffer stomp, or eager-vs-graph divergence is suspected, invoke the `debugger` MCP tool (`mindie-debugger`) to inspect pointer stability, padding sentinel invariants, or operator storage mutations:
    ```text
    # 1. Inspect / guard a reproduction or test script for ACLGraph pointer & stream violations:
    debugger(argv=["aclgraph", "guard", "--script", "<repro_or_test.py>"])
    debugger(argv=["aclgraph", "inspect", "--script", "<repro_or_test.py>", "--dump-dir", "<dump_dir>"])

    # 2. Check a suspect operator/function for storage aliasing, in-place mutation, or reference drift:
    debugger(argv=["precision", "check-call", "--module", "<pkg.mod>", "--func", "<fn>", "--script", "<repro.py>"])
    ```
  - For online `vllm-ascend` graph debugging in Python, `from mindie_debugger import ACLGraphVllmInspector, ACLGraphAddressTracker, ACLGraphDumper, StorageGraph, MemoryWatchpoint, LockstepDiffer` provides:
    - `ACLGraphVllmInspector`: hooks `vllm_ascend.compilation.acl_graph.ACLGraphWrapper` and `update_full_graph_params` (deferring `FULL`-graph eager comparison until after `torch.npu.ExternalEvent`s are recorded on `update_stream` and temporarily clearing `forward_context.capturing` / `_EXTRA_CTX.capturing` / `cudagraph_runtime_mode = CUDAGraphMode.NONE`), tracks `attn_metadata` (`slot_mapping`, `seq_lens`, `query_start_loc`, `block_table`, `state_indices`, `conv_state_indices`, `ssm_state_indices`), and flags `DYNAMIC_ALLOC_IGNORED_BY_GRAPH` when an operator returns a freshly allocated tensor instead of writing into the static graph output buffer.
    - `MemoryWatchpoint` & `StorageGraph`: snapshot `(storage_ptr, nbytes, checksum)` before and after a region to pinpoint unintended writes to `conv_state[0]`, `ssm_state[0]`, or partial slices of `device_buffer`.
- **Missing Facts & Minimal Probe (`condition_unknown`)**:
  - If `graph_mode` or eager baseline parity is not yet recorded, run a single-step comparison on the same input with `--enforce-eager` vs the active `aclgraph` configuration and log `num_scheduled_tokens`, `padded_num_tokens`, and `uniform_decode` before adopting this SOP.
- **When NOT to Adopt (Switch / Default Boundary)**:
  - Do **not** adopt if `--enforce-eager` already fails or diverges on the same single step (route to single-operator numerical or dtype overflow triage instead).
  - Do **not** adopt if multi-rank execution hangs in HCCL collective initialization before entering `model_runner_v1` graph capture (route to HCCL / DP collective triage).

## 2. Next Discriminating Action & Branch Routing

1. **Step 1 — Verify Bucket Coverage vs Scheduled Token Count**
   - Record `num_scheduled_tokens`, `num_reqs`, and `padded_num_tokens` at the step boundary in `model_runner_v1.py`.
   - Confirm whether `uniform_decode` holds (`max_query_len == 1` across active requests) and whether `padded_num_tokens` matches a pre-captured graph key.
   - **Branch A (Mixed prefill-decode chunk enters `full_decode_only` replay)**: Bypass full-graph replay for non-uniform batches rather than writing variable-length tokens into fixed-shape capture buffers.

2. **Step 2 — Isolate Host Synchronization & Dynamic Buffer Allocation Inside Capture (`ACLGraphAddressTracker`)**
   - Run one warm capture step with `ASCEND_LAUNCH_BLOCKING=0` or `debugger(argv=["aclgraph", "guard", ...])` while tracing host-device sync calls (`torch.npu.synchronize()`, `.item()`, `.to("cpu")`, or dynamic `torch.empty` allocations inside the captured region).
   - **Branch B (Host scalar or fresh tensor allocation inside captured region)**: Move scalar precomputation to host metadata preparation and copy all input metadata tensors (`positions`, `slot_mapping`, `block_table`, `seq_lens`, `query_start_loc`) into persistent static device buffers via in-place `.copy_()` on the capture stream before replay. Ensure custom ops executed inside `NPUGraph` write in-place into a pre-allocated `out=` buffer rather than returning a fresh tensor on each call.

3. **Step 3 — Audit Padded Tail Sentinels (`PAD_SLOT_ID = -1`), Multi-Step `UpdatableGraph` Refresh, and TP-Sharded Capture Buffers**
   - Run a two-step bucket transition probe (`B_large -> B_small`) or padded decode step (`live < B_padded`) and check the four recurring `vllm-ascend` graph-replay failure modes:
   - **Branch C1 (Stale tail metadata across `B_large -> B_small`)**: Padded tail slots `[num_actual_tokens : padded_num_tokens]` retained stale metadata from Step A. Enforce `slot_mapping[num_actual_tokens:padded_num_tokens] = -1` and zero out padded `seq_lens` / `query_start_loc` deltas before replay.
   - **Branch C2 (Negative `PAD_SLOT_ID = -1` sentinel disables `fla_npu` null-block filtering in `causal_conv1d` / `conv_state`)**: In `fla_npu`'s AscendC `causal_conv1d_update` operator, passing a negative `null_block_id` (`-1` / `PAD_SLOT_ID`) disables null-block filtering so inactive rows with index `-1` underflow `kernel_state[-1]` (`507011` or reading the last packed state row), whereas clamping `-1 -> 0` overwrites request `0`'s state. When `needs_state_copy` packs `requests` rows in `causal_conv1d.py` and `conv_state.py`, allocate one extra dedicated non-negative null slot at the end (`kernel_state = torch.empty((requests + 1, state_len, dim), ...)`, `null_block_id = requests`), map inactive requests in `_copy_conv_state_kernel` to `REQUESTS` (`tl.where(active, request, REQUESTS)`), and pass `out=output` to `causal_conv1d_update` so skipped rows remain zero and output storage addresses stay fixed across `NPUGraph` replay.
   - **Branch C3 (ModelRunner V2 `AutoRegressiveAclGraphManager._updatable_graph_replay` & PCP draft `block_table` rebuild)**: In multi-step speculative decoding (`vllm_ascend/worker/v2/spec_decode/autoregressive/aclgraph.py`, `speculator.py`), replaying an `UpdatableGraph` (`_updatable_graph_replay`) without calling `self.speculator.build_draft_attn_metadatas(num_reqs, desc.num_tokens, self.is_draft_model_prefill)` before `build_fia_params` / `resolve_tasks` reuses stale target or step-0 attention parameters. Furthermore, when `pcp_size > 1`, `build_fia_params` in `speculator.py` must reconstruct `block_table` from `self.runner.block_tables.slot_mappings[attn_group_id]` via `torch.index_select(..., dim=0, index=block_table_indices).view(num_reqs, max_blocks) // block_size` instead of reading `self.runner.block_tables.input_block_tables[0]` (which is not rebuilt under PCP).
   - **Branch C4 (Partial `device_buffer` write under TP-sharded `ALLTOALL`/`MC2` in `RoutedExpertsCapturer`)**: When capturing routed experts (`vllm_ascend/patch/worker/patch_routed_experts_capture.py`) into a static `self.device_buffer` under `DP == 1` and `TP > 1` with `TokenDispatcherWithAll2AllV` or `TokenDispatcherWithMC2` (`ALLTOALL`, `MC2`, `FUSED_MC2`), `topk_ids` entering `capture(layer_id, topk_ids)` is TP-sharded (`batch_size // tp_size` rows). Copying `topk_ids` directly into `self.device_buffer[:batch_size, layer_id, :]` only updates the first `batch_size // tp_size` rows and leaves the upper rows `[batch_size // tp_size : batch_size]` containing zeros or stale warmup batch expert IDs. `all_gather` `topk_ids` across the TP group (`get_tp_group().all_gather(topk_ids, 0)`) and slice to `num_tokens` (`get_forward_context().num_tokens`) before writing to `self.device_buffer`.

## 3. Verification, Exit Criteria & Scoped Counter-Evidence

- **Verification & Exit**: Exit with `verified` only when (a) eager and `aclgraph` replay match within pre-registered tolerance across `B_large -> B_small -> B_large` transitions, (b) zero unplanned graph re-captures occur during steady-state decode, and (c) uninstrumented end-to-end decode throughput does not regress.
- **Scoped Counter-Evidence**: If resetting padded tail metadata (`slot_mapping = -1`, `seq_lens = 0`) and confirming in-place `.copy_()` on persistent buffers does not eliminate the `B_large -> B_small` divergence under `Ascend910B` / `Ascend910_93`, record a conditional counter-evidence finding against graph metadata buffering and switch to operator-level internal workspace / recurrent state life-cycle inspection.
