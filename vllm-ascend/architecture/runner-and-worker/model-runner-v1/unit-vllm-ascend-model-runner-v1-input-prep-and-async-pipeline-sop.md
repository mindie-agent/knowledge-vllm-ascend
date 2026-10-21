---
category: sop
conditions:
  component: model_runner_v1
  framework: vllm-ascend
domain_path: vllm-ascend/architecture/runner-and-worker/model-runner-v1
evidence_strength: software_contract
id: unit-vllm-ascend-model-runner-v1-input-prep-and-async-pipeline-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/worker/model_runner_v1.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/ModelRunner_prepare_inputs.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/worker/npu_input_batch.py
  source_digest: 41cc77d437d04fef820417bad2d870a27088f97c31cb3eb4c9b113bb0ef1581d
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/worker/model_runner_v1.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: NPUModelRunner V1 _prepare_inputs, CpuGpuBuffer async H2D copy, and Sequence
  Parallel padding SOP
triggers:
  error_signatures:
  - _prepare_inputs host preparation dominates step latency or triggers unexpected
    stream sync
  - ACLGraph replay reads unpadded or misaligned input_ids when Sequence Parallelism
    is enabled
  - speculative token rejection count mismatches query_start_loc during async scheduling
  keywords:
  - NPUModelRunner
  - model_runner_v1
  - _prepare_inputs
  - CpuGpuBuffer
  - query_start_loc
  - seq_lens
  - slot_mapping
  - pad_size
  - async_scheduling
---
# NPUModelRunner V1 _prepare_inputs, CpuGpuBuffer async H2D copy, and Sequence Parallel padding SOP

## Scope and preconditions

Use this SOP when profiling, debugging, or modifying `NPUModelRunner` (`vllm_ascend/worker/model_runner_v1.py`) input preparation (`_prepare_inputs`, `_calc_spec_decode_metadata`,attention metadata construction, and TP/SP/graph padding).

## Procedure

1. **Audit dual-view `CpuGpuBuffer` allocations and H2D copies:**
   - `NPUModelRunner` pre-allocates paired pinned-CPU (`pin_memory=True`) and persistent NPU tensors via `CpuGpuBuffer` (`input_ids`, `positions`, `query_start_loc`, `seq_lens`, `slot_mapping`).
   - During `_prepare_inputs`, populate `.np` / `.cpu` slices in place and invoke `.copy_to_gpu(n)` (`non_blocking=True`) onto the current NPU compute stream. Never allocate fresh `torch.tensor(..., device="npu")` inside `_prepare_inputs`, as new device addresses break `ACLGraph` static buffer bindings.
2. **Maintain request ordering and decode/prefill split invariants:**
   - `NPUInputBatch` (`vllm_ascend/worker/npu_input_batch.py`) orders decode requests (`num_scheduled_tokens == 1 + num_spec_tokens`) before prefill/chunked-prefill requests, or groups them according to the active attention backend's `split_decodes_and_prefills` contract.
   - Verify that `query_start_loc` (`shape=[num_reqs + 1]`, starting at `0`) and `seq_lens` (`shape=[num_reqs]`) remain monotonically aligned with `slot_mapping` (`shape=[num_tokens]`, padded with `-1` for dummy tokens).
3. **Compute Sequence Parallel (SP / FlashComm) and `ACLGraph` padding consistently:**
   - When TP sequence parallelism (`enable_sp` / `flashcomm1`) is active, total token count `num_tokens_padded` must be rounded up to a multiple of `tp_size` (and aligned to the selected `cudagraph_capture_sizes` bucket when `ACLGraph` is enabled).
   - Ensure `mc2_mask` and `padded_num_tokens_across_dp` reflect the exact padded token count across all DP ranks before entering collective dispatch (`AllGather` / `ReduceScatter` / `AllToAll`).
4. **Handle async scheduling and speculative decoding (`num_rejected_tokens_gpu`) without host sync:**
   - When async scheduling is enabled with speculative decoding (MTP / Eagle / DSpark / DFlash), token acceptance from step $t-1$ lives on device (`num_accepted_tokens` / `num_rejected_tokens`). Update `input_ids` and `positions` on device via Triton/NPU pre-step kernels rather than calling `.item()` or `.cpu()` in `_prepare_inputs`.

## Verification

- Run a mixed batch (decode + chunked prefill) under `VLLM_LOGGING_LEVEL=DEBUG` and verify that `input_ids.data_ptr()`, `positions.data_ptr()`, and `slot_mapping.data_ptr()` remain constant across consecutive decode graph replays.
- Confirm zero host synchronization calls (`aclrtSynchronizeStream` / `.item()`) inside `_prepare_inputs` during steady-state decode profiling.
