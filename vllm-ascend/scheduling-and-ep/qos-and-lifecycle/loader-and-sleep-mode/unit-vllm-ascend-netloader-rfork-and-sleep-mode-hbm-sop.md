---
category: sop
conditions:
  component: netloader_rfork_and_sleep_mode
  framework: vllm-ascend
domain_path: vllm-ascend/scheduling-and-ep/qos-and-lifecycle/loader-and-sleep-mode
evidence_strength: software_contract
id: unit-vllm-ascend-netloader-rfork-and-sleep-mode-hbm-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/device_allocator/sleep_mem_optimized.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/device_allocator/camem.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/model_loader/netloader/netloader.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/model_loader/rfork/rfork_loader.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/netloader.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/rfork.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/sleep_mode.md
  source_digest: 6d52a81d3caf734b132bed63ff8bf79cbef9cc886d31e1cf81ef31da878afa0f
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/device_allocator/sleep_mem_optimized.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Elastic P2P weight loading (Netloader, RFork) and RL Sleep Mode (Level 1/2
  + extra cleanup) SOP
triggers:
  error_signatures:
  - Netloader MTP draft server port overflows 65535 when LISTEN_PORT is set above
    55535
  - RFork receiver loads stale quantized layout or silently bypasses transfer when
    dynamic_eplb is enabled
  - Sleep Mode wake_up fails to recapture ACLGraph or leaves HCCL process groups consuming
    NPU HBM during RL training
  keywords:
  - netloader
  - rfork
  - YuanRong TransferEngine
  - enable_sleep_mode
  - sleep_mode_extra_cleanup
  - sleep_mem_optimized
  - camem
  - reload_weights
---
# Elastic P2P weight loading (Netloader, RFork) and RL Sleep Mode (Level 1/2 + extra cleanup) SOP

## Scope and preconditions

Use this SOP when accelerating elastic instance startup via P2P NPU weight transfer (`--load-format netloader` or `--load-format rfork` in `vllm_ascend/model_loader/`) or managing NPU HBM lifecycle during RL post-training (PPO/GRPO/DPO) via `--enable-sleep-mode` (`docs/source/user_guide/feature_guide/sleep_mode.md`, `vllm_ascend/device_allocator/`).

## Procedure

1. **Configure `Netloader` HCCL P2P weight streaming (`vllm_ascend/model_loader/netloader/`, `netloader.md`):**
   - Pass `--load-format netloader` with `VLLM_SLEEP_WHEN_IDLE=1`. Reserve ~200 MB HBM for `HCCL_BUFFERSIZE`.
   - Each rank listens on `LISTEN_PORT + RANK`; for MTP/draft models, the draft server listens on `LISTEN_PORT + RANK + 10000`, so keep `LISTEN_PORT < 55535` to avoid port overflow above `65535`.
   - For quantized models with int8 weights, set `"INT8_CACHE": "dram"` (or `"hbm"`) in `--model-loader-extra-config` so pre-processed int8 buffers are cached for P2P send.
2. **Configure `RFork` planner-coordinated warm start (`vllm_ascend/model_loader/rfork/`, `rfork.md`):**
   - Requires `openyuanrong-transfer-engine` and an RFork planner (`rfork_scheduler_url`).
   - RFork builds a topology-aware seed key (`model_url`, `model_deploy_strategy_name`, `kv_role`, `node_rank`, `pp_rank`, `tp_rank`, `ep_rank`, `draft`). For quantized models, RFork transfers live NPU tensors **after** Ascend post-processing (`NZ` format, transposed/packed weights, `W_UV`/`W_UK_T`). Always bump `model_deploy_strategy_name` after changing model flags, and note that `dynamic_eplb` automatically bypasses RFork.
3. **Operate RL Sleep Mode (`level=1` vs `level=2`) and `rl_config.sleep_mode_extra_cleanup` (`sleep_mode.md`):**
   - Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` and `VLLM_ASCEND_ENABLE_NZ=0` when using `camem` / `sleep_mem_optimized` allocators.
   - **Level 1 (`llm.sleep(level=1)`):** Offloads `"weights"` to CPU DRAM and discards `"kv_cache"`.
   - **Level 2 (`llm.sleep(level=2)`):** Discards both `"weights"` and `"kv_cache"`. Wake up in two phases: `llm.wake_up(tags=["weights"])` -> `reload_weights` -> `llm.wake_up(tags=["kv_cache"])`.
   - **Extra cleanup (`"rl_config": {"enabled": true, "sleep_mode_extra_cleanup": true}`):** Also clears ACLGraph workspaces, resets the graph manager, and destroys HCCL process groups on `sleep()`, then restores HCCL groups, refreshes MoE dispatcher metadata, and recaptures ACLGraphs on `wake_up(tags=["kv_cache"])`.

## Verification

- Verify after `llm.sleep(level=1)` that `torch.npu.mem_get_info()` shows NPU memory consumption drop below `1 GiB` (even lower with `sleep_mode_extra_cleanup`), and that `llm.wake_up()` restores bit-identical greedy output.
