---
category: sop
conditions:
  component: model_runner_v2
  framework: vllm-ascend
domain_path: vllm-ascend/architecture/runner-and-worker/model-runner-v2
evidence_strength: software_contract
id: unit-vllm-ascend-model-runner-v2-persistent-batch-and-eplb-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/worker/v2/model_runner.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/worker/v2/block_table.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/worker/v2/eplb.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/model_runner_v2_eplb.md
  source_digest: c24d4a715e0ee985343a939818f35f020e3b695584da10544ea96d733aa9c7af
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/worker/v2/model_runner.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: ModelRunner V2 persistent InputBatch, UVA block tables, state managers, and
  EPLB integration SOP
triggers:
  error_signatures:
  - VLLM_ASCEND_USE_V2_MODEL_RUNNER fails on missing UVA tensor view or hybrid Mamba
    state manager
  - ModelRunner V2 speculator slot_mappings diverge from BlockTables across KV cache
    groups
  - EPLB asynchronous weight transfer in ModelRunner V2 collides with active MoE layer
    forward
  keywords:
  - VLLM_ASCEND_USE_V2_MODEL_RUNNER
  - NPUModelRunner
  - worker/v2
  - BlockTables
  - ModelState
  - MambaHybridState
  - EplbModelRunner
  - patch_uva
---
# ModelRunner V2 persistent InputBatch, UVA block tables, state managers, and EPLB integration SOP

## Scope and preconditions

Use this SOP when enabling or debugging `VLLM_ASCEND_USE_V2_MODEL_RUNNER=1` (`vllm_ascend/worker/v2/model_runner.py`), including persistent on-device batch state, Unified Virtual Addressing (UVA) emulation (`patch_uva.py`), hybrid model states (`model_states/mamba_hybrid.py`), V2 speculative decoding (`worker/v2/spec_decode/`), and V2 EPLB (`worker/v2/eplb.py`).

## Procedure

1. **Verify V2 activation and UVA host-device buffer emulation (`vllm_ascend/patch/worker/patch_v2/`):**
   - `VLLM_ASCEND_USE_V2_MODEL_RUNNER=1` activates `patch_use_v2_model_runner.py` and swaps in `vllm_ascend.worker.v2.model_runner.NPUModelRunner`.
   - Because Ascend NPU does not expose CUDA-style zero-copy UVA pointers directly to Torch kernels, `patch_uva.py` and `worker/v2/input_batch.py` synchronize CPU pinned buffers to NPU device buffers explicitly before kernel launch. Never assume a CPU write to a UVA-wrapped buffer is visible on NPU without the V2 pre-step sync hook.
2. **Use unified `BlockTables` (`vllm_ascend/worker/v2/block_table.py`) across KV cache groups and speculators:**
   - V2 centralizes per-group block tables and `slot_mappings` inside `self.block_tables` (`AscendBlockTables`).
   - V2 speculators (`eagle/speculator.py`, `dflash/speculator.py`, `dspark/speculator.py`, `mtp/speculator.py`) must consume `self.block_tables.slot_mappings` and `build_slot_mappings_by_layer` instead of maintaining shadow per-gid dictionaries.
3. **Wire model state via `ModelState` (`default.py` vs `mamba_hybrid.py`):**
   - Standard attention models use `DefaultModelState`; hybrid recurrent + attention models (e.g., Qwen3.5, Kimi K3) use `MambaHybridModelState` (`vllm_ascend/worker/v2/model_states/mamba_hybrid.py`) to manage conv/SSM state indices alongside paged KV blocks.
4. **Coordinate EPLB lifecycle through `EplbModelRunner` (`vllm_ascend/worker/v2/eplb.py`):**
   - `EplbModelRunner` wraps expert load collection, asynchronous rebalance policy execution (`EplbProcess`), and non-blocking HBM expert weight transfers (`EplbUpdator`) at the end of `execute_model`, ensuring `expert_map` updates are committed only between forward steps.

## Verification

- Run with `VLLM_ASCEND_USE_V2_MODEL_RUNNER=1` on a dense/MoE or hybrid model and confirm that `NPUModelRunner` from `vllm_ascend/worker/v2/model_runner.py` is instantiated, `ACLGraph` captures cleanly via `worker/v2/aclgraph_utils.py`, and greedy output matches V1.
