---
category: sop
conditions:
  framework: vllm-ascend
  soc_family:
  - Ascend310P
domain_path: vllm-ascend/compilation/custom-ops/ascend-310p-ops
evidence_strength: hardware_spec
id: unit-vllm-ascend-310p-dedicated-ops-fp16-cast-and-nz-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-soc-generation-feature-matrix-and-abi-axioms
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/_310p/model_runner_310p.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/_310p/attention/attention_v1.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/_310p/fused_moe/fused_moe.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/_310p/quantization/modelslim_config.py
  source_digest: 336c5cde3d8bf4cdc7e801040493c597764e67d9985b87849c0bb5c7722dd577
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/_310p/model_runner_310p.py
  source_type: hardware_profile_specification
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Ascend 310P (Atlas 300I DUO) dedicated runner, FP16-only execution, NZ KV cache,
  and W8A8S/SC quantization SOP
triggers:
  error_signatures:
  - Atlas 300I DUO fails with unsupported bfloat16 dtype or OOMs on automatic max_model_len
    attention mask
  - Ascend310P MoE or linear layer fails when weight_nz_mode is disabled or Triton
    is installed
  - ACLGraph capture on 310P fails on CANN < 9.0.0 or excessive cudagraph_capture_sizes
  keywords:
  - Ascend310P
  - Atlas 300I DUO
  - NPUModelRunner310
  - model_runner_310p
  - AscendAttentionBackend310
  - w8a8s
  - w8a8sc
  - FRACTAL_NZ
---
# Ascend 310P (Atlas 300I DUO) dedicated runner, FP16-only execution, NZ KV cache, and W8A8S/SC quantization SOP

## Scope and preconditions

Use this SOP when deploying or debugging models on Atlas 300I DUO (`Ascend310P`) using the dedicated `_310p` runtime stack (`vllm_ascend/_310p/model_runner_310p.py`, `_310p/attention/attention_v1.py`, `_310p/fused_moe/fused_moe.py`, `_310p/quantization/modelslim_config.py`).

## Procedure

1. **Enforce `--dtype float16` and explicit `--max-model-len` (`vllm_ascend/_310p/model_runner_310p.py`):**
   - `Ascend310P` hardware does not support `bfloat16`; always pass `--dtype float16`.
   - Always pass a conservative `--max-model-len` (e.g., `1024`–`16384` depending on model size); leaving `--max-model-len` to auto-detect a 128K checkpoint limit can allocate an $O(L^2)$ or oversized attention workspace that OOMs the 48 GB / 24 GB per-die memory on Atlas 300I DUO.
2. **Preserve `FRACTAL_NZ` layout for KV caches and weights (`_310p/attention/attention_v1.py`):**
   - `NPUModelRunner310._check_and_update_kv_cache_config` and `AscendAttentionBackend310` allocate KV caches in 5D `FRACTAL_NZ` layout `(2, num_blocks, num_kv_heads * head_size // 16, block_size, 16)` via `torch_npu.npu_format_cast(..., ACL_FORMAT_FRACTAL_NZ)`.
   - Linear and MoE weights on 310P use `ACL_FORMAT_FRACTAL_NZ` (or `ACL_FORMAT_FRACTAL_Z` for conv3d in `qwen3vl_310.py`), and `_310p/quantization/methods/` provides dedicated `w8a8_static`, `w8a8_dynamic`, `w8a8s`, and `w8a8sc` (sparse-compressed) schemes.
3. **Remove `triton` / `triton-ascend` and bound `ACLGraph` capture sizes:**
   - Source builds on 310P must uninstall `triton` and `triton-ascend` (`pip uninstall -y triton-ascend triton`) so 310P uses its native `_310p/ops/fla/` and `aclnn` paths (`gdn_310.py`, `idex.py`, `patch_mamba_config_310.py`).
   - On CANN `< 9.0.0`, run with `--enforce-eager`. On CANN `>= 9.0.0` with `ACLGraph`, restrict `cudagraph_capture_sizes` (e.g., `[1, 4]` or `[512, 1024]`) and set `{"ascend_compilation_config": {"fuse_norm_quant": false, "enable_npugraph_ex": false}}` when hardware stream limits are reached.

## Verification

- Confirm startup logs show `NPUModelRunner310` and `AscendAttentionBackend310` initialization, `ACL_FORMAT_FRACTAL_NZ` KV cache allocation, and clean FP16 output generation.
