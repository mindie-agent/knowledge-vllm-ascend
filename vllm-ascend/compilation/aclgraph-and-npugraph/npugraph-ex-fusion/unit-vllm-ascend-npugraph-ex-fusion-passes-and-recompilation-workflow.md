---
category: workflow
conditions:
  component: npugraph_ex
  framework: vllm-ascend
domain_path: vllm-ascend/compilation/aclgraph-and-npugraph/npugraph-ex-fusion
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-npugraph-ex-fusion-passes-and-recompilation-workflow
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-aclgraph-full-piecewise-bucket-and-mempool-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/compilation/compiler_interface.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/compilation/graph_fusion_pass_manager.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/compilation/passes/norm_quant_fusion_pass.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/compilation/passes/qknorm_rope_fusion_pass.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/npugraph_ex.md
  source_digest: e4ca0c26581f49751cd0a265c4f005818450bd5b317b8f414eb3fca9812b3a6b
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/compilation/compiler_interface.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: npugraph_ex TorchAir FX graph fusion passes (AddRMSNormQuant, QKNormRoPE) and
  static-kernel workflow
triggers:
  error_signatures:
  - enable_npugraph_ex triggers repeated FX graph recompilation or fails pattern matching
    on custom ops
  - fuse_norm_quant or fuse_qknorm_rope causes precision mismatch on 310P or multimodal
    vision layers
  - enable_static_kernel compilation hangs or exhausts disk cache across multi-rank
    startup
  keywords:
  - enable_npugraph_ex
  - npugraph_ex
  - AscendCompiler
  - GraphFusionPassManager
  - NormQuantFusionPass
  - QKNormRopeFusionPass
  - AllGatherChunkNoopPass
  - enable_static_kernel
---
# npugraph_ex TorchAir FX graph fusion passes (AddRMSNormQuant, QKNormRoPE) and static-kernel workflow

## Trigger and scope

Use this workflow when enabling or debugging `ascend_compilation_config` (`enable_npugraph_ex`, `fuse_norm_quant`, `fuse_qknorm_rope`, `fuse_allgather_slice`, `fuse_muls_add`, `enable_static_kernel`) in `vllm_ascend/compilation/` (`compiler_interface.py`, `graph_fusion_pass_manager.py`, `passes/`).

## Stage 1 — Configure `AscendCompiler` and TorchAir `npugraph_ex` backend

1. **Understand the `npugraph_ex` compilation pipeline (`docs/source/developer_guide/Design_Documents/npugraph_ex.md`):**
   - When `additional_config.ascend_compilation_config.enable_npugraph_ex = true` (default on supported A2/A3 paths), `AscendCompiler` (`vllm_ascend/compilation/compiler_interface.py`) invokes `GraphFusionPassManager` on each piecewise FX subgraph and lowers it through `torch_npu.npu.npugraph_ex.compile_fx`.
   - `npugraph_ex` requires a valid meta/fake implementation (`meta_registration.py`) for every `torch.ops._C_ascend` and `torch.ops.vllm` node inside the traced subgraph.
2. **Audit built-in FX fusion passes (`vllm_ascend/compilation/passes/`):**
   - `NormQuantFusionPass` (`fuse_norm_quant`): Fuses `AddRMSNorm` + dynamic/static quantization (`npu_add_rms_norm_quant`), eliminating intermediate FP16/BF16 global memory round-trips.
   - `QKNormRopeFusionPass` (`fuse_qknorm_rope`): Fuses per-head `RMSNorm(q)`, `RMSNorm(k)`, and RoPE into `npu_qk_rms_norm_rope`.
   - `AllGatherChunkNoopPass` (`fuse_allgather_slice`) and `MulsAddPass` (`fuse_muls_add`): Eliminate redundant slice/copy views after TP/SP collectives.

## Stage 2 — Isolate pass-level numerical regressions or unsupported SoCs

1. **Selectively disable individual fusion passes for bisection:**
   - On `Ascend310P` or models with non-standard RoPE/norm layouts (e.g., `DeepSeek-OCR-2`, `Qwen3-ASR-1.7B`, `MiniMax-M3`), disable the specific pass via `--additional-config`:
     ```json
     {
       "ascend_compilation_config": {
         "enable_npugraph_ex": true,
         "fuse_norm_quant": false,
         "fuse_qknorm_rope": false
       }
     }
     ```
2. **Triage `enable_static_kernel` binary compilation:**
   - `enable_static_kernel: true` compiles static-shape AICore binaries for the captured bucket shapes. Keep `enable_static_kernel: false` during rapid bring-up or when dynamic bucket shapes change frequently, and enable it only on frozen production topologies.

## Stage 3 — Verify fusion pass application and zero recompilation

1. **Confirm pass hit counts and graph stability:**
   - Enable `VLLM_LOGGING_LEVEL=DEBUG` during warmup and verify that `NormQuantFusionPass` and `QKNormRopeFusionPass` replace the expected number of layer nodes, with zero FX recompilations during online serving.
