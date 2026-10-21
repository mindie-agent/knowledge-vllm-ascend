---
category: sop
conditions:
  component: patch
  framework: vllm-ascend
domain_path: vllm-ascend/architecture/platform-and-patch/monkey-patch-hierarchy
evidence_strength: software_contract
id: unit-vllm-ascend-patch-hierarchy-and-upstream-migration-sop
inherits_from:
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/patch/__init__.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/patch/platform/__init__.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/patch/worker/__init__.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/patch.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/balance_schedule_refactor.md
  source_digest: 05d1b94f3753409c0fcb8eb0c8be503412c69109a4968c12dd4c407b52c91e7c
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/patch/__init__.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: vLLM-Ascend two-stage monkey-patch hierarchy (platform vs worker) and upstream
  drift guard SOP
triggers:
  error_signatures:
  - worker process ignores a platform patch under spawn multiprocessing
  - patch_balance_schedule or patch_distributed raises AttributeError or TypeError
    after upstream vLLM bump
  - unconditional monkey patch breaks PD-disaggregated RecomputeScheduler or non-enabled
    features
  keywords:
  - vllm_ascend/patch
  - patch/platform
  - patch/worker
  - NPUPlatform
  - patch_balance_schedule
  - vllm_version_is
  - monkey patch
---
# vLLM-Ascend two-stage monkey-patch hierarchy (platform vs worker) and upstream drift guard SOP

## Scope and preconditions

Use this SOP when adding, auditing, or refactoring runtime patches in `vllm_ascend/patch/` (`patch/platform/` and `patch/worker/`) or diagnosing silent patch drift across upstream `vllm` revisions.

## Procedure

1. **Respect the two-stage patch lifecycle (`docs/source/developer_guide/Design_Documents/patch.md`):**
   - **Platform patches (`vllm_ascend/patch/platform/`):** Applied when `NPUPlatform.__init__()` runs in the API server / EngineCore process. Place scheduler, config, distributed group coordinator, renderer/parser, and EngineCore patches here (`patch_balance_schedule.py`, `patch_dyntra_lb_core.py`, `patch_kv_cache_coordinator.py`, `patch_mamba_config.py`, `patch_multiproc_executor.py`).
   - **Worker patches (`vllm_ascend/patch/worker/` and `patch_v2/`):** Applied inside `NPUWorker.__init__()` in each spawned/forked NPU worker process. Place model-executor, custom-op, Triton, CUDA-graph/ACLGraph, and weight-loader hooks here (`patch_cudagraph.py`, `patch_fused_moe.py`, `patch_rejection_sampler.py`, `patch_qwen3_5.py`). Never assume a platform-only patch is active in a `spawn` worker unless imported by the worker path.
2. **Gate intrusive class swaps on explicit feature activation:**
   - Follow the `patch_balance_schedule.py` contract (`balance_schedule_refactor.md`): never unconditionally replace `DPEngineCoreProc` or `Scheduler` behaviors when the feature is disabled. Delegate the disabled path directly to `super()` so orthogonal features (`AsyncRecomputeScheduler`, `DyntraLBScheduler`) remain untouched.
3. **Avoid full-function copy-paste and guard upstream version transitions:**
   - Hook narrow lifecycle seams (such as `_has_global_unfinished_reqs`) instead of copying 500-line upstream methods.
   - When supporting both a pinned release tag (`.github/vllm-release-tag.commit`) and upstream main (`.github/vllm-main-verified.commit`), use `vllm_version_is(...)` (`vllm_ascend/utils.py`) and document the exact upstream PR that will retire the patch.

## Verification

- Verify that importing `vllm_ascend` applies `patch/platform` cleanly, worker initialization applies `patch/worker`, and the patched symbol's call signature matches the installed upstream vLLM symbol.
