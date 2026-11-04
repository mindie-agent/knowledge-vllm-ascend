---
category: workflow
conditions:
  framework:
  - vllm-ascend
  - any
domain_path: vllm-ascend/ops/triton
evidence_strength: reproduced_empirical
id: unit-triton-ascend-tiling-persistent-grid-and-ub-budget-workflow
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/techniques/tiling_and_grid_910_95.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/optimizations/techniques/tiling_and_grid_910_95.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: canonical_engineering_guide
  supersedes: []
revision: 3
schema: mindie-graded-unit/1
tier: L3
title: Triton-Ascend Tiling Geometry, Persistent Grid Scheduling, and UB Budget Fitting
  Workflow (`spatial_group_norm_reduce.py`)
triggers:
  error_signatures:
  - Numerical check failed on shape (2, 16, 16, 16) max_abs_err=0.046875 > 5e-3
  - 1. Platform-Specific UB Capacity & Alignment Accounting (`OPT_28`)
  - 2. Engine-Aware Grid Scheduling & Persistent Core Clamping (`OPT_29`, `OPT_41`)
  keywords:
  - unit-triton-ascend-tiling-persistent-grid-and-ub-budget-workflow
  - spatial_group_norm_reduce
  - spatial_group_norm_reduce.py
  - GroupNorm
  - max_abs_err
  - BLOCK_SIZE
  - multibuffer
---
# Triton-Ascend Tiling Geometry, Persistent Grid Scheduling, and UB Budget Fitting Workflow (`spatial_group_norm_reduce.py`)

## Stage 1 — Clamp 1D Grid to Physical VectorCore Count (`grid = (min(total_tiles, NUM_CORES),)`)

1. Select `NUM_CORES = 48` and clamp `grid = (min(total_tiles, NUM_CORES),)` via `compute_persistent_grid(total_tiles)`.

## Stage 2 — Multi-Row Spatial Tiling (`BLOCK_ROWS = 4`) along `H` with Fixed `BLOCK_SIZE = 512`

1. Process slices along `H` in chunks of `4` rows (`for h_start in range(0, height, 4)`) with fixed `BLOCK_SIZE = 512`.

## Stage 3 — Enable `multibuffer=True` Double-Buffering on Tiled Reductions

1. Keep `enable_multibuffer = True` and accumulate row tiles across `(c, h)` slices.
