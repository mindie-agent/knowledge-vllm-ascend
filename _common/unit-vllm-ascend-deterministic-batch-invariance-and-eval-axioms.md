---
category: measurement
conditions:
  framework: vllm-ascend
domain_path: _common
evidence_strength: software_contract
id: unit-vllm-ascend-deterministic-batch-invariance-and-eval-axioms
inherits_from:
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/batch_invariant.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/batch_invariance.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/evaluation/using_ais_bench.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/evaluation/using_evalscope.md
  source_digest: 1572ab210ce06678f5288eda812eebc5882243375e6a48b47eb9a4ff3c79629a
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/batch_invariant.py
  source_type: verification_specification
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Deterministic batch invariance, HCCL deterministic reduction, and AISBench/EvalScope
  evaluation gates
triggers:
  error_signatures:
  - identical request produces different logits or tokens when batch size or concurrent
    request mix changes
  - AISBench or EvalScope accuracy fluctuates across runs at temperature 0
  - flashcomm or fused MC2 all-to-all introduces non-deterministic reduction order
    during accuracy regression
  keywords:
  - VLLM_BATCH_INVARIANT
  - batch_invariant
  - HCCL_DETERMINISTIC
  - LCCL_DETERMINistic
  - AISBench
  - EvalScope
  - lm_eval
  - PYTHONHASHSEED
  - deterministic
---
# Deterministic batch invariance, HCCL deterministic reduction, and AISBench/EvalScope evaluation gates

## Condition

Applies when validating numerical reproducibility across varying batch sizes (`BS=1` vs `BS>1`), comparing eager vs `ACLGraph` outputs, or running standardized accuracy benchmarks (`AISBench`, `EvalScope`, `lm_eval`) on `vllm-ascend`.

## Constraint

1. **Batch-invariance mode (`VLLM_BATCH_INVARIANT=1`, `vllm_ascend/batch_invariant.py`):**
   - Setting `VLLM_BATCH_INVARIANT=1` activates `init_batch_invariance()`, which enforces deterministic operator tiling and reduction trees across batch sizes by setting:
     - `HCCL_DETERMINISTIC=true` and `LCCL_DETERMINISTIC=1` (deterministic collective reduction order);
     - `CLOSE_MATMUL_K_SHIFT=1` (disables batch-dependent K-dimension split-K / shift optimizations in cube matmul);
     - Overrides `torch.mm`, `torch.matmul`, `torch.nn.functional.linear`, and `vllm_ascend` custom attention/norm dispatch to use fixed-tile deterministic paths;
     - Disables non-deterministic communication/fusion shortcuts (`enable_flashcomm1=False`, `enable_fused_mc2=0`) when strict bitwise batch invariance is required.
2. **Prefix-cache and hash determinism:**
   - Any evaluation or multi-node KV pool / UCM / Mooncake run must pin `export PYTHONHASHSEED=0` (or an identical seed across all ranks/instances) and `--seed <int>` with `temperature=0.0` before comparing token sequences or prefix-cache hit rates.
3. **Standardized evaluation harness contract (`AISBench` / `EvalScope`):**
   - Never judge model or quantization accuracy from a single ad-hoc prompt. Run `ais_bench` (`vllm_api_general_chat` / `vllm_api_stream_chat`) or `evalscope` against the full dataset split (e.g., `gsm8k`, `ceval`, `mmlu`, `math_500`, `gpqa_diamond`) with `--max-model-len` sized above the longest prompt plus `max_tokens` (at least `32768`–`35000` for reasoning/CoT models so `<think>` blocks do not truncate before the final answer).

## Rationale

Cube matmul K-splits and asynchronous HCCL/AIV reductions vary their accumulation tree with the active token count `T`, causing `BS=1` vs `BS=16` logit drift even at `temperature=0`. `VLLM_BATCH_INVARIANT=1` isolates genuine operator bugs from reduction-order drift.

## Evidence

Implemented in `vllm_ascend/batch_invariant.py` and documented in `docs/source/user_guide/feature_guide/batch_invariance.md`, `docs/source/developer_guide/evaluation/using_ais_bench.md`, and `docs/source/developer_guide/evaluation/using_evalscope.md`.
