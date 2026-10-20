---
category: workflow
conditions:
  execution_mode:
  - ge_graph
  - npugraph_ex
  - aclgraph
  npu_model:
  - Ascend910B
  - Ascend910B2
  - Ascend910B3
  - Ascend910B4
  - Ascend910_93
  optimization_mechanism:
  - multi_stream_overlap
  - npu_prefetch
  - limit_core_num
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - cann/cann-recipes-infer
domain_path: vllm-ascend/performance/multi-stream-prefetch
evidence_strength: software_contract
id: unit-vllm-ascend-multi-stream-overlap-and-npu-prefetch-workflow
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
- unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-multi-stream/SKILL.md#L27-L125
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-multi-stream/references/timeline-overlap-check.md#L5-L89
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-multi-stream/references/ge-reorder-design-check.md#L1-L27
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-prefetch/SKILL.md#L13-L396
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/utils/stream_utils.py#L33-L114
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/utils/common_utils.py#L120-L155
  experiment_status: not_run_source_only
  source_digest: 4b1ef07a0d42a4c592e3116a26d2f8355ddc5e3ca6cdd380c73a83136e40116d
  source_locator: cann/cannbot-skills:f305e200aa4b1e1fed73cd21ff1f083b8df25a1e:model/model-infer-multi-stream/SKILL.md
  source_type: cannbot_skill_and_source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Multi-stream overlap verification, GE reorder barrier triage, and npu_prefetch window sizing workflow
triggers:
  error_signatures:
  - side stream operators land on distinct Stream ID but module wall-clock duration is unchanged
  - overlap_pct <= 0.05 because GE auto-reorders cheap precompute onto the main stream ahead of the dominator
  - torch.compile with fullgraph=True and dynamic=True rejects Stream or Event objects lacking as_proxy
  - torch_npu.npu_prefetch degrades the dependency window because max_size saturates HBM bandwidth
  keywords:
  - multi-stream
  - npu_stream_switch
  - npu_wait_tensor
  - npu_tagged_event_record
  - npu_tagged_event_wait
  - record_stream
  - limit_core_num
  - overlap_pct
  - GE auto-reorder
  - npu_prefetch
  - dependency window
  - roofline
---

# Multi-stream overlap verification, GE reorder barrier triage, and npu_prefetch window sizing workflow

## Objective & Applicability

Use this workflow when introducing or diagnosing multi-stream compute/communication overlap, core-count partitioning (`limit_core_num`), or `torch_npu.npu_prefetch` weight prefetching on Ascend NPU inference (`ge_graph` or `npugraph_ex` / `aclgraph`). It prevents three common failure modes:
1. Mixing GE Ascend IR scope APIs with explicit `torch.npu.Stream` / `npugraph_ex` APIs.
2. Mistaking a logical `Stream ID` split for physical overlap when GE compile-time reordering inserts a cross-stream barrier ahead of the main-stream dominator (`overlap_pct <= 0.05`).
3. Inserting `torch_npu.npu_prefetch` without a non-memory-bound dependency window or with an oversized `max_size` that starves the predecessor kernel.

This unit is a software-contract workflow derived from `cann/cannbot-skills` and `cann/cann-recipes-infer`; it claims no unmeasured speedup.

---

## Stage 1 — Route Multi-Stream & Core-Limit APIs by Execution Backend

Before writing stream or event code, lock the active graph backend. Never mix GE Ascend IR APIs and explicit `torch.npu.Stream` / `npugraph_ex` APIs in the same execution path:

| Capability | GE Graph Mode (`torchair.get_npu_backend()`) | `npugraph_ex` / ACLGraph (`backend="npugraph_ex"`) |
| --- | --- | --- |
| **Stream switch** | `torchair.scope.npu_stream_switch(stream_tag, stream_priority=0)` | Explicit `with torch.npu.stream(s):` or string-tag `torch.npu.npugraph_ex.scope.npu_stream_switch(tag)` when `fullgraph=True, dynamic=True` blocks `Stream` proxy tracing |
| **Cross-stream ordering** | `torchair.scope.npu_wait_tensor(consumer_in, producer_out)` or `torchair.ops.npu_tagged_event_record` / `npu_tagged_event_wait` | `torch.npu.Event()` (`record()` / `wait()`) or `torch.npu.npugraph_ex.scope.npu_tagged_event_record` / `npu_tagged_event_wait` |
| **Lifetime protection** | `torchair.ops.npu_record_tagged_stream(out, stream_id)` | `out.record_stream(side_stream)` mandatory for short-lived tensors consumed across streams |
| **Core-count partitioning** | Operator-scope `torchair.scope.limit_core_num(aic_num, aiv_num)` (overrides global `config.ge_config.aicore_num`) | Stream-scope `torch.npu.npugraph_ex.scope.limit_core_num(aic_num, aiv_num)` |

### Design-Time GE Auto-Reorder Pre-Check (`ge_graph` only)
`torchair.scope.npu_stream_switch` attaches a logical tag; GE still schedules operators by topological data dependencies across the `with` block boundary. Before running a candidate in `ge_graph`:
1. List every main-stream consumer of a side-stream output tensor `T`.
2. Check whether `T` feeds a lightweight precompute op (`Cast`, `Reshape`, `Swish`, `Sigmoid`, `Mul`, e.g., `silu(z) = Swish(Cast(z))`).
3. If that lightweight op is left outside the `with npu_stream_switch(...)` block, GE can hoist it onto the main stream FIFO queue **ahead** of the main-stream dominator (`MatMul` / `FA` / `Conv1d`), forcing the main stream to stall until the side stream finishes `T`.
4. **Fix at design time**: Explicitly place the lightweight precompute inside the `with npu_stream_switch(...)` block so it executes on the side stream before the single join point.

---

## Stage 2 — Diagnose Multi-Stream Profit via Gantt `overlap_pct` (3a → 3b → 3c)

Never declare a multi-stream candidate effective merely because `kernel_details.csv` shows multiple `Stream ID` values, and never jump to host-bound or HBM-contention explanations before checking `overlap_pct` in a single profiling capture (skipping 2–3 cold-start steps):

1. **Extract per-op intervals from one steady step** (`Stream ID`, `Start Time(us)`, `Duration(us)`):
   $$\text{side\_window} = [\min s_{\text{side}}, \max(s_{\text{side}} + d_{\text{side}})], \quad \text{main\_window} = [s_{\text{dom}}, s_{\text{dom}} + d_{\text{dom}}]$$
   $$\text{overlap} = \max\bigl(0, \min(\text{side\_end}, \text{main\_end}) - \max(\text{side\_start}, \text{main\_start})\bigr)$$
   $$\text{overlap\_pct} = \frac{\text{overlap}}{\min(\text{side\_len}, \text{main\_len})}$$
2. **Branch strictly in order `3a → 3b → 3c`**:
   - **Branch 3a — Logical stream switch did not fire**: Side-stream ops still share the main `Stream ID`. Check whether `enable_multi_stream` / `is_prefill` guards bypassed the context manager or `_TORCHAIR_AVAILABLE` fell back to a no-op context.
   - **Branch 3b — False overlap (`overlap_pct <= 0.05`, parallelism $\le 1.05\times$)**: Side-stream ops have a secondary `Stream ID` but execute serially before or after `main_window`. Sort main-stream ops by `Start Time(us)`, locate the first main-stream op consuming a side-stream tensor ahead of the dominator (the GE-hoisted barrier op), and pin that op inside the side-stream `with` block.
   - **Branch 3c — True overlap (`overlap_pct >= 0.5`) without wall-clock reduction**: Only after proving `overlap_pct >= 0.5`, inspect resource contention:
     - Compare the sum of concurrent `Block Dim` values in `kernel_details.csv` against hardware `ai_core_cnt` (do not guess from op names alone, as small-batch `MatMul` may use `Block Dim = 1`).
     - If concurrent Cube or Vector blocks exceed physical cores and cause tail stretching, apply `limit_core_num(aic_num, aiv_num)` or repartition the candidate so one stream is Cube-heavy and the other is Vector/memory-heavy.
     - Check whether cross-stream tensor transfer introduced a long-tail `TransData`, `BroadcastTo`, or `MemSet`.

---

## Stage 3 — Roofline-Guided `torch_npu.npu_prefetch` Window & Size Selection

Apply `torch_npu.npu_prefetch(weight, depend, max_size, offset=0)` only when profiling confirms a memory-bound weight-loading hotspot (`MatMul`, `BatchMatMul`, `QuantBatchMatmul`, `GroupedMatmul`, or `MLAProlog`):

1. **Select a safe `dependency` tensor window**:
   - The `depend` tensor must lie on the same logical execution path immediately preceding the target op (never reuse a tensor from an unrelated branch or phase in graph mode).
   - Valid dependency windows are either:
     - **Compute/vector-bound predecessors** (`RMSNorm`, `LayerNorm`, `RoPE`, `SwiGLU`, router `topk`, `npu_dequant_swiglu_quant`) whose execution time overlaps the prefetch DMA without competing for HBM bandwidth, or
     - **The output tensor of a just-completed large MatMul** (e.g., triggering `gate_up_proj` or `router.classifier` prefetch on `o_proj` output, or next-layer `q_a_proj`/`q_b_proj`/`kv_a_proj` prefetch on `down_proj` / `gmm2_out` output) so prefetch begins as soon as the predecessor finishes.
   - Do **not** attach `npu_prefetch` to a window saturated by concurrent collective communication (`all_reduce`, `all_to_all`) or large `Concat`/`Transpose` copies.
2. **Size `max_size` conservatively first ($50\% \to 70\% \to 100\%$)**:
   - Compute theoretical weight bytes from the per-rank shard shape and dtype (`2` bytes for `bf16`/`fp16`, `1` byte for `int8`):
     $$\text{GMM1}_{\text{prefetch}} = \frac{H \times I \times 2 \times \text{dtype\_bytes}}{\text{moe\_tp\_size}} \times \frac{\text{experts\_per\_rank}}{2}$$
     $$\text{GMM2}_{\text{prefetch}} = \frac{H \times I \times \text{dtype\_bytes}}{\text{moe\_tp\_size}} \times \text{experts\_per\_rank}$$
   - Start at $0.5\times$ of theoretical weight bytes on a single target op behind an `enable_prefetch` switch.
   - If the target op accelerates and the dependency window duration is stable, step up to $0.7\times \to 1.0\times$. If the predecessor window elongates, reduce `max_size` ($50\% \to 30\% \to 10\%$) or move `depend` earlier.

## Verification & Rollback Gates

- Every multi-stream or prefetch change must be gated by an explicit `enable_multi_stream` / `enable_prefetch` configuration flag with a clean single-stream fallback.
- Verify output token parity against the single-stream baseline and confirm both `overlap_pct >= 0.5` (for multi-stream) and net critical-path reduction in `kernel_details.csv` before retaining the change.
