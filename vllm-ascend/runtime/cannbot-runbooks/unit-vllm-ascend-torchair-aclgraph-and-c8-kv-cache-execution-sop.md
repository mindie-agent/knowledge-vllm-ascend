---
category: sop
conditions:
  execution_mode:
  - aclgraph
  - torchair
  - piecewise_graph
  framework: vllm-ascend
domain_path: vllm-ascend/runtime/cannbot-runbooks
evidence_strength: software_contract
id: unit-vllm-ascend-torchair-aclgraph-and-c8-kv-cache-execution-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/issues/2410
  - https://github.com/vllm-project/vllm-ascend/pull/2113
  - https://github.com/vllm-project/vllm-ascend/pull/3158
  - https://github.com/vllm-project/vllm-ascend/pull/2681
  - https://gitcode.com/cann/cann-recipes-infer/issues/80
  - https://gitcode.com/cann/cann-recipes-infer/issues/245
  - https://gitcode.com/cann/cann-recipes-infer/merge_requests/179
  - https://gitcode.com/cann/cann-recipes-infer/merge_requests/206
  - https://gitcode.com/cann/cann-recipes-infer/merge_requests/328
  experiment_status: not_run_source_only
  source_digest: 128f1eabeb7dac06a396cb33188f901698ed8948b78da4498b5a5f7d1153cece
  source_locator: cann/cannbot-knowledge/knowledge/model/inference/runbooks/runtime/aclgraph_sq_cq_alloc_fail_capture_sizes_budget.md
  source_type: cannbot_runbook_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: TorchAir/ACLGraph execution and C8 KV-cache SOP for vllm-ascend
triggers:
  error_signatures:
  - ACLGraph replay falls back to eager or recaptures a new shape after warmup
  - Alloc sq cq fail with 207005 or EL0006 during graph capture
  - positions, slot_mapping, block_tables or seq_lens disagree between capture and replay
  - C8 model raises "enable_mla_prolog must be set to True" or produces wrong logits after KV quantization
  - torch.compile reports a Meta or FakeTensor failure for a custom NPU operator
  - graph capture OOMs while steady-state memory is lower
  - multistream decode returns nondeterministic tokens without an exception
  keywords:
  - ACLGraph
  - TorchAir
  - torchair_graph_config
  - use_cached_graph
  - capture_sizes
  - graph_batch_sizes
  - piecewise
  - static buffer
  - positions
  - slot_mapping
  - block_tables
  - seq_lens
  - attn_metadata
  - C8
  - int8 KV cache
  - reshape_and_cache
  - FIA
  - FlashMLA
  - antiquant_scale
  - antiquant_offset
  - dequant_scale_key
  - register_fake
  - FakeTensor
  - Meta
  - symbolic shape
  - PYTORCH_NPU_ALLOC_CONF
  - expandable_segments
  - gpu_memory_utilization
  - Alloc sq cq
---

# TorchAir/ACLGraph execution and C8 KV-cache SOP for vllm-ascend

## Applicability and symptom scope

Use this SOP when a `vllm-ascend` model runs under ACLGraph, TorchAir GE graph, or a piecewise graph mode, especially when the graph includes paged attention, MoE, or an int8 (C8) KV cache. It applies to three families of symptoms:

- the graph captures but replay diverges, falls back to eager, or recaptures on a later step;
- capture fails with a resource error that looks like OOM but is actually a stream or workspace budget problem;
- a custom operator or a C8 cache path works in eager but not inside the captured graph.

The SOP is a software-contract procedure. It does not claim a speedup and does not replace a per-operator numerical investigation once one operator is isolated.

## Procedure

1. **Choose the graph mode and freeze the input signature before capturing.**

   Record whether the model uses piecewise ACLGraph, full-decode-only ACLGraph, or TorchAir GE graph, and record the resolved config keys. In `vllm-ascend` the relevant block is `torchair_graph_config`; the capture bucket list is named `capture_sizes` in some revisions and `graph_batch_sizes` in others, and `use_cached_graph` controls whether a compiled graph cache is reused. Write down the resolved names and values, not just the YAML.

   ```yaml
   torchair_graph_config:
     enabled: true
     use_cached_graph: true
     capture_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256]
   ```

   The graph is keyed by shape, dtype and stride. A decode step that lands between two capture sizes recaptures or falls back. Freeze the decode shape with `torch._dynamo.mark_static` only for the dimensions that are genuinely fixed, make sliced inputs contiguous, and use the same dtype for the warmup dummy and the real decode input.

   ```python
   # Fixed decode shape: mark static. Do not mark the dynamic prefill sequence length.
   torch._dynamo.mark_static(positions)
   prev_hidden_states = torch.cat([prev, prev_hidden_states[:, -1:, :]], dim=1)[:, -L:, :].contiguous()
   ```

   **Invariant:** every steady-state decode step maps to one captured bucket; no `torch.compile` recompilation occurs after warmup; the warmup dtype and layout equal the serving dtype and layout.

   **Verify:** run `TORCH_LOGS=recompiles` or a Dynamo explanation for ten steady decode steps and confirm the recompile count is zero. Confirm `use_cached_graph` actually hit the cache by checking the compile log rather than assuming the flag is enough.

2. **Pin every graph input and metadata tensor to a persistent static buffer.**

   The captured region may not allocate, synchronize the host, or change a data pointer. `positions`, `slot_mapping`, `block_tables`, `seq_lens` (and `query_start_loc` where present), `attn_metadata`, and any mask or padding sentinel must be copied in place into persistent device storage on the capture stream before replay.

   ```python
   # Persistent buffers are allocated once, outside capture.
   positions.copy_(new_positions, non_blocking=True)
   slot_mapping.copy_(new_slot_mapping, non_blocking=True)
   block_tables.copy_(new_block_tables, non_blocking=True)
   seq_lens.copy_(new_seq_lens, non_blocking=True)
   ```

   Do not call `.item()`, `.cpu()`, `torch.npu.synchronize()` or a dynamic `torch.empty` inside the captured region. A host scalar extraction is a graph break; a dynamic allocation changes addresses and can replay stale data.

   For a padded bucket, set the tail explicitly: `slot_mapping = -1` for padded positions, or `seq_lens = 0` for the padded requests, so a padded token never writes a real KV block or a recurrent state slot. Keep the sentinel convention identical between capture and replay.

   **Invariant:** the `data_ptr()` of every graph input is unchanged across replays; no host synchronization or allocation occurs inside capture; the padded tail cannot address a live KV block.

   **Verify:** record the pointers of the five metadata tensors for three consecutive replays and confirm equality. Run a `B_large -> B_small -> B_large` bucket transition and compare replay logits with eager at the same inputs. Inspect the tail rows for `slot_mapping == -1` and confirm the KV checksum of live blocks is unchanged.

3. **Validate the C8 KV-cache path end to end: scale provenance, int8 write, and FIA/FlashMLA antiquant contract.**

   For an MLA model, C8 requires `enable_mla_prolog=True`; without it the runner raises `ValueError("if kv_cache_quant_mode is C8, then enable_mla_prolog must be set to True.")`. C8 and the PyPTO Lightning Indexer fast path are mutually exclusive, so do not diagnose a missing PyPTO path as a hardware problem when C8 is on.

   In the DeepSeek/MLA C8 path described by the source, C8 scales are pre-calibrated per-model artifacts, not values recomputed from the weights at load time. A changed weight snapshot invalidates a preset scale package; a structurally isomorphic sibling model may reuse a package only with an explicit unverified-path caveat. Record the package identity and the checkpoint hash together.

   The int8 write path is a quantize-then-scatter, not a cast. A `reshape_and_cache`-style fused operator is acceptable only if it accepts an int8 cache dtype and the explicit quantization scale; otherwise use the two-step form:

   ```python
   # Per-tensor C8 example; per-token-head or per-token-head-tile variants
   # use the scale layout required by the target model.
   cache_k_int8 = torch_npu.npu_quantize(
       cache_k, reciprocal(k_scale), None, torch.qint8, -1, False
   )
   # slot_mapping is int32 and addresses the paged cache row.
   src = cache_k_int8  # rows for this step, shaped to match slot_mapping
   torch_npu.npu_scatter_nd_update_(cache_k_int8_view, slot_mapping.view(-1, 1), src)
   ```

   For the MLA prolog C8 branch, `kr_cache` (`k_rope`) is an empty int8 tensor because nope, rope and scales are concatenated into `k_nope`. Preallocate it once and reuse the same object:

   ```python
   self.kr_cache = torch.empty((1, 2, 1, 0), dtype=torch.int8, device="npu")
   ```

   Any graph-mode operator that updates the KV cache in place must be guarded so the graph observes the update. This is independent of C8: apply it to whichever custom operator owns the in-place write. A common pattern is `torch.add(cache, 0)` on each in-place input and returning the result; without the guard, downstream nodes can read the pre-update cache.

   The attention operator's antiquant contract depends on the FA version. FIA v1 takes `antiquant_scale` and `antiquant_offset`; FIA v2 takes `dequant_scale_key` and `dequant_scale_value`. Do not mix names across versions. The scale and offset must be floating point and broadcastable to the key/value head or channel dimension that the selected layout declares; a scalar is only correct for a per-tensor scheme. A packed `uint64` scale or an int64 tensor is not a valid float scale and will be interpreted as an enormous value.

   **Invariant:** C8 is enabled together with MLA prolog; the scale package matches the checkpoint; the cache dtype is int8 and the write uses the model's declared granularity; FIA v1 and v2 parameter names are not mixed; scale and offset dtype/shape match the selected schema.

   **Verify:** run a one-layer known-answer decode with C8 on and off; compare the dequantized KV row and the layer output at the frozen tolerance. Check the first and last page, a zero offset and a non-zero storage offset. Confirm the attention profile shows the expected antiquant path rather than a silent BF16 fallback.

4. **Register a symbolic-shape-safe Meta or FakeTensor implementation for every custom operator that must enter the graph.**

   `torch.compile` traces with fake tensors and does not run the real NPU kernel. A custom operator without a `Meta` implementation either graph-breaks out of the compiled region or is called eagerly, so it never enters ACLGraph. The symptom is often silent: the service runs, but the operator is absent from the captured graph.

   For a Python `torch.library.custom_op`, register `@op.register_fake` next to the definition. For a C++ `_C` operator, register a `*_meta` function with `TORCH_LIBRARY_IMPL_EXPAND(_C, Meta, ops)`. The Meta function returns empty tensors with the correct shape, dtype and device and performs no real computation.

   ```python
   @torch.library.custom_op("vllm_ascend::my_op", mutates_args=())
   def my_op(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
       return torch.ops._C_ascend.my_op(x, scale)

   @my_op.register_fake
   def _(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
       # Symbolic-shape safe: no .item(), no concrete numel arithmetic.
       return torch.empty_like(x, dtype=x.dtype, device=x.device)
   ```

   If the output dtype differs from the input (dynamic quant to int8 is the common case), declare it explicitly. `torch.empty_like(x)` inherits `x.dtype` and will make the graph produce a BF16 tensor where the kernel produces int8.

   Registration must happen after the custom operator library is imported. In `vllm-ascend` the meta module is imported from the custom-op enable path; importing it before the extension means the operator is registered but the Meta implementation is not.

   **Invariant:** every custom operator that must enter the graph has a Meta implementation; the Meta implementation is symbolic-shape safe, uses no host scalar extraction and runs no NPU kernel; output dtype and shape match the real kernel.

   **Verify:** compile a small model with `fullgraph=True` and a checking backend, and assert the operator name appears in `str(gm.graph)`. Replay the captured graph and compare with eager. If the operator is absent, the Meta registration or the import order is wrong, not the kernel arithmetic.

5. **Separate capture-time OOM from workspace-pool and stream-resource exhaustion.**

   Three failures look similar and have different fixes:

   - **HBM OOM:** `EL0004` / `Memory_Allocation_Failure`, or a device allocation failure reported by the allocator while `torch.npu.memory_summary()` shows reserved memory near capacity.
   - **Host OOM:** `EH0010` or a host allocator failure; do not free NPU HBM for it.
   - **SQ/CQ or workspace exhaustion:** `Alloc sq cq fail`, `207005`, `EL0006`, or `rtStreamBeginCapture execute failed`. This is stream queue resource exhaustion, not HBM. Freeing memory or killing other processes will not fix it.

   For the SQ/CQ case, reduce the number of capture gears. The source's planning formulas are implementation estimates:

   ```text
   A3: num_capture_sizes = (1920 - pg_num * 40) / (num_hidden_layer + 1) / (1 + pg_num * 2)
   A2: num_capture_sizes = 1920 / (num_hidden_layer + 1) / (1 + pg_num * 1)
   ```

   `pg_num` is the number of communication process groups and `num_hidden_layer` is the number of hidden layers. Reduce `capture_sizes` to the computed ceiling; reducing communication groups also lowers the pressure.

   For genuine HBM pressure at capture time, first check whether the steady-state footprint is lower than the capture footprint. The caching allocator may hold fragmented segments that cannot be reused for the capture pool:

   ```bash
   export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
   python3 - <<'PY'
   import torch, torch_npu
   print(torch.npu.memory_summary())
   print("allocated", torch.npu.memory_allocated() / 2**30, "GiB")
   print("reserved ", torch.npu.memory_reserved() / 2**30, "GiB")
   PY
   ```

   `gpu_memory_utilization` sets the fraction of device memory the engine may use for weights, activations and KV cache; the KV-cache capacity is what remains after the model and runtime footprint. Lowering it can avoid a capture OOM, but it also reduces KV capacity; raise it only after confirming that the capture peak, not the steady-state peak, is the binding constraint.

   If a MoE token dispatcher holds dispatch/combine intermediates as `self` attributes, the ACLGraph peak reservation includes tensors that are logically dead after combine. Set those attributes to `None` at the end of `token_combine`; this shortens the lifetime without changing results.

   **Invariant:** the first failing resource is identified before any knob is changed; `Alloc sq cq fail` is not treated as HBM OOM; `gpu_memory_utilization` is not used to paper over a stream-resource failure; capture and replay use the same pool.

   **Verify:** capture the model with the reduced gear set and confirm no `207005` or `EL0006`; replay for 100 steps and confirm memory reaches a plateau rather than growing; compare replay logits with eager; confirm `Alloc sq cq fail` is absent from the plog after the change.

## Boundaries and rollback

The A3 and A2 capture-size formulas are estimates from one `vllm-ascend` implementation and may change with the torch_npu and CANN version. C8 granularity differs by model: per-tensor, per-token-head and per-token-head-tile-128 are all legitimate, and the scale layout must match the model rather than a generic recipe. The `antiquant_scale`/`dequant_scale_key` naming split is an FA-version contract; confirm the installed operator schema before changing a call site.

Rollback is per step: restore the previous `capture_sizes` or `use_cached_graph` value, restore eager attention if the C8 scale package is suspect, remove a Meta registration only if the operator is intentionally left out of the graph, and restore the previous allocator configuration. Keep the failed capture log and the pre-change config; a clean retry without them is not evidence.
