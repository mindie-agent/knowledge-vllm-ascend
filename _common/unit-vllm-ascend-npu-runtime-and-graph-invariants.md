---
category: axiom
conditions:
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - vllm-ascend-idle-dp
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
domain_path: _common
evidence_strength: hardware_spec
id: unit-vllm-ascend-npu-runtime-and-graph-invariants
inherits_from: []
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py
  source_digest: 61ee8bc56d4ac0e1a7f43d1804bfe41c1a6699181774938907c97174f2348343
  source_locator: vllm-project/vllm-ascend@d8c9e3a673af918c94a7794ef6f1f69c024d9c62:vllm_ascend/worker/model_runner_v1.py
  source_type: runtime_architecture_specification
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Ascend NPU runtime and graph invariants in vllm-ascend model_runner_v1
triggers:
  error_signatures:
  - idle DP rank hangs or diverges because a zero-token step skipped the DP collective
  - GDN conv_state or ssm_state changes after a zero-token dummy step
  - decode logits diverge after query_start_loc or num_accepted_tokens reuse
  - torch.ops._C_ascend is missing or an NZ-layout weight is read as ND
  keywords:
  - vllm-ascend
  - model_runner_v1
  - dummy_run
  - skip_gdn_state_update
  - coordinate_batch_across_dp
  - GDN
  - GatedDeltaNet
  - conv_state
  - ssm_state
  - query_start_loc
  - num_accepted_tokens
  - _C_ascend
  - NZ
  - Ascend910_93
---

# Ascend NPU runtime and graph invariants in `model_runner_v1`

## Scope

These axioms describe the execution contract of `vllm_ascend.worker.model_runner_v1.NPUModelRunner` on Ascend 910B (A2) and Ascend 910_93 (A3 dual-Die). They are source-scoped to the pinned revision in `provenance`; they are architecture invariants, not a measured performance claim and not evidence that every model, graph mode or cache mode has been validated. A consuming procedure must still establish its own model, dtype, graph mode and software stack.

## Axiom 1: `execute_model` enters the DP collective through `_dummy_run`

`execute_model` first updates request state, then reads `scheduler_output.total_num_scheduled_tokens`. When that value is zero and the runner uses `distributed_executor_backend == "external_launcher"` with `data_parallel_size > 1`, the runner calls:

```python
self._dummy_run(1, skip_gdn_state_update=True)
```

before returning an empty runner output. The reason is collective synchronization, not request execution: every DP rank must enter `coordinate_batch_across_dp`, otherwise the idle rank's absence leaves the active ranks blocked in an all-rank operation.

The call graph is:

```text
execute_model
  -> _update_states
  -> zero scheduled tokens + external_launcher + DP>1
     -> _dummy_run(1, skip_gdn_state_update=True)
        -> _build_attention_metadata
           -> _build_attn_group_metadata (per KV-cache group)
```

The dummy step must therefore preserve the tensor ranks and participation shape that graph replay and HCCL expect, while carrying no real request tokens. It is not a normal one-token request step and must not be recorded or scheduled as one.

## Axiom 2: recurrent state is not paged KV state

A hybrid model has two different kinds of cache:

- **Stateful recurrent layers** — `GatedDeltaNet`/`GDN`, Mamba and related linear-attention layers — own a convolution state (`conv_state`) and a recurrent state (`ssm_state`) in the KV-cache tensors. These tensors persist across steps and are indexed per request by state-index tensors. A write to the wrong row changes a later request even if its paged KV blocks are untouched.
- **Stateless attention layers** — MLA and ordinary paged KV attention — own paged key/value blocks. A step reads and writes blocks selected by `slot_mapping` and `block_table`; it has no per-request recurrent state of this kind.

`num_actual_tokens` and the query-start-location tensor select how many rows of a state tensor are active. They are not a cosmetic shape hint. In the GDN metadata builder, a nonzero `num_actual_tokens` with a synthetic query schedule makes the recurrent kernel treat dummy tokens as real tokens.

The idle-DP fix adds `skip_gdn_state_update=True`. In that mode:

- `_dummy_run` fills `gdn_query_start_loc` with zeros instead of a cumulative one-token schedule;
- `_build_attn_group_metadata` replaces the GDN group metadata with the zero-filled start-loc prefix, `num_actual_tokens=0`, `max_query_len=0`, and an all-false `is_prefilling`;
- non-GDN attention groups keep the dummy one-token metadata so graph and collective shapes stay stable;
- real batches leave the flag at its default `False`.

The in-code comment calls the resulting GDN work a kernel no-op because operators slice `mixed_qkv[:0]`. That is a design statement, not a state-parity measurement. A shape-preserving operator can still write a state row, so a claimed state invariance must be established by the checksum gates in `unit-vllm-ascend-verification-and-state-parity-gates`.

Two rules follow. Never pass `skip_gdn_state_update=True` for a batch with scheduled tokens. Never reuse the zero-filled idle `gdn_query_start_loc` as a real decode schedule.

## Axiom 3: decode and prefill use different token indexing

`query_start_loc` is a cumulative token-to-request map. For a real batch it is built from the scheduled token counts, and GDN reads the unpadded `gdn_query_start_loc` prefix whose first entry is zero and whose last entry is the real token count. Padding to a graph shape appends repeated or padded positions; those rows must be marked with `PAD_SLOT_ID` or `-1` and must not be associated with a real request's state row.

Decode and prefill differ in more than sequence length:

- Prefill consumes a prompt chunk; `is_prefilling` is derived from computed versus prompt tokens, and GDN prefill may gather an initial state and zero rows whose `has_initial_state` is false before the chunk rule.
- Decode consumes accepted tokens per request. `num_accepted_tokens` records how many draft tokens survived verification, and speculative decode indexes state with `spec_state_indices_tensor` plus `num_accepted_tokens`; non-speculative decode uses the normal state indices and a one-token (or accepted-token) query length.
- `num_actual_tokens` may be smaller than the padded token count. The padding tail is a graph/collective artifact, not extra requests.

A correct metadata adaptation keeps four things aligned: the request-to-state-row mapping, the cumulative `query_start_loc` prefix, the accepted-token count, and the padding sentinel. Reusing any one of them from a different phase corrupts the mapping even when tensor shapes still match.

## Axiom 4: custom operators and NZ layout are runtime bindings

Ascend kernels are reached through `torch.ops._C_ascend`. The binding is provided by the compiled extension loaded during `vllm-ascend` initialization; meta registrations describe the operators to the compiler. A missing or partially loaded `.so` can surface as an `AttributeError`/`RuntimeError` at the first operator call rather than at import time, and a retry that adds a vendor library path can change which symbols are visible. Operator availability is therefore a runtime precondition, not a property of the Python source alone.

Weights selected for the NZ fractal layout are physically different from ND tensors. NZ selection is gated by dtype, device and the configured weight-NZ mode; float32 and meta tensors are excluded, and device-specific policy decides the remaining cases. A tensor produced in NZ must be consumed by an operator that expects NZ, and an ND-shaped view must not be used to reinterpret it. The layout invariant is part of the weight contract, not an optional optimization detail. Mixing the two layouts produces wrong values that can look like a model or precision failure.

## Failure signatures and limits

Treat the following as invariant violations, not tuning opportunities:

- an idle rank skipped the DP collective and peers hang in `coordinate_batch_across_dp`;
- a zero-token GDN step reports nonzero `num_actual_tokens` or a nonzero `gdn_query_start_loc`;
- an idle step changes `conv_state` or `ssm_state` while the request set is unchanged;
- a real decode step consumes the zero-filled idle start-loc or a stale accepted-token count;
- an operator call fails because `_C_ascend` was not bound, or a weight is read with the wrong NZ/ND interpretation.

These axioms do not establish numerical parity, HCCL completion or state invariance by themselves. They define what must be probed, and the paired measurement unit defines the ordered gates that produce that evidence.
