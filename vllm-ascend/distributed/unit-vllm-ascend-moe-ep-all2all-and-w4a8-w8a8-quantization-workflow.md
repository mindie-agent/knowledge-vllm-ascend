---
category: workflow
conditions:
  architecture: moe
  framework: vllm-ascend
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: vllm-ascend/distributed
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-moe-ep-all2all-and-w4a8-w8a8-quantization-workflow
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-pd-disaggregation-mooncake-kv-transfer-and-mc2-sop
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/issues/4051
  - https://github.com/vllm-project/vllm-ascend/pull/5521
  - https://github.com/vllm-project/vllm-ascend/issues/495
  - https://github.com/vllm-project/vllm-ascend/pull/3005
  - https://github.com/vllm-project/vllm-ascend/pull/2681
  - https://gitcode.com/cann/cann-recipes-infer/merge_requests/4
  - https://gitcode.com/cann/cann-recipes-infer/merge_requests/12
  - https://gitcode.com/cann/cann-recipes-infer/merge_requests/187
  - https://gitcode.com/cann/cann-recipes-infer/merge_requests/421
  - https://gitcode.com/cann/cann-recipes-infer/issues/120
  source_digest: aad16f59cf8332fc7142540739697eb7e69f26f5a92c63e8c4a3d1ab9376e453
  source_locator: cann/cannbot-knowledge/knowledge/model/inference/optimizations/experience/rules/moe_ep_dispatch_path_by_stage.md
  source_type: cannbot_runbook_and_transcript_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: MoE EP communication-regime selection with All2All/MC2 and W8A8/W4A8 quantization workflow
triggers:
  error_signatures:
  - MoeDistributeDispatchV2 do tiling failed or HCCL_BUFFSIZE is too SMALL during profile_run
  - npu_moe_init_routing rejects active_expert_range or topk_ids because of dtype
  - npu_grouped_matmul returns wrong rows or hits 507015 when one expert receives zero tokens
  - accuracy drops only after enabling W4A8 while the service still runs
  - activation quantization kernels are absent although a4w8 mode is configured
  - one DP or EP rank exits early on an empty batch and peers time out with EI0006
  keywords:
  - MoE
  - expert parallel
  - EP
  - TP
  - AllGather
  - All2All
  - double routing
  - MC2
  - npu_moe_distribute_dispatch_v2
  - npu_moe_distribute_combine_v2
  - npu_moe_gating_top_k_softmax
  - npu_moe_init_routing
  - npu_grouped_matmul
  - group_list_type
  - expert_token_num
  - W8A8
  - W4A8
  - W4A16
  - SmoothQuant
  - AWQ
  - GPTQ
  - per-token
  - per-channel
  - group_size
  - shared expert
  - dummy run
  - HCCL
---

# MoE EP communication-regime selection with All2All/MC2 and W8A8/W4A8 quantization workflow

## Trigger and scope

Use this workflow when a `vllm-ascend` MoE deployment must choose or validate its expert-parallel communication path and its activation/weight quantization path together. Typical entries are a new MoE model, a change from TP to EP, a move from prefill-heavy to decode-heavy serving, a new W4A8 checkpoint, a switch between eager and graph mode, or a rank-timeout that appears only when one rank has no tokens.

The workflow is about regime selection and cross-layer contracts. It does not claim that one communication regime or one quantization scheme is universally faster. It also does not replace the kernel-level accuracy triage when a single grouped matmul or attention operator is already isolated as numerically wrong.

The four stages are ordered. A later stage is not meaningful until the earlier stage's invariant holds: a wrong communication regime cannot be rescued by better quantization, and a corrupt group list cannot be rescued by a larger HCCL buffer.

## Facts to freeze before profiling

Record these before the first benchmark or accuracy run, and keep them attached to every result:

| Fact | Why it changes the decision |
| --- | --- |
| `num_experts`, `experts_per_rank`, `top_k`, `shared_expert_num`, `num_zero_experts` | Selects the regime and sets the HCCL buffer requirement. |
| `moe_tp_size`, `moe_ep_size`, `dp`, `pp`, `cp` | TP and EP use different collectives; DP and EP change rank participation. |
| `hidden_size`, `intermediate_size`, `num_kv_heads`, `head_dim` | Sets per-token dispatch bytes, GMM shapes and KV geometry. |
| SoC and board topology | A2 has an experts-per-rank ceiling; A3/A5 have different capture and link behavior. |
| Intra-node HCCS versus inter-node RoCE | Changes transport, alignment and the failure signature. |
| `max_num_batched_tokens`, `max_num_seqs`, actual prefill and decode token distribution | Determines whether the path is bandwidth-bound or latency-bound. |
| Weight bits, activation bits, group size, symmetry, dynamic/static, packed layout | Determines W8A8, W4A8 or W4A16. |
| Graph mode (`eager`, ACLGraph, TorchAir) and capture sizes | Determines whether the path must be static and whether it can be captured. |
| `HCCL_BUFFSIZE`, `HCCL_INTRA_ROCE_ENABLE`, `HCCL_CONNECT_TIMEOUT`, `HCCL_OP_EXPANSION_MODE` | Must be identical across ranks; a mismatch is a bring-up failure. |
| Baseline revision, checkpoint hash, tokenizer and evaluation set | Prevents comparing a new quantized checkpoint with an old baseline. |

Freeze the tolerance before seeing outputs: for a BF16 reference use a per-layer maximum relative error near `1e-3`; for an int8 or int4 quantized path use a wider, pre-declared bound near `1e-2`. Do not widen the tolerance after the result.

## Stage 1 — Select the MoE communication regime from stage, EP degree and topology

Start from the parallel layout, not from a preference:

1. **`moe_tp_size > 1` and the model exposes a tensor-parallel MoE path:** the expert weights are tensor-parallel. Run grouped matmul on the local shard and finish with an all-reduce on the MoE TP group. There is no expert all-to-all in this path. Use it when the model fits the TP layout and the EP degree would be too small to justify dispatch.
2. **`moe_tp_size == 1`, `moe_ep_size > 1`, prefill or long-token chunk:** use double-routing. `npu_moe_init_routing_v2` sorts locally, two `all_to_all_single` calls exchange per-expert counts and tokens, `npu_moe_re_routing` reorders inside the destination rank, grouped matmul runs, the reverse all-to-all returns tokens, and `npu_moe_finalize_routing` accumulates. This path sends only valid tokens and has a lower peak than an all-gather of all tokens.
3. **`moe_tp_size == 1`, `moe_ep_size > 1`, decode with few tokens:** use MC2 `npu_moe_distribute_dispatch_v2` plus `npu_moe_distribute_combine_v2`. The fused pair amortizes host launch and fixed communication latency and produces a fixed-shape intermediate that can be captured. The expert grouped matmul still runs between the two operators.
4. **AllGatherEP:** only use it when the model and revision explicitly support the fused all-gather expert chain and the enabling environment switch is set, typically a 256-expert DeepSeek V3/R1 family with `finalize_routing` fusion and an NZ `w2` weight. It skips the TP pad and the ordinary all-gather and finishes with a tensor-model-parallel all-reduce. Do not generalize it to an arbitrary MoE; the source evidence for this path is unit-test level.
5. **Topology gate:** MC2 on an HCCS super-node can use `comm_alg="fullmesh_v2"`. A cross-node RoCE domain can still run MC2, but the transport, buffer and link checks in the PD SOP apply. If the rank table maps a rank to the wrong die, or `HCCL_INTRA_ROCE_ENABLE` differs across ranks, fix the topology before comparing regimes.
6. **Platform and structure gate:** on A2, MC2 dispatch/combine v2 is limited to 24 experts per rank in the unlayered mode. Above that, disable `enable_moe_mc2_dispatch` or run eager and use double-routing. A MoE with zero or identity experts must not enable `fullmesh_v2`; keep the dedicated MC2 domain but omit `comm_alg` and pass `copy_expert_num`.
7. **Batch gate:** the source reports a dispatch batch ceiling around 512 tokens for one LongCat-Flash-Lite configuration. Treat that as an implementation-specific measurement, not a universal cutoff. The safe rule is: if prefill routinely exceeds the dispatch operator's supported token count, use double-routing; if decode is a handful of tokens per step, MC2's fixed-shape and launch amortization usually wins.

A compact decision checklist:

```text
moe_tp_size > 1                         -> TP GMM + all_reduce
moe_tp_size == 1, prefill/long chunk     -> double-routing all_to_all
moe_tp_size == 1, decode, A3/A5         -> MC2 dispatch/combine v2
moe_tp_size == 1, decode, A2 >24/rank   -> double-routing or eager fallback
zero/identity experts                   -> MC2 without fullmesh_v2
```

Do not select a regime from a single end-to-end number. Record the per-step token count, the number of active EP ranks, the communication time and the grouped-matmul time separately. A regime that is faster at batch 1 may lose at prefill length 32K.

## Stage 2 — Align routing, permutation and grouped-matmul group lists

The routing chain has three independent contracts. Check each one before profiling.

**2.1 Gating output.** `npu_moe_gating_top_k_softmax` produces the top-k weights and expert indices. If the model renormalizes, verify the selected weights sum to one per token. The indices must be `torch.int32` before they reach `npu_moe_init_routing`; a float or int64 tensor is a common silent source of a later dispatch failure.

**2.2 Init-routing scalar contract.** `npu_moe_init_routing` and its V2 form accept `active_expert_range` as a Python `int`, not a 0-dimensional tensor. When the local expert count comes from `torch.sum(expert_map != -1)`, normalize it at the source:

```python
local_num_experts = (
    torch.sum(expert_map != -1).item()
    if expert_map is not None else global_num_experts
)
```

For a custom routing function, convert the result and return immediately; do not let the default softmax/top-k branch process it a second time:

```python
topk_weights, topk_ids = custom_routing_function(
    hidden_states=hidden_states,
    gating_output=router_logits,
    topk=top_k,
    renormalize=renormalize,
)
return topk_weights, topk_ids.to(torch.int32)
```

**2.3 Group list.** `npu_grouped_matmul` consumes either cumulative offsets (`group_list_type=0`) or per-group counts (`group_list_type=1`). Pick one and enforce its invariant:

- cumulative mode: the list is non-decreasing and its last element equals `total_expanded_tokens`; if the operator schema uses an explicit zero prefix, `group_list[0] == 0`; confirm whether the expected length is `num_experts` or `num_experts + 1` from the installed schema rather than assuming it;
- count mode: `sum(group_list) == total_expanded_tokens`, and every entry is non-negative.

A negative count, a non-monotonic cumulative list, or a last offset larger than the output buffer makes the kernel compute an offset past the allocated GM region. That is the out-of-bounds write path that surfaces as an AI Core exception, not as a Python shape error.

**2.4 Empty experts are legal.** The EP decode path calls grouped matmul unconditionally after dispatch; it does not guard on `expert_token_num.sum() > 0`. A rank with zero received tokens, or an expert with zero routed tokens, still participates in the grouped matmul over an empty group. Do not add a local early return: it would remove that rank from the following combine and turn a load imbalance into an HCCL timeout.

Test the boundary deliberately:

```python
# Eager boundary check: force every token to one expert, then force at least
# one expert to zero tokens. Assert the dispatch counts and the GMM output.
assert int(expert_token_num.sum()) == num_tokens * top_k
assert bool((expert_token_num >= 0).all())
# The single-expert result must match the dense reference; the empty expert
# segment must stay untouched and the run must not raise 507015.
```

**2.5 Verify the routing path.** In eager mode, print the dispatch output tuple and check `expert_token_num` (the per-expert received-token count). Its sum must equal `num_tokens * top_k`. Compare the per-expert counts with the group list used by the grouped matmul. If they disagree, fix the list before touching quantization.

## Stage 3 — Align W8A8/W4A8 activation and weight quantization contracts

Classify the checkpoint from the full scheme, not from the number of weight bits:

| Observed scheme | Correct path | Key contract |
| --- | --- | --- |
| Weight int8 per-channel static, activation int8 per-token dynamic | W8A8 | First GMM produces int32; dequantize after the matmul. |
| Weight int4 symmetric static, activation int8 per-token dynamic | W4A8 | Two int4 per int8, per-expert weight scale, per-token activation scale, per-expert float bias, bf16 output. |
| Weight int4 group-wise, activation BF16/FP16 | W4A16 | Int4 weights are antiquantized inside the kernel; activations stay high precision. |
| AWQ-style int4 weight-only | W4A16 | There is no separate `awq` switch; express it as compressed-tensors W4A16. |
| GPTQ-style int4 group-wise weight-only | W4A16 | Group size and packed layout must match the GMM kernel. |
| SmoothQuant-calibrated W8A8 | W8A8 | Smooth scales are a calibration artifact; they do not change the W8A8 operator contract. |

`SmoothQuant`, `AWQ` and `GPTQ` are calibration or packing provenance labels, not runtime operator names. Resolve the actual tuple (weight bits, activation bits, symmetry, group size, packed layout) before selecting a kernel. A label alone is not a contract, and applying a second smoothing transform to a checkpoint that already contains smooth scales corrupts the weights.

**3.1 Gate on the capability, not the literal.** An activation-int8 branch gated by exact `quant_mode == "W8A8"` silently skips W4A8. Test `"a8" in quant_mode` for every path that is genuinely "activation is 8-bit": init-routing dynamic quant, dispatch quant mode and smooth scale, dequant-SwiGLU-quant, MLA `mm_quant`. Keep an exact `== "w8a8"` only for a branch that is genuinely W8A8-only, such as a specific dense or shared-expert down-projection choice.

**3.2 W8A8 flow.** Run `npu_dynamic_quant` on the activation to obtain int8 plus a per-token scale. The first grouped matmul consumes the int8 activation and the per-channel int8 weight and produces int32. Apply the per-token and per-channel scales, then SwiGLU and the second grouped matmul. The weight scale is per output channel; do not broadcast a per-tensor scale across channels.

**3.3 W4A8 flow.** Pack two int4 values per int8 (`pack_factor = 8 // 4 = 2`). Keep a per-expert per-channel weight scale and a per-token activation scale. The W4A8 grouped matmul needs a per-expert float bias auxiliary matrix and `output_dtype=bf16`; this differs from the W8A8 int32-first-stage contract. Apply the learned clip or alpha factor before dynamic quantization to control outliers. Under EP, each device stores only its mapped experts' smooth scales; the first grouped matmul needs the full set, so all-gather the smooth scale across the EP group.

```python
out = torch_npu.npu_grouped_matmul(
    x=[activation_int8],
    weight=[weight_int4_packed],
    bias=per_expert_bias,            # one float auxiliary tensor per expert
    scale=per_expert_weight_scale,   # one per-expert per-channel scale tensor
    per_token_scale=[per_token_scale],
    group_list=expert_token_num,
    group_list_type=1,
    output_dtype=torch.bfloat16,
)
```

**3.4 W4A16 flow.** The int4 GMM antiquant kernel accepts N-dim packed weights. A compressed-tensors checkpoint is commonly K-dim packed, so unpack, transpose and repack in `process_weights_after_loading`; transpose the per-group scale and offset at the same time. If the checkpoint is already N-dim packed, do not repack it again. Int4 group size must divide the K dimension, or the last partial group must be handled explicitly. The activation stays BF16/FP16 and no activation quantization is applied.

**3.5 Shared-expert fallback.** The shared expert sees every token and is the most outlier-sensitive part of the MoE. Keep it at BF16/FP16, or at most W8A8, when a W4A8 routed-expert configuration is new or when the evaluation shows a regression that is localized to shared-expert outputs. A mixed configuration (routed experts W4A8, shared expert BF16, attention W8A8C8) is a legitimate and often safer first deployment.

**3.6 Verify the quantization path is actually active.** A quantized model can run while silently skipping the intended kernels. Check for:

- `npu_dynamic_quant` before the first grouped matmul in the profile;
- an int8 activation tensor and a per-token scale whose length equals the token count;
- a weight scale with one value per output channel per expert, not one scalar for the whole tensor;
- a W4A8 grouped matmul with `output_dtype=bf16` and a non-empty per-expert bias;
- a W4A16 weight tensor in N-dim packed layout and an `antiquant_scale` consistent with the group size.

Then run a per-layer comparison against the BF16 checkpoint: embedding, first MoE layer output, last MoE layer output, final logits. A final-logit-only comparison can hide a wrong expert or a wrong group boundary.

## Stage 4 — Align dummy runs and empty batches across DP and EP ranks

Every rank that belongs to a collective must enter it, even when it has no request. This is a participation invariant, not a performance optimization.

1. **DP idle rank.** A zero-token `execute_model` must still enter `coordinate_batch_across_dp`. Use `_dummy_run(1, skip_gdn_state_update=True)` for a hybrid GDN model so the graph and collective shape are preserved while recurrent state is masked. Never return early on `num_scheduled_tokens == 0`.
2. **EP empty batch.** After dispatch, call grouped matmul and combine unconditionally. Zero received tokens is a valid group; the operator handles the empty group. An early return on one rank removes it from the combine and peers block until timeout.
3. **Profile-run traversal.** A `profile_run` that does not traverse the MC2 dispatch/combine path underestimates the MC2 communication buffer and the graph-capture peak. When the selection function resolves to MC2, run at least one dummy forward through MC2 so the buffer and workspace are allocated before the real service starts.
4. **EPLB during profiling only.** Forced expert balancing is useful to measure a balanced upper bound, but it changes routing and must not be used for an accuracy claim. Disable forced balancing for the accuracy run, and keep dynamic EPLB consistent across ranks.
5. **Timeout is a symptom.** `HCCL_CONNECT_TIMEOUT` may be raised to keep a debugging session alive. It does not repair an asymmetric collective. Before changing it, compare each rank's last launched collective and its process stack.

A minimal two-rank test: DP=2, TP=1, EP=2, one rank idle for ten steps, then one active step. Run with `ASCEND_LAUNCH_BLOCKING=1` and collect a per-rank collective counter. All ranks must enter and leave the same sequence and exit zero. Then repeat with launch blocking disabled and measure whether the idle rank's state and the active rank's output are unchanged.

## Verification and exit criteria

The workflow exits only when all four gates pass on the same code state, checkpoint and trace:

1. **Functional gate.** The MoE output matches the BF16 or reference implementation at the frozen per-layer tolerance; empty-expert and single-expert cases are included.
2. **Collective gate.** Every DP and EP rank enters the same collectives in the same order, including idle and empty-batch steps; no rank exits early; no `EI0006` or `HCCL_BUFFSIZE is too SMALL`.
3. **Quantization gate.** The intended W8A8/W4A8/W4A16 kernels are present in the profile; the activation and weight scales have the expected granularity; a sibling model's scale package is not treated as validated calibration.
4. **Performance gate.** Report prefill TTFT, decode TPOT/ITL and `output_tokens/s` separately, with the communication regime and the quantization scheme stated. A regime change is accepted only if it improves the targeted metric without failing gates 1–3.

Stop and record the first failing gate. Do not promote a result that passes only the performance gate, and do not compare a W4A8 run against a W8A8 baseline from a different checkpoint or tokenizer.
