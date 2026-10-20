---
category: workflow
conditions:
  framework: vllm-ascend
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - vllm-ascend-idle-dp
domain_path: vllm-ascend/optimizations/moe-and-ep/eplb-and-multistream
evidence_strength: software_contract
id: unit-vllm-ascend-moe-shared-expert-dual-stream-and-eplb-workflow
inherits_from:
- unit-vllm-ascend-moe-ep-stage-dispatch-mc2-and-double-routing-sop
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/moe_shared_expert_decode_dual_stream.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/cases/moe_multi_stream_limit_core_num.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/perfect_eplb_profiling_upper_bound.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/precision/eplb_remap_drops_independent_scale_tensor.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/cases/afd_attn_ffn_disaggregation.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-multi-stream/SKILL.md
  source_digest: 33224d8e208f13ecc390286c06a31238641374caa6294f037755481e68417d9f
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/moe_shared_expert_decode_dual_stream.md
  source_type: cannbot_workflow_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: MoE shared-expert dual-stream overlap, core-limit deadlock prevention, EPLB
  scale-tensor remapping, and AFD disaggregation workflow
triggers:
  error_signatures:
  - Dual-stream routed-expert dispatch/combine and shared-expert matmul hang or serialize
    due to full AICore occupation
  - Enabling EPLB expert weight remapping degrades W8A8/W4A8 accuracy because per-expert
    scale tensors were not remapped
  - MoE decode TPOT remains communication-bound despite EP scaling
  keywords:
  - moe_shared_expert_decode_dual_stream
  - moe_multi_stream_limit_core_num
  - limit_core_num
  - perfect_eplb
  - eplb_remap_drops_independent_scale
  - afd_attn_ffn_disaggregation
---
# MoE shared-expert dual-stream overlap, core-limit deadlock prevention, EPLB scale-tensor remapping, and AFD disaggregation workflow

## Objective

Maximize MoE compute-communication overlap and expert load balance in `vllm-ascend` without triggering hardware core-contention deadlocks or quantized scale-expert misalignment.

## Workflow Steps

1. **Diagnose theoretical expert load-balance ceiling via column-major round-robin EPLB probing (`perfect_eplb_profiling_upper_bound.md`).**
   - Before deploying dynamic EPLB, quantify the latency gap caused by expert routing skew by overriding `topk_ids` in profiling mode with a synthetic balanced assignment (`torch.arange(num_tokens * top_k, device="npu").reshape(top_k, num_tokens).T % num_experts`).
   - Construct the synthetic matrix **column-major** (`reshape(top_k, num_tokens).T`) so no single token is assigned duplicate expert IDs within its `top_k` row. Keep this override strictly behind a profiling-only flag.

2. **Remap every companion quantization scale/offset tensor alongside expert weights during EPLB updates (`eplb_remap_drops_independent_scale_tensor.md`).**
   - When EPLB swaps or replicates physical expert slots across EP ranks, remapping only `w13_weight` and `w2_weight` while leaving `w13_weight_scale`, `w2_weight_scale`, `w13_input_scale`, or `quant_bias` in the original expert order silently multiplies expert $E_i$'s INT8/INT4 weights by expert $E_j$'s dequantization scale.
   - **Gate:** Assert that every tensor indexed by `expert_id` is included in the EPLB permutation list and verify logit cosine similarity ($\ge 0.999$) against non-remapped EP before benchmarking.

3. **Overlap Shared-Expert compute with Routed-Expert Dispatch/GMM/Combine on a secondary stream with explicit AICore/Vector partitioning (`moe_shared_expert_decode_dual_stream.md`, `moe_multi_stream_limit_core_num.md`).**
   - Place the routed-expert chain (`router` -> `dispatch` -> `routed_GMM` -> `combine`) on the primary stream and the `shared_expert` MLP (`gate_up_proj` -> `act` -> `down_proj`) on a secondary stream (`torchair.scope.npu_stream_switch` in graph mode or `torch.npu.stream` in eager mode), guarded by `wait_tensor` data dependencies.
   - **Deadlock & Contention Prevention:** Because MC2 `dispatch`/`combine` uses AIV vector cores for local reduce/unpacking, letting the concurrent shared-expert stream claim all AIC/AIV cores starves the communication kernel or causes hardware stream deadlock. Partition core budgets across the two streams via `torchair.scope.limit_core_num(aicore_num, vector_num)` (e.g., splitting total Cube/Vector cores proportionally to FLOPs).

4. **Evaluate Attention-FFN Disaggregation (AFD) when single-card core partitioning limits both streams (`afd_attn_ffn_disaggregation.md`).**
   - For ultra-large MoE models in decode where splitting cores on the same NPU elongates both Attention/Shared-Expert and Routed-MoE kernels, disaggregate Attention and MoE FFN onto separate rank groups connected via pipelined micro-batch `Send`/`Recv` over HCCS/SDMA.
