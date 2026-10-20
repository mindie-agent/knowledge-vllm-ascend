---
category: sop
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
domain_path: vllm-ascend/optimizations/moe-and-ep/dispatch-combine-routing
evidence_strength: software_contract
id: unit-vllm-ascend-moe-ep-stage-dispatch-mc2-and-double-routing-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-hccl-and-dp-collective-invariants
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/moe_ep_dispatch_path_by_stage.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/moe_ep_decode_graphmode_dispatch.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/moe_ep_double_routing_skeleton.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/moe_zero_expert_copy_expert_num.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/fused_allgather_ep_moe_256expert_deepseek_v3r1.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/moe_dispatch_combine_v2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/shmem_dispatch_combine.md
  source_digest: 08ff5b4d9dc9a16af2d4ad475f2f32630e1cb90ae7198fae221cd513e59ba272
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/moe_ep_dispatch_path_by_stage.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Stage-aware MoE Expert Parallelism dispatch selection: double-routing AllToAll,
  MC2 fused dispatch-combine, SHMEM, and 256-expert AllGather-EP'
triggers:
  error_signatures:
  - Double-routing AllToAllV fails TorchAir/ACLGraph capture due to host-dependent
    split_sizes D2H sync
  - MoE prefill OOMs when AllGather-EP replicates full token activations across all
    EP ranks
  - Zero/identity experts in LongCat-Flash trigger redundant AllToAll traffic when
    not mapped via copy_expert_num
  keywords:
  - moe_ep_dispatch_path_by_stage
  - moe_ep_double_routing
  - npu_moe_distribute_dispatch_v2
  - npu_moe_distribute_combine_v2
  - copy_expert_num
  - shmem_dispatch_combine
  - fused_allgather_ep_moe
---
# Stage-aware MoE Expert Parallelism dispatch selection: double-routing AllToAll, MC2 fused dispatch-combine, SHMEM, and 256-expert AllGather-EP

## Applicability and symptom scope

Use this SOP when configuring or optimizing MoE token dispatch and combine paths across Prefill and Decode stages in `vllm-ascend`.

## Procedure

1. **Switch MoE EP dispatch strategy by stage and token volume (`moe_ep_dispatch_path_by_stage.md`, `moe_ep_decode_graphmode_dispatch.md`).**
   - **Prefill (high token count, eager mode):** Use the **8-step Double-Routing `AllToAllV` skeleton** (`moe_ep_double_routing_skeleton.md`):
     1. Compute local router top-k and `npu_moe_init_routing_v2` to sort tokens by expert ID;
     2. Exchange per-rank token counts via `dist.all_to_all_single` (metadata exchange);
     3. Permute and dispatch token hidden states (and dynamic quantization scales) via `dist.all_to_all_single` so each EP rank receives *only* tokens routed to its local experts ($1/\text{ep\_size}$ memory and routing work compared to `AllGather`);
     4. Secondary local sort by local expert ID -> `npu_grouped_matmul` (GMM) -> `npu_swiglu` -> GMM down-projection -> inverse `all_to_all_single` -> `npu_moe_finalize_routing`.
   - **Decode (low token count, graph mode):** Double-routing requires host-visible `split_sizes` (`D2H` copy), which breaks `ACLGraph` / `TorchAir` capture. In graph-mode decode, switch to **MC2 fused `npu_moe_distribute_dispatch_v2` + `npu_moe_distribute_combine_v2`** (or SHMEM-based `shmem_dispatch_combine` on Atlas A3), which performs device-side routing, quantization, and RDMA/SDMA transfer inside a graph-capturable operator pair on a dedicated HCCL domain.

2. **Handle zero/identity experts inside MC2 via `copy_expert_num` (`moe_zero_expert_copy_expert_num.md`).**
   - For MoE architectures with zero-compute / identity pass-through experts (such as `LongCat-Flash`), pass `copy_expert_num` to `npu_moe_distribute_dispatch_v2` / `combine_v2` so identity-routed tokens are handled locally without cross-rank communication.

3. **Apply the 4-op Fused AllGather-EP chain for 256-expert medium-batch decode when MC2 is unavailable (`fused_allgather_ep_moe_256expert_deepseek_v3r1.md`).**
   - For 256-expert DeepSeek-V3/R1 on smaller EP topologies where AllGather-EP is selected, fuse routing and expert execution into `npu_moe_init_routing_v2` -> `npu_grouped_matmul` (`w13`) -> `npu_swiglu` -> `npu_grouped_matmul` (`w2` cast to `FRACTAL_NZ`) -> `npu_moe_finalize_routing`, ensuring `w2` is in `FRACTAL_NZ` format and scales match the finalize-routing contract.
