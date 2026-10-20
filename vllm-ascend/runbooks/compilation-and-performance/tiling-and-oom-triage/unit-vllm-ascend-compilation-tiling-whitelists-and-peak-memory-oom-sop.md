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
domain_path: vllm-ascend/runbooks/compilation-and-performance/tiling-and-oom-triage
evidence_strength: software_contract
id: unit-vllm-ascend-compilation-tiling-whitelists-and-peak-memory-oom-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/fused_op_host_tiling_hardcoded_shape_whitelist.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/fia_mla_noquant_group_num_power_of_two.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/mla_graph_mode_queries_per_kv_head_ratio_set.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/cache_compile_fail_dynamic_build_in_compile_region.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/ascendc_custom_op_build_packaging_availability.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/ascendc_custom_op_cross_platform_cann_build.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/performance/moe_token_dispatcher_self_attr_oom_aclgraph.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/performance/moe_prefill_oom_peak_memory.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/performance/profile_run_peak_memory_underestimate_untraversed_path.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/performance/sglang_tp1_oom_disable_nz_weight.md
  source_digest: 5f1c32383d02aad0b265b4bacf27942a7822d2a4a11967188c3be960a39593a3
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/runbooks/compilation/fused_op_host_tiling_hardcoded_shape_whitelist.md
  source_type: cannbot_runbook_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Compilation & performance runbook distillation: Host-tiling shape whitelists,
  FIA group_num power-of-two, cache_compile purity, and peak-memory OOM triage'
triggers:
  error_signatures:
  - aclnnFusedInferAttentionScoreV3 fails with 561002 CheckFeatureMlaNoQuantShape
    when group_num is not a power of two
  - MLA graph compilation rejects num_heads / num_kv_heads ratio outside {1, 2, 4,
    8, 16, 32, 64, 128}
  - torchair cache_compile fails or warms up twice because lazy KV cache or row_idx
    initialization sits inside compiled region
  - ACLGraph capture or MoE prefill OOMs because TokenDispatcher stores transient
    dispatch tensors on self attributes
  keywords:
  - '561002'
  - CheckFeatureMlaNoQuantShape
  - fused_op_host_tiling_hardcoded
  - queries_per_kv_head
  - cache_compile_fail_dynamic_build
  - moe_token_dispatcher_self_attr_oom
  - profile_run_peak_memory
  - moe_prefill_oom
---
# Compilation & performance runbook distillation: Host-tiling shape whitelists, FIA group_num power-of-two, cache_compile purity, and peak-memory OOM triage

## Applicability and symptom scope

Use this SOP when troubleshooting custom operator host-tiling rejections, `aclnnFusedInferAttentionScoreV3` (`561002`) compilation failures, `torchair.inference.cache_compile` invalidations, or unexpected HBM OOM during `profile_run`, prefill, or `ACLGraph` capture in `vllm-ascend`.

## Procedure

1. **Resolve MLA `group_num` power-of-two and `queries_per_kv_head` ratio constraints in FIA and GE Graph mode (`fia_mla_noquant_group_num_power_of_two.md`, `mla_graph_mode_queries_per_kv_head_ratio_set.md`, `fused_op_host_tiling_hardcoded_shape_whitelist.md`).**
   - In non-quantized MLA (`head_dim = 512` + `64` rope), `aclnnFusedInferAttentionScoreV3` (`FIA`) enforces `group_num = num_q_heads // num_kv_heads in {2^k}` (and GE graph lowering enforces `queries_per_kv_head in {1, 2, 4, 8, 16, 32, 64, 128}`). If a model has a non-power-of-two head count per TP rank (e.g., `48` or `24` heads after TP slicing), adjust `tp_size` so local `num_q_heads` is a power of two or pad query heads to the next power-of-two boundary before calling FIA.
   - When adapting new MLA/DSA models (e.g., `GLM-5` where `q_lora_rank != 1536`) that fail inside `mla_prolog` or custom AscendC host tiling, inspect `op_host` for hardcoded shape whitelists (`Hcq == 1536`) and generalize the host tiling check rather than mutating model architecture configs.

2. **Keep `torchair.inference.cache_compile` regions pure of lazy initialization (`cache_compile_fail_dynamic_build_in_compile_region.md`, `graph_mode_stable_input_signature_recompile.md`).**
   - `cache_compile` requires the FX graph traced on the first call to match the steady-state decode graph identically. Any `if self.kv_cache is None:` lazy buffer allocation or one-time `row_idx` table construction inside the compiled `forward_decode` region makes the first trace differ from subsequent steps and breaks cache generation.
   - Move all lazy state/buffer initialization into model setup (`init_kv_cache` / `warmup`) outside the compiled method.

3. **Verify AscendC custom-op target SoC (`-c ascend910b` vs. `ascend910_93`) and symbol registration (`ascendc_custom_op_build_packaging_availability.md`, `ascendc_custom_op_cross_platform_cann_build.md`).**
   - `cann-recipes-infer/ops/ascendc` defaults `-c|--compute-unit` to `ascend910_93` (Atlas A3). Deploying onto Atlas A2 (`Ascend910B`) requires rebuilding with `-c ascend910b` against the target container's exact CANN toolkit version, and verifying registration via `hasattr(torch.ops.custom, op_name)` before model launch.

4. **Eliminate transient tensor retention and `profile_run` peak-memory underestimation (`moe_token_dispatcher_self_attr_oom_aclgraph.md`, `moe_prefill_oom_peak_memory.md`, `profile_run_peak_memory_underestimate_untraversed_path.md`, `sglang_tp1_oom_disable_nz_weight.md`).**
   - Clear transient dispatch metadata (`self.hidden_shape = None`, `self.permuted_probs = None`, `self.tokens_per_expert = None`) at the end of `token_combine` so `ACLGraph` capture does not pin dead intermediate tensors across layers.
   - Ensure `profile_run` traverses the true peak-activation path (including chunked-MoE workspaces, multimodal projector buffers, and MTP draft passes) so KV cache block sizing does not over-allocate HBM and OOM on the first real prefill batch.
   - When single-card (`TP=1`) deployment OOMs during load-time `FRACTAL_NZ` weight casting (`npu_format_cast` temporarily holds both ND and NZ copies in HBM), either free the ND tensor immediately per layer with `torch.npu.empty_cache()` or disable NZ weight casting when HBM headroom is tight.
