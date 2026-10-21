---
category: sop
conditions:
  component: spec_decode_dspark_dflash
  framework: vllm-ascend
domain_path: vllm-ascend/speculative-decoding/proposers/dspark-and-dflash
evidence_strength: software_contract
id: unit-vllm-ascend-llm-base-dspark-and-dflash-proposer-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/spec_decode/dspark_proposer.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/spec_decode/dflash_proposer.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/deepseek_v4/dspark.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/qwen3_dspark.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/spec_decode/llm_base_proposer.py
  source_digest: a3bca4c696b247bd6f77734e31cf6628453d7e141b40b8c630625ae316a2e76e
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/spec_decode/dspark_proposer.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: DSpark block drafter (Markov + Confidence head), DFlash parallel diffusion
  drafter, and DynamicSpecScheduler SOP
triggers:
  error_signatures:
  - DSpark on v1 model runner raises ValueError when enable_reduce_sample or lmhead_tensor_parallel_size
    is enabled
  - AscendQwen3DSparkForCausalLM fails to apply rotation_matrix to fc weight or misses
    confidence_head parameters
  - copy_and_expand_dflash_and_dspark_inputs_kernel slot_mapping mismatches multi-group
    KV cache block sizes
  keywords:
  - AscendDSparkProposer
  - AscendDflashProposer
  - dspark_proposer
  - dflash_proposer
  - DeepseekV4DSparkModel
  - AscendQwen3DSparkForCausalLM
  - DSparkMarkovHead
  - DSparkConfidenceHead
  - DynamicSpecScheduler
---
# DSpark block drafter (Markov + Confidence head), DFlash parallel diffusion drafter, and DynamicSpecScheduler SOP

## Scope and preconditions

Use this SOP when deploying or debugging `DSpark` (`AscendDSparkProposer`, `DeepseekV4DSparkModel`, `AscendQwen3DSparkForCausalLM`) and `DFlash` (`AscendDflashProposer`) block speculative decoding proposers on `vllm-ascend` (`vllm_ascend/spec_decode/dspark_proposer.py`, `dflash_proposer.py`, `worker/v2/spec_decode/`).

## Procedure

1. **Understand DFlash vs DSpark block drafting mechanics (`dflash_proposer.py`, `dspark_proposer.py`):**
   - **DFlash (`--speculative-config '{"method": "dflash", "model": "...", "num_speculative_tokens": K}'`):** Uses target hidden states from `dflash_target_layer_ids` to populate context K/V, then runs one parallel query block of length `1 + K` (anchor + mask tokens) prepared on-device by `copy_and_expand_dflash_and_dspark_inputs_kernel` (launched across `min(cdiv(max(num_context, num_query), 256), get_vectorcore_num())` vector cores).
   - **DSpark (`method: "mtp"` or `"dspark"` with DSpark checkpoint):** Stores block-drafter weights under `mtp.*` (DeepSeek-V4-Flash-0731 / Kimi-K3) or standalone `dspark` checkpoint (Qwen3-DSpark). Combines target multi-layer hidden states via `main_proj` with `DSparkMarkovHead` (`markov_w1`, `markov_w2` low-rank transition bias) and `DSparkConfidenceHead` (`proj` over `[hidden_states, markov_embed]`) to emit and dynamically truncate draft blocks via `DynamicSpecScheduler`.
2. **Enforce DSpark V1/V2 configuration invariants (`vllm_ascend/spec_decode/dspark_proposer.py`):**
   - On ModelRunner V1, `AscendDSparkProposer` explicitly forbids:
     - `additional_config.enable_reduce_sample = true` (bypasses Markov-head logit correction);
     - `additional_config.finegrained_tp_config.lmhead_tensor_parallel_size > 0`;
     - `speculative_config.draft_sample_method == "probabilistic"` (use `"greedy"`).
   - For Qwen3-DSpark with rotary-quantized target models (`vllm_ascend/models/qwen3_dspark.py`), `AscendQwen3DSparkForCausalLM.load_weights` multiplies `fc.` weights by the FP32 `rotation_matrix` chunk-by-chunk before materializing parameters.
3. **Maintain per-group slot mappings across heterogeneous KV caches:**
   - On hybrid/DSA models where different KV cache groups have distinct `kernel_block_size` (e.g., Compress-4 vs Compress-128 in DSV4), `AscendDSparkProposer` builds per-group expanded block tables and slot mappings (`_per_group_query_slot_mapping_buffers`, `_per_group_context_slot_mapping_buffers`) so each draft layer indexes its own cache group geometry.

## Verification

- Verify that `AscendDSparkProposer` / `AscendDflashProposer` runs without D2H sync in `copy_and_expand_dflash_and_dspark_inputs_kernel`, achieves positive acceptance rates (`> 60%` on coding/reasoning prompts), and preserves greedy target parity.
