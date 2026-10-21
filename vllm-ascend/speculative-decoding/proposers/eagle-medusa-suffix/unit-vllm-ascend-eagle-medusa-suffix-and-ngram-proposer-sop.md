---
category: sop
conditions:
  component: spec_decode_eagle_medusa_suffix_ngram
  framework: vllm-ascend
domain_path: vllm-ascend/speculative-decoding/proposers/eagle-medusa-suffix
evidence_strength: software_contract
id: unit-vllm-ascend-eagle-medusa-suffix-and-ngram-proposer-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/spec_decode/eagle_proposer.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/models/llama_eagle3.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/spec_decode/medusa_proposer.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/spec_decode/suffix_proposer.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/spec_decode/ngram_proposer_npu.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/speculative_decoding.md
  source_digest: aa5994f69f3050dd0a7cd374676857ddb787153083d7f6f9b3566cd3e4dc3474
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/spec_decode/eagle_proposer.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: EAGLE / EAGLE3, Medusa, SuffixDecoding, and NPU Ngram speculative proposer
  SOP
triggers:
  error_signatures:
  - AscendEagleProposer fails on auxiliary hidden state extraction or PP eagle3 layer
    wiring
  - Eagle3 VL or QuaRot target model mismatches draft embedding or fc projection
  - NgramProposer CPU matching stalls decode loop when NPU ngram kernel can be used
  keywords:
  - AscendEagleProposer
  - eagle_proposer
  - llama_eagle3
  - medusa_proposer
  - suffix_proposer
  - ngram_proposer_npu
  - extract_hidden_states_proposer
  - rejection_sampler
---
# EAGLE / EAGLE3, Medusa, SuffixDecoding, and NPU Ngram speculative proposer SOP

## Scope and preconditions

Use this SOP when configuring or debugging `eagle` / `eagle3` (`vllm_ascend/spec_decode/eagle_proposer.py`, `models/llama_eagle3.py`), `medusa` (`medusa_proposer.py`), `suffix` (`suffix_proposer.py`), or `ngram` (`ngram_proposer.py`, `ngram_proposer_npu.py`) speculative decoding on `vllm-ascend`.

## Procedure

1. **Configure EAGLE / EAGLE3 (`AscendEagleProposer`, `vllm_ascend/models/llama_eagle3.py`):**
   - `EAGLE3` concatenates hidden states from multiple target layers (configured via `eagle3_aux_hidden_state_layers` and patched for Pipeline Parallelism by `patch_eagle3_pp_aux.py`) and projects them through `fc` before the draft decoder layer.
   - When the target model uses QuaRot / Hadamard rotation (`rotation_path` in `llama_eagle3.py`), `get_rotation_matrix` rotates the draft `fc` weights at load time so target rotated hidden states align with the unrotated Eagle3 draft head.
   - When `ACLGraph` (`FULL_DECODE_ONLY` or `PIECEWISE`) is enabled on the Eagle proposer (`worker/v2/spec_decode/eagle/aclgraph.py` or V1 `eagle_proposer.py`), capture the draft loop at batch buckets matching the target runner's request count.
2. **Use model-free proposers (`SuffixDecoding` and `NgramProposerNPU`) for repetitive/agentic workloads:**
   - **SuffixDecoding (`suffix_proposer.py`, `docs/source/tutorials/features/suffix_speculative_decoding.md`):** Maintains a suffix tree over prompt and historical completions (`suffix_decoding_max_tree_depth`, `suffix_decoding_max_spec_factor`)—ideal for code editing, RAG, and multi-turn agent loops without extra HBM draft weights.
   - **NPU N-gram (`ngram_proposer_npu.py`):** Offloads prompt n-gram pattern matching (`prompt_lookup_num_tokens`, `max_matching_ngram_size`, `min_matching_ngram_size`) onto Ascend NPU vector cores to avoid host CPU string/array scans on long prompts.
3. **Verify device-side rejection sampling (`vllm_ascend/sample/rejection_sampler.py`):**
   - `AscendRejectionSampler` (warmed up via `rejection_sampler_triton_warmup.py`) verifies $K$ draft tokens in a single target forward step (`1 + K` tokens per request) and writes `bonus_token_ids` and `num_accepted_tokens` directly on NPU.

## Verification

- Compare speculative decoding (`temperature=0`, `num_speculative_tokens=3`) against non-speculative greedy decoding on the same prompt set and verify exact output token parity alongside reduced end-to-end TPOT.
