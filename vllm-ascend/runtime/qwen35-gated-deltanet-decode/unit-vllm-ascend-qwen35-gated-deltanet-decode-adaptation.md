---
category: sop
conditions:
  model_repository: Qwen/Qwen3.5-0.8B
  model_revision: 2fc06364715b967f1860aea9cf38778875588b17
  source0_repository: https://github.com/vllm-project/vllm-ascend
  source0_revision: a78a00e0b1b618f26a3bfce2519055b40cf4bf1e
content_digest: sha256:33661ebd978f03c8355b1728e082fb8937bb5bf245bea38557d069def80ef14c
domain_path: vllm-ascend/runtime/qwen35-gated-deltanet-decode
evidence_strength: single_observation
id: unit-vllm-ascend-qwen35-gated-deltanet-decode-adaptation
inherits_from: []
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/0f289fa2a8a381a644a5f329af61715a73ec2aec/vllm_ascend/patch/worker/patch_qwen3_5.py
  - https://github.com/vllm-project/vllm-ascend/blob/0f289fa2a8a381a644a5f329af61715a73ec2aec/vllm_ascend/patch/worker/__init__.py
  - https://github.com/vllm-project/vllm-ascend/blob/0f289fa2a8a381a644a5f329af61715a73ec2aec/vllm_ascend/patch/__init__.py
  - https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/models/qwen3_5/modeling_qwen3_5.py
  experiment_status: not_run_source_only
  source_digest: d48fbf9acefff670b66fc9941de53b725607d8239153bad1ee7b79fdacd56bdd
  source_locator: vllm-project/vllm-ascend@0f289fa2a8a381a644a5f329af61715a73ec2aec:vllm_ascend/patch/worker/patch_qwen3_5.py
  source_type: public_source
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
target_domain: vllm-ascend
tier: L2
title: Qwen3.5 GatedDeltaNet decode adaptation in vLLM-Ascend (historical PR7109)
triggers:
  error_signatures:
  - torch_npu.npu_recurrent_gated_delta_rule does not support ssm_state inputs in
    float32 format
  - Qwen3_5GatedDeltaNet reuses Qwen3NextGatedDeltaNet._forward_core but AscendC ops
    lack float32 ssm_state support
  - worker import is version-gated by not vllm_version_is("v0.16.0")
  keywords:
  - qwen3.5
  - qwen3_5
  - gated deltanet
  - linear attention
  - ssm_state
  - vllm-ascend
  - text-only prefill
  - recurrent decode
  - chunk_gated_delta_rule
---
This unit records the historical Dev adaptation that made Qwen3.5's gated-delta-net (GDN) linear-attention layers run through vLLM-Ascend's decode path. It is source-derived procedural knowledge at L2, not an execution result.

Historical delta. The before views at task-start revision a78a00e0b1b618f26a3bfce2519055b40cf4bf1e show no Qwen3.5 entry in the worker import list and no Qwen3.5 item in the patch index. The after revision 0f289fa2a8a381a644a5f329af61715a73ec2aec adds vllm_ascend/patch/worker/patch_qwen3_5.py and imports it from vllm_ascend/patch/worker/__init__.py under this exact guard:

    if not vllm_version_is("v0.16.0"):
        import vllm_ascend.patch.worker.patch_qwen3_5  # noqa

The after patch index documents item 17: Qwen3_5GatedDeltaNet reuses Qwen3NextGatedDeltaNet._forward_core, but the AscendC operators used by the Qwen3Next path do not support a float32 ssm_state; the patch replaces the core with fused FLA/Triton operators. The patch file ends by assigning AscendQwen3_5GatedDeltaNet._forward_core onto Qwen3_5GatedDeltaNet._forward_core. This is a runtime monkeypatch, not a new model class registered with vLLM. Applicability is gated by the exact task-start model revision and source revision recorded in the original task condition; the after fix revision is evidence only.

Why a separate core. The patch header states that torch_npu.npu_recurrent_gated_delta_rule does not accept float32 ssm_state, so the Qwen3.5 implementation keeps a local _forward_core until that operator supports float32 state. The replacement combines vLLM's chunk_gated_delta_rule and fused_recurrent_gated_delta_rule with vLLM-Ascend's fused_gdn_gating_patch and fused_sigmoid_gating_delta_rule_update. This is source evidence of a compatibility implementation using fused FLA/Triton operators, not evidence that float32 NPU execution or numerical equivalence was achieved.

What _forward_core does. The model's custom op calls it after hidden-state projections produce mixed_qkv, b, a and an output buffer. It:
- reads get_forward_context().attn_metadata, indexes it by self.prefix and asserts GDNAttentionMetadata; takes has_initial_state, spec/non-spec query start locations, token indices, state-index tensors, num_actual_tokens, num_accepted_tokens and num_prefills/num_decodes;
- reads self.kv_cache[forward_context.virtual_engine]; kv_cache[0] is read as conv_state after a transpose, and kv_cache[1] is ssm_state;
- truncates mixed_qkv/b/a to num_actual_tokens unless enable_sp() is on;
- splits tokens into speculative and non-speculative streams using spec_token_indx/non_spec_token_indx when spec_sequence_masks is present;
- runs the causal convolution per stream: speculative tokens use causal_conv1d_update with spec_state_indices_tensor[:,0][:num_spec_decodes], num_accepted_tokens, spec_query_start_loc and max_query_len from spec_state_indices_tensor.size(-1); prefill tokens use torch.ops._C_ascend.causal_conv1d_fn with conv_state, has_initial_state, non_spec_state_indices_tensor, non_spec_query_start_loc and PAD_SLOT_ID; non-spec decode tokens use causal_conv1d_update with validate_data=True;
- computes g and beta with fused_gdn_gating_patch(A_log, a, b, dt_bias) when there is a prefill or speculative stream; for a pure non-spec decode it instead lets fused_sigmoid_gating_delta_rule_update fuse gating and the recurrent update (softplus_beta=1.0, softplus_threshold=20.0);
- for prefill, gathers initial_state = ssm_state[non_spec_state_indices_tensor].contiguous() and zeroes rows whose has_initial_state is false: initial_state[~ has_initial_state, ...] = 0. It then calls chunk_gated_delta_rule with output_final_state=True, cu_seqlens=non_spec_query_start_loc, head_first=False and use_qk_l2norm_in_kernel=True, and writes last_recurrent_state back into ssm_state at those indices after casting to ssm_state.dtype;
- when gating has been computed and there are non-spec decode tokens, calls fused_recurrent_gated_delta_rule with initial_state=ssm_state, inplace_final_state=True, cu_seqlens=non_spec_query_start_loc[:num_decodes+1], ssm_state_indices=non_spec_state_indices_tensor and qk L2 normalization in kernel;
- for speculative decode, calls the same recurrent rule with spec_query_start_loc[:num_spec_decodes+1], spec_state_indices_tensor and num_accepted_tokens;
- merges spec/non-spec outputs into the provided core_attn_out with index_copy and SP-aware slicing, then calls maybe_save_kv_layer_to_connector("", []).

The prefill predicate is now exact from the corrected public view: rows without an initial state are zeroed before the chunk rule. The earlier masked identifier was a public Python name (has_initial_state), not a private value. No source loss remains in patch_qwen3_5.py.

Reference semantics at official Transformers revision a08ace4bbd97e721c98751deec37d87b026acadc:
- Each decoder layer is either linear_attention (Qwen3_5GatedDeltaNet) or full_attention (Qwen3_5Attention). When layer_types is absent, the config creates one full-attention layer every fourth layer (full_attention_interval=4), i.e. a 3:1 linear:full pattern; the actual Qwen3.5-0.8B layer_types come from its checkpoint config and are not in these views.
- GDN computes beta=sigmoid(b) and g=-exp(A_log)*softplus(a+dt_bias), applies L2 normalization to q and k inside the kernel, repeats q/k when value heads outnumber key heads, applies a gated RMSNorm (variance in float32, weight multiply, then multiply by silu(gate) - norm before gate), and projects with out_proj. The depthwise causal conv uses kernel linear_conv_kernel_dim with kernel-1 left context.
- The cache is hybrid: conv_state plus recurrent_state per GDN layer, and ordinary KV cache per full-attention layer. A single-token cached decode uses causal_conv1d_update plus the recurrent delta rule. A multi-token continuation prepends conv_state to the new conv input, computes and left-pads the new conv state, uses the chunk delta rule with the cached recurrent state, and drops the prepended positions from the output. Both paths update conv and recurrent states.
- Full attention projects q and a sigmoid gate from one q_proj, applies per-head q_norm/k_norm, partial RoPE (default partial_rotary_factor=0.25; only the first int(head_dim*factor) channels rotate, the rest pass through), updates the KV cache, applies attention, multiplies by sigmoid(gate), then o_proj.
- The model is natively multimodal, but text-only inputs use four position-id rows whose three mrope rows are copies of the text positions; with no vision tokens, interleaved mrope reduces to ordinary RoPE. In the conditional-generation path, the no-vision branch sets rope_deltas to zeros and clears pixel_values/pixel_values_videos after the first iteration.
- Full-attention layers use create_causal_mask; linear-attention layers use create_recurrent_attention_mask. They are distinct mask objects selected by config.layer_types.
- The official tests encode two useful invariants: padded and padding-free logits must match, and the first position of a multi-token cached continuation must match the same token fed as a single-token cached decode. The test file marks quantized cache as incompatible with this hybrid linear-attention cache.

Integration and decision boundaries.
- Registration is import-time and version-gated. The guard is exactly "not v0.16.0"; the reason for that boundary is not stated in the admitted views. The patch also assumes vllm.model_executor.models.qwen3_5.Qwen3_5GatedDeltaNet and the GDNAttentionMetadata fields exist; a different task-start revision can fail at import or attribute access before any model runs.
- The patch needs Triton modules and the Ascend _C_ascend.causal_conv1d_fn custom op. Availability is a runtime precondition; the worker __init__ only guards patch_triton with HAS_TRITON, not this import.
- enable_sp() changes token truncation and merge behavior; SP and non-SP paths need separate validation.
- The patch file changes only _forward_core. It does not implement checkpoint loading, config mapping, tokenizer behavior, scheduler integration, or server behavior. Complete checkpoint loading (the task condition records 488 weight-index keys and a 1,746,942,600-byte full-weight file) must be evidenced by the execution lane, not inferred from this patch.
- For the text-only lane, the official tests exercise Qwen3_5ForCausalLM with the text config; Qwen3_5ForConditionalGeneration is the full multimodal conditional-generation class. The test shows AutoModelForCausalLM unwraps the composite config to Qwen3_5TextConfig. Ascend-side model selection is upstream of this patch.
- The after eagle_proposer.py change adds Qwen3_5ForConditionalGeneration to the multimodal model-name list used to set image_token_index, plus one further withheld entry. It is a service/speculative-decoding compatibility change, not part of text-only GDN decode; MTP/speculative acceptance is out of scope.
- Precision: reference execution is CPU float32; the patch's rationale is float32 ssm_state compatibility, and the reference gating/norm arithmetic upcasts to float32. No NPU dtype, tolerance, throughput, or numerical result is established by these views.

Stage support. Registration: source-supported. Load: not supported by this patch; only an import dependency and checkpoint metadata are visible. Prefill: source-supported after the corrected view restores has_initial_state; no source-loss gap remains, but no NPU execution or numerical comparison exists. Decode: source-supported for single-token recurrent and multi-token chunk continuation, including state indexing and speculative branches. Precision: semantics visible, measurement absent. Service: only the registration hook and the eagle model-name list are visible; no engine/server acceptance.

Remaining source losses. After the public-name correction, the admitted source-role set has 10 remaining redacted spans / 293 bytes: view 0 (1/36), view 2 (2/68), view 3 (1/36), view 6 (3/102), and corrected view 9 (3/51). Views 1, 4, 5, 7, 8, 10, 11, 12 and 13 are exact; views 0, 2, 3, 6 and 9 remain lossy. View 9 is a .reference.txt with three non-mechanism comment/docstring spans and cannot be called complete executable source. The 16 restored public names are not mechanism losses. The old 26-span/794-byte accounting is obsolete. The admitted union is 14 source roles plus 4 corrected view files (18 unique view files; 4 corrected files supersede the old lossy files for 5/9/10/12), not 18 independent observations and not a new family.

Minimum real evidence before calling this adapted. The independent execution lane must: verify the full-weight checkpoint hash and that all 488 index keys load with no unexpected text-model keys; run text-only prefill and short greedy decode on the historical NPU stack; compare teacher-forced logits and per-step greedy tokens against the CPU float32 official reference using a pre-registered tolerance not tuned from the target; check conv/recurrent state continuity between prefill and decode and single-token vs multi-token equivalence, including both has_initial_state branches; report the runtime's observed ssm_state/conv_state dtypes and SP on/off without treating absent CANN/dtype values as task-start conditions; and record the actual vLLM/vLLM-Ascend environment. Until those exist, this remains an L2 source mechanism, not a full 0-day acceptance.
