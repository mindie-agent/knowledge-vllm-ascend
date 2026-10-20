---
schema: mindie-graded-unit/1
id: unit-vllm-ascend-kimi-k3-model-bringup-and-hybrid-cache-workflow
revision: 1
content_digest: sha256:e9526314c744efd2d901e293efaf8d350f61cea129660e91186a1f593c50b351
tier: L3
category: workflow
domain_path: vllm-ascend/models/kimi-k3-hybrid
evidence_strength: reproduced_empirical
title: 'Kimi K3 hybrid bring-up workflow: loader, hybrid cache, eager parity, ACLGraph'
conditions:
  framework: vllm-ascend
  model_family:
  - kimi-k3
  - hybrid-kda-gqa
  - mamba-hybrid
inherits_from:
- unit-vllm-ascend-kimi-k3-kda-mamba-hybrid-attention-axioms
- unit-vllm-ascend-kimi-k3-nan-inf-quarot-and-norm-precision-sop
- unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop
provenance:
  source_type: session_distill
  source_locator: docs/evidence/k3-hybrid-transcript-provenance.json#s-k3-mrv2-hybrid-cache
  source_digest: cdd5c523ecadd81544bc3d12deba68bdc7889c96cbd10dbb3b52a88deaee0655
  evidence_refs:
  - docs/evidence/k3-hybrid-transcript-provenance.json
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/c1818fdb07ecc530a5b51bd21dc5238d1ab83c88889708b10772066b4a6b6715/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/ea7876a55b7e5eb30b5d33a44b89ac1216c10bc4f3db2ad45b3ff3255bebd020/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/061f6320b291045cfccfba33e62a2f06f1f4175c6ce9ca3e3d04bc0ef30ad530/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/0b56642daab083ed631393983685973b478dc935583b844114d32da7b0ed3c0a/index.md
  supersedes: []
triggers:
  error_signatures:
  - model loads but the first forward fails on a missing KDA or causal-conv operator
  - a hybrid cache manager rejects the mixed MLA/GQA/Mamba cache specification
  - decode output is correct on a cold request and diverges on a warm prefix-cache request
  - eager logits are finite but the first ACLGraph replay produces NaN or repeated '!'
  - decode latency is dominated by host synchronization or repeated graph re-capture
  keywords:
  - Kimi K3
  - hybrid cache
  - MambaCacheManager
  - KDA
  - GQA
  - TP
  - EP
  - sharding
  - eager parity
  - ACLGraph
  - TorchAir
  - piecewise
  - prefix cache
  - chunked prefill
  - state carryover
  - decode latency
---

# Kimi K3 hybrid bring-up workflow: loader, hybrid cache, eager parity, ACLGraph

## Trigger and scope

Use this workflow when bringing up a Kimi K3-family checkpoint on `vllm-ascend` where some layers use KDA linear-recurrent attention with a causal-conv state, some use grouped-query attention with a paged KV cache, and the feed-forward block may be a fused or expert-parallel MoE. It covers the path from a loaded checkpoint to a captured decode graph, on Ascend 910B, 910_93, or 950. It is not a tuning guide for a model that has already passed eager parity, and it does not replace the NaN/Inf bisection SOP when the first failure is already known to be a single operator.

The workflow is deliberately staged. A failure at a later stage is usually a defect that was hidden by an earlier shortcut: a cache manager that silently coerced a per-layer spec, a state slot that was never zeroed, or a graph capture that specialized on the warmup's empty metadata. Do not advance a stage because the service became healthy; advance only when that stage's own evidence is recorded.

## Facts to freeze before profiling

Record these before the first run and keep them identical across the four stages. A changed value is a new experiment, not a rerun.

- **Source and stack:** exact `vllm-ascend` and vLLM revisions, image or container digest, CANN version, `torch`/`torch_npu` version, custom-operator bundle hash, and whether the build registered the KDA and causal-conv operators at all.
- **Checkpoint:** model revision, weight shard count and hash, quantization descriptor, and whether QuaRot was applied by the loader. A shared `embedding`/`lm_head` must be rotated or anti-rotated under one flag.
- **Topology:** `tp`, `dp`, `ep`, `pp`, `dcp`, node count, HCCL configuration, and the exact rank-to-device map. Unequal TP between prefill and decode is a separate case and must be declared.
- **Model geometry:** `layer_types`, full-attention interval, `num_attention_heads`, `num_key_value_heads`, `head_dim`, KDA head count and state size, conv kernel width, MoE expert count and top-k, and the maximum sequence length.
- **Cache geometry:** KV dtype and page size, attention block size (the observed K3 v0.26 path used 384 tokens to align with the Mamba state page), `state_len`, speculative state-pool size, block-table length, prefix-cache on/off, chunked-prefill size, `max_num_batched_tokens`, and the cache layout name (`BLHNC`/`BSND`) with its first-axis stride and storage offset.
- **Graph mode:** eager, piecewise, or full decode; capture buckets; TorchAir/`npugraph_ex` flags; whether `enforce_eager` is set; and whether the prefill path is inside or outside the captured region.
- **Decoding:** greedy, temperature 0, seed, `max_tokens`, stop tokens, and whether speculative decoding is enabled with its draft length `K`.
- **Workload:** prompt lengths 128, 4096, 16384; batch sizes 1 and 16; cold, warm-prefix, and reset runs; at least one mixed prefill-decode batch; and the page-boundary neighborhood 383/384/385/386.
- **Measurement:** warmup count, measurement window, clock source (and clock skew across nodes), tolerance, and the frozen eager/fp32 golden oracle. Do not choose tolerance after seeing the candidate.

## Stage 1 — Config, weight loader, and TP/EP sharding alignment

The goal is to prove that every parameter has one owner and that every fused projection is split exactly once. A model can load successfully and still be sharded incorrectly; the first symptom is often a runtime parameter error in a KDA helper or a decode result that changes with TP degree.

1. **Resolve the model config before loading.** Confirm `layer_types` marks the expected KDA and full-attention layers, the full-attention interval matches the checkpoint, and the KDA state size, conv width, head counts, and MoE expert count are the checkpoint's values rather than template defaults. Freeze whether the model is text-only or multimodal; a multimodal wrapper on a text-only test changes the input path and is not the same experiment.
2. **Build the shard map.** For QKV and gate projections, check that the output dimension is divisible by `tp` and that `num_kv_heads` is divisible by `tp` or explicitly replicated. For KDA, shard the local heads and their recurrent state together; a rank that owns a head's projection but not that head's state is a silent cross-rank corruption. Shard the conv state by the same channel ownership as the conv weights. For EP, check that the expert count is divisible by `ep` and that the router and shared-expert weights have one declared owner.
3. **Verify the loader contract for packed KDA/conv weights.** Packed q/k/v conv weights and derived buffers must be rebuilt after every load path: initial load, dummy load, checkpoint layerwise reload, v1/v2 reload, and direct kernel-format update. A direct `param.copy_()` does not trigger a packing hook, and a layerwise finalize can re-register a derived buffer with `persistent=True`, so the buffer enters `state_dict()` and a later sharded reload sees a stale or duplicated copy. Register derived buffers with `persistent=False`, and assert that they appear in `named_buffers()` but not in `state_dict()`. A capture-time `.to()` or an offload reload can still change the address; treat those as stage 4 inputs, not loader successes.
4. **Check unequal-TP prefill/decode explicitly.** If prefill and decode use different TP degrees, the KDA state split is not implied by either side's local heads. Map the prefill state to the decode state with an explicit index map and verify the transfer on one request before enabling a disaggregated service. A KDA helper launch failure on the prefill side can be an old build with a zero operator-registration count; clean and rebuild before attributing it to the sharding map.
5. **Gate the stage.** All shards load with no missing or unexpected keys; the derived-buffer/state-dict assertion holds; a dummy eager forward produces finite activations; and each KDA layer's recurrent and conv state shapes equal the declared local-head/channel layout. Record the exact rank map and the operator bundle hash. If the model only starts after cleaning an old build, that is an environment fix, not a sharding result.

## Stage 2 — Hybrid KV + recurrent/conv state manager wiring

The goal is to prove that the KV cache, the conv state, and the recurrent state are allocated, zeroed, reused, and transferred as three separate planes, even when the manager groups them under one page size.

1. **Declare one cache spec per plane.** The paged KV cache is block-indexed. The conv state is a per-request ring. The recurrent state is a per-request accumulator. Wire the planes through the hybrid allocator (`MambaCacheManager` or the project's Ascend wrapper) rather than through a single KV block table. If the manager requires uniform pages, group them only after proving that the grouping preserves each plane's own stride and that the block table has an entry for every state slot. Do not reinterpret a state slot as a KV block, and do not let a GQA page packing scheme overwrite an MLA or Mamba page. In the observed K3 path, an independent GQA physical page was replaced by a 4:1 parent-page/subslot packing inside the validated MLA plus Mamba layout to preserve first-axis contiguity; that is a layout decision, not a license to alias state storage.
2. **Allocate and zero every state slot at admission.** The state pool length must cover the speculative state length, not only the current batch. A block table that is short by the speculative slot count is a hard failure. Zero new conv and recurrent slots, including the tail slots added when the pool grows from a shorter to a longer speculative length; an uninitialized tail is a valid-looking but arbitrary state. On preemption, free the request's slot and zero or invalidate it before reuse. A KV block reference count is not a state-slot reference count.
3. **Make prefix caching state-aware.** For the first bring-up, disable prefix caching and prove cold correctness. When enabling it, snapshot and restore the KV pages and the recurrent/conv state at the same boundary; a KV-only hit, a partial-tail hit, or a copy-on-write target that leaves the state slot at the old position is corrupt. The observed K3 failures included a 98.73% prefix hit that still emitted repeated `!` and an exact-boundary recurrent-state admission gap. A 32-token cold/warm/reset sequence is the minimum gate; a long-prefix repeat is the real one.
4. **Carry state across chunked prefill.** For chunk `k` of a request, the conv and recurrent state written by chunk `k-1` is the input to chunk `k`. The `query_start_loc` is per chunk, not the full prompt's cumulative length. Do not reset the state between chunks, and do not pass a chunk-local slot as a global block index. Verify a two-chunk prefill against a one-shot prefill of the same tokens before trusting a long-context result.
5. **Gate the stage.** A single request passes cold, warm-prefix, and reset runs with identical greedy output; a 32-token cache-hit sequence reports the expected hit counts and no divergence; a 128K-class request completes with zero NaN/Inf in the checked state; and the block table contains the full speculative slot count. If any of these fail, do not proceed to graph capture.

## Stage 3 — Eager numerical parity and NaN elimination

The goal is to prove the model is correct before compilation changes the execution path. Eager execution is slower, but it is the only stage where a per-layer hook and a Python-level state dump are trustworthy.

1. **Run single-request greedy parity against a frozen oracle.** Use the same weights, dtype, tokenizer, prompt, seed, and stop conditions. Compare the next-token logit vector and the top-1 token at every position, not only the final answer. A model that returns a plausible sentence can still have a wrong recurrent state; the recorded 385-token case returned token 0 and repeated `!` while the state copy was elementwise identical.
2. **Run the boundary matrix.** At minimum: prompt lengths 128, 4096, and 16384 at batch 1 and batch 16, plus 383/384/385/386 around the page boundary. Include a mixed prefill-decode batch where one long prefill shares a step with speculative decodes; the K3 archive shows this mixed batch as a trigger that a pure prefix-cache test did not reproduce.
3. **Apply the NaN/Inf SOP when a case fails.** Use the layer/sub-op bisection, QuaRot norm/range check, recurrent-state dtype/decay audit, and fused-versus-eager oracle from `unit-vllm-ascend-kimi-k3-nan-inf-quarot-and-norm-precision-sop`. Do not proceed to stage 4 with a known failing case, and do not mask it by enabling a graph mode or a different attention backend.
4. **Separate a state-copy result from a hidden-state result.** An elementwise-identical state copy does not prove the entry point is correct. If the state is identical but the next hidden state is NaN, inspect the metadata classification, the `query_start_loc` length, the slot sentinel, and the kernel's empty-slot branch before changing the copy.
5. **Gate the stage.** Zero NaN/Inf in logits and checked states; top-1 token agreement for greedy decoding; cosine similarity over the next-token logit vector greater than 0.999; identical output hashes across repeated cold/warm/reset runs; and a recorded per-layer first-finite/last-finite trace for the longest case. A pass at one length or one batch is a localization, not a stage pass.

## Stage 4 — Piecewise TorchAir / ACLGraph capture and decode latency convergence

The goal is to capture a decode graph that reads the current step's metadata at fixed addresses, mutates state only through declared operations, and runs without a steady-state host D2H synchronization. Capture is not a correctness fix; it is a performance step that must preserve the stage 3 result.

1. **Capture decode only, piecewise first.** Keep prefill eager until decode replay is stable. Capture the decode bucket sizes that the scheduler actually uses, including the padded bucket and the speculative `K+1` bucket. A graph that is captured for one bucket and replayed for another is a recapture, not a hit.
2. **Bind static input and workspace buffers.** Copy `query_start_loc`, `seq_lens`, `slot_mapping`, block tables, and state indices into persistent device buffers before replay. `state.copy_()` must target a preallocated buffer at a fixed address. Do not allocate, call `empty_like`, change streams, or read a host scalar inside the captured region. A warmup with empty metadata must not be the shape that the graph specializes on; the observed K3 compile path could freeze the empty-metadata branch and then replay it for a real request.
3. **Declare or exclude mutating custom operators.** A custom op that mutates `conv_state` or `recurrent_state` must expose that mutation to the graph/compiler. If it cannot, keep it outside the captured region and split the graph around it. The observed `npu_causal_conv1d_custom` runtime failure had legal-looking inputs and indices, so an undeclared conv-state mutation is a first-class suspect, not an implementation detail.
4. **Keep AICPU metadata outside the graph when required.** Follow the existing DSA-v1-style `DeviceMetadataExecutor` pattern: compute the metadata on the host/AICPU, copy it to the device buffer, then replay the graph. AICPU metadata does not need to be a graph node. Do not create a new attention backend just to hold the metadata; extend the existing backend.
5. **Eliminate steady-state host D2H synchronization.** Count `.item()`, `.cpu()`, `.numpy()`, `torch.npu.synchronize()`, and host-side `num_accepted_tokens` reads in the decode loop. A graph replay that waits for a host scalar is not a captured decode. Move the value to a device buffer or restructure the step so the next replay consumes device data. A side-stream state update needs a GPU event edge to the default stream and to the next H2D buffer reuse; a CPU synchronize can hide the race in a probe while destroying the production timing.
6. **Measure convergence, not a fabricated speedup.** Record capture count, replay count, recapture count, host-sync count, decode TPOT/ITL, and output hash for eager versus graph on the same frozen workload. The target is stable replay with no recapture and no steady-state host sync, and a decode latency that converges to the graph-safe baseline. Do not publish a speedup from a run that changed the bucket, disabled a feature, or used a different workload.
7. **Gate the stage.** Decode replay count equals the expected number of steps; recapture count is zero after warmup; steady-state host D2H count is zero; output hash matches the eager stage 3 result; and the latency report is paired on the same workload. If a custom op cannot be made graph-safe, leave it out of the graph and document the piecewise boundary rather than claiming a full graph.

## Verification and exit criteria

The workflow exits only when all four stages pass on one frozen code state:

- **Loader/sharding:** every shard is owned, the derived-buffer/state-dict assertion holds, the KDA operator bundle is registered, and unequal-TP state mapping is explicit if used.
- **Hybrid cache/state:** cold, warm-prefix, reset, and chunked-prefill runs agree; the speculative slot count is fully allocated and zeroed; a 128K-class request has finite checked state.
- **Eager parity:** zero NaN/Inf, top-1 token agreement, logit cosine greater than 0.999, and identical output hashes across repeats for lengths 128/4096/16384 and batches 1/16, including the 383/384/385/386 boundary.
- **Graph:** decode replay is stable, no recapture after warmup, no steady-state host D2H, output hash equals eager, and the latency report is paired.

Record the exact source revisions, image digest, operator bundle hash, config, rank map, cache geometry, prompt/seed, per-stage result, and every failed case. If a stage fails, stop there and preserve the first failing artifact; do not advance on a health check, a single passing prompt, or a graph-mode rerun that makes the symptom disappear. The archive records partial fixes and explicit non-pass states for exact-boundary state admission and DSpark hot-cache reuse; this workflow does not convert those into a claim of complete support.
