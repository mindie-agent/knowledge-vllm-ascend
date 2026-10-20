---
schema: mindie-graded-unit/1
id: unit-vllm-ascend-kimi-k3-kda-mamba-hybrid-attention-axioms
revision: 1
content_digest: sha256:e7cc915b93dd52c231bfb68d9a19bb25905d76ca4694fd3a0546c6cbbb9e6699
tier: L1
category: axiom
domain_path: vllm-ascend/models/kimi-k3-hybrid
evidence_strength: software_contract
title: 'Kimi K3 KDA/Mamba hybrid state, layout, and graph-capture invariants'
conditions:
  framework: vllm-ascend
  model_family:
  - kimi-k3
  - hybrid-kda-gqa
  - mamba-hybrid
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  source_type: session_distill
  source_locator: docs/evidence/k3-hybrid-transcript-provenance.json#s-k3-quarot-long-context
  source_digest: cdd5c523ecadd81544bc3d12deba68bdc7889c96cbd10dbb3b52a88deaee0655
  evidence_refs:
  - docs/evidence/k3-hybrid-transcript-provenance.json
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/8db58966cd27b59d5d669539bf2a249b5b2b3023d511deb3ca4caffd4fde2c46/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/6e8e56e03f14cabd74fa97482deafe594561d16aa829fb94866cabfdd965200b/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/061f6320b291045cfccfba33e62a2f06f1f4175c6ce9ca3e3d04bc0ef30ad530/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/d1e573822d2b0e234bdb5d6be1d304adbadf96bcae377a5acc3c01f5dedbc238/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/ea7876a55b7e5eb30b5d33a44b89ac1216c10bc4f3db2ad45b3ff3255bebd020/index.md
  supersedes: []
triggers:
  error_signatures:
  - a decode step turns non-finite only after a long prefill or after crossing a page or chunk boundary
  - output becomes repeated '!' or repeated token 0 while the health endpoint stays successful
  - a recurrent or conv-state slot contains data from a previous request after preemption or prefix reuse
  - ACLGraph replay reads a stale query_start_loc or a state buffer from the previous step
  - Triton and AscendC causal-conv or gated-delta-rule kernels disagree only on slot 0 or on an empty slot
  keywords:
  - Kimi K3
  - KDA
  - gated delta rule
  - Mamba
  - causal_conv1d
  - conv_state
  - recurrent_state
  - chunk_gated_delta_rule
  - fused_recurrent_gated_delta_rule
  - query_start_loc
  - cu_seqlens
  - hybrid cache
  - ACLGraph
  - Ascend910_93
  - Ascend950
---

# Kimi K3 KDA/Mamba hybrid state, layout, and graph-capture invariants

**Condition.** A `vllm-ascend` Kimi K3-family hybrid model on Ascend 910B, 910_93, or 950, where each block alternates a KDA (Kernelized Delta Attention) linear-recurrent mixer — the K3 gated delta rule, exposed through GDN-named kernels in some revisions — with grouped-query attention, and where the KDA mixer also owns a causal-conv1d state. The model is served with paged KV cache and may use prefix caching, chunked prefill, speculative decoding, and ACLGraph replay. Shapes below are illustrative; the rule is the separation, not a particular head count.

**Constraint.**

1. **Partition state by producer, never by one inherited block-table stride.** The hybrid block has three independent state planes:
   - paged GQA KV cache, indexed by KV block id and page offset;
   - Mamba/KDA `conv_state`, one ring per request and layer, indexed by a request slot;
   - KDA `recurrent_state`, one accumulator per request, layer, and local head, also indexed by a request slot.

   A stride, page size, or block-table entry derived from one plane must not be used to address another. If an allocator forces a common page size, that equality must be proven once, recorded, and enforced by the allocator; it must not be assumed by a kernel. In the observed K3 v0.26 path the attention block size was fixed at 384 tokens to match the Mamba state page, and the 384-to-385 transition was the leading reproducible corruption boundary. Removing DSpark still reproduced that boundary, so the page mismatch was not merely a speculative-decoding artifact. Physical backing must also be disjoint: a draft KV page and a target KV page that alias the same storage produced a NaN in the target cache before any arithmetic diagnosis was possible.

   State slots follow the request lifecycle independently of KV blocks. A slot is allocated, zeroed or explicitly invalidated, bound to `request_id`, and released on completion or preemption. A KV block reference count says nothing about whether the corresponding recurrent slot is valid. Speculative state pools are the sharpest case: a `state_len=10` pool in which only three slots were initialized fed seven uninitialized slots into decode. A block table that is short by the speculative slot count is a hard failure, not a capacity warning.

   Prefix reuse is valid only when KV pages and recurrent/conv state are snapshotted and admitted at the same boundary. A KV-only prefix hit, a partial-tail hit, or a copy-on-write target that leaves the state slot at the old position is a corrupt request even when every block-table entry is in range. Exact-boundary admission must restore both planes or neither.

2. **Honor each KDA and conv kernel's layout and dtype contract.** Prefill and decode are different entry points, not two shapes of the same call:
   - Variable-length prefill packs requests with `cu_seqlens` or `query_start_loc`, whose length is `num_requests + 1` and whose last element is the total token count. `chunk_gated_delta_rule` consumes packed chunks and writes the final recurrent state; `causal_conv1d_fn` consumes the full sequence and advances `conv_state`.
   - Single-token decode passes one token per request. `fused_recurrent_gated_delta_rule` and `causal_conv1d_update` consume the per-request slot and update it in place. A padded or packed prefill `query_start_loc` must not be passed to the decode entry, and a decode slot index must not be interpreted as a token offset in the packed prefill buffer.
   - A stateful single-token prefill must be classified as such in the metadata. The 385-token K3 failure is the canonical example: the 138-element state copy was identical, yet the decode graph hidden state `[1,7168]` became all NaN because the GDN metadata omitted the stateful single-token-prefill classification. Correct arithmetic did not save a misclassified entry point.
   - Maintain `conv_state` and `recurrent_state` in `torch.float32` (or an explicitly declared fp32 master) and cast only at the kernel boundary. Activations may be bf16 or fp16; the recurrent accumulator must not be. Keep `beta` finite and in its declared range, keep the decay `g` in log space, and clamp `exp(g)` so a long prefill cannot overflow. A recurrent decay drift is a slow-motion correctness failure that looks like a bad prompt.
   - Normalize slot sentinels before switching kernels. In the observed K3 path Triton treated `state_idx <= 0` as invalid while AscendC treated slot 0 as valid; an illegal or empty slot could also make an AscendC core return early and expose `empty_like` memory that was never written. `dt_bias` shape is part of the contract (`[num_heads * head_dim]` versus `[num_heads, head_dim]`); a shape mismatch can masquerade as a numerical fault.

3. **Capture only fixed-address, declared state mutation.** ACLGraph replay removes the Python launch path, so it cannot repair an address, an allocation, or a host synchronization that changed after capture:
   - `state.copy_()` is permitted only into a preallocated device buffer whose address is fixed for the graph's lifetime. Do not allocate inside the captured region, do not use `empty_like` there, and do not let a warmup path specialize on empty metadata that is replaced by real metadata at replay.
   - Dynamic `query_start_loc`, `seq_lens`, `slot_mapping`, and state indices must be copied into static device buffers before replay. The graph must see the current step's metadata at the same addresses, not a captured Python scalar.
   - A custom operator that mutates `conv_state` or `recurrent_state` must declare that mutation to the graph/compiler, or stay outside the captured region. The observed `npu_causal_conv1d_custom` failure mode was exactly this: a runtime ACLGraph forward went non-finite while inputs, padding, `query_start_loc`, and slot indices looked legal, and the conv-state mutation contract was the leading unexplained edge.
   - A state write on a side stream needs a GPU event edge to the default stream and to the next step's H2D buffer reuse. A CPU synchronize is not a fix: it can hide the race in a diagnostic run while destroying graph timing and leaving the production path broken. Cross-stream visibility races at the KDA layer boundary produced intermittent non-finite recurrent state that a layer-boundary sync made deterministic.
   - Padded tail slots must be inert. A captured bucket larger than the scheduled batch must carry `slot_mapping = -1` or an equivalent mask so padded tokens never advance a real request's conv or recurrent state.

**Rationale.** The three state planes have different producers, shapes, update rules, and lifetimes. Paged KV assumes block-granular allocation and reference counting; a recurrent state assumes one live accumulator per request; a conv state assumes a ring buffer advanced once per token. Forcing them through one stride or one slot map creates a class of failures that looks numerical but is structural: a valid-looking index reads a stale or uninitialized buffer, and the first visible symptom is NaN, repeated token 0, or repeated `!`. Graph replay then removes the last chance to repair the mistake in Python. The invariant is therefore stated as a precondition: separate the planes, prove any shared geometry once, and make every mutation replay-safe before attributing a failure to kernel math.

**Evidence.** The provenance record is `docs/evidence/k3-hybrid-transcript-provenance.json`. It pins the following archive observations:

- K3 v0.26 aligned attention blocks to a 384-token Mamba page; the 384-to-385 boundary produced corruption/NaN, and the hybrid cache manager, state copy/restore, and zeroing lifecycle were the remaining suspects after DSpark was removed.
- A speculative `state_len=10` pool had only three initialized slots; zeroing the seven new slots fixed the no-cache path but a 98.73% prefix hit still emitted repeated `!`. A later 32-token hot-cache run still diverged even after a block-table allocation was corrected.
- A 385-token case had an identical 138-element state copy but an all-NaN `[1,7168]` decode hidden state; the missing GDN metadata classification for stateful single-token prefill was the identified cause.
- Runtime ACLGraph forwards failed with legal-looking inputs and indices, with the `npu_causal_conv1d_custom` conv-state mutation contract unresolved; a side-stream state update without an event edge to the next default-stream/H2D reuse later deadlocked concurrent short requests.
- A draft KV page and a target KV page aliased the same physical backing before target KV became NaN, and KDA/GDN could not consume a first-axis non-contiguous view in that path.

These observations establish the failure modes and the contract boundaries. They do not establish a universal page size, a complete fix for exact-boundary state admission, or a throughput result; those limits are recorded in the evidence file.
