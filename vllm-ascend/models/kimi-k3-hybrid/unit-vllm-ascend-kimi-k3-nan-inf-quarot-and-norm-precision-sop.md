---
schema: mindie-graded-unit/1
id: unit-vllm-ascend-kimi-k3-nan-inf-quarot-and-norm-precision-sop
revision: 1
content_digest: sha256:2892560065cb1bb6b278da344b8665af2d90567dc244d83d287cf909f70d104d
tier: L2
category: sop
domain_path: vllm-ascend/models/kimi-k3-hybrid
evidence_strength: reproduced_empirical
title: 'Kimi K3 NaN/Inf, QuaRot rotation, and RMSNorm precision bisection SOP'
conditions:
  framework: vllm-ascend
  symptom_class:
  - nan_inf
  - precision_divergence
  - garbled_output
inherits_from:
- unit-vllm-ascend-kimi-k3-kda-mamba-hybrid-attention-axioms
provenance:
  source_type: session_distill
  source_locator: docs/evidence/k3-hybrid-transcript-provenance.json#s-k3-quarot-long-context
  source_digest: cdd5c523ecadd81544bc3d12deba68bdc7889c96cbd10dbb3b52a88deaee0655
  evidence_refs:
  - docs/evidence/k3-hybrid-transcript-provenance.json
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/8db58966cd27b59d5d669539bf2a249b5b2b3023d511deb3ca4caffd4fde2c46/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/6e8e56e03f14cabd74fa97482deafe594561d16aa829fb94866cabfdd965200b/index.md
  - docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/tasks/fb3e496700cf29216ee2ee4aa8c85ccdc60604c39c90887d0bd52af6c889b5b4/index.md
  supersedes: []
triggers:
  error_signatures:
  - logits or a decoder hidden state become NaN or Inf at a specific layer after a long prefill
  - greedy output degenerates to repeated '!' or repeated token 0 while HTTP health remains 200
  - QuaRot/Hadamard fused output has a norm different from the input activation
  - fused RMSNorm matches an eager oracle without alias but diverges when input and residual share storage
  - KDA recurrent state is non-finite only on some runs of the same prompt
  keywords:
  - Kimi K3
  - NaN
  - Inf
  - QuaRot
  - Hadamard
  - RMSNorm
  - add_rms_norm
  - npu_rms_norm
  - KDA
  - recurrent_state
  - beta
  - log-space decay
  - forward hook
  - cosine similarity
---

# Kimi K3 NaN/Inf, QuaRot rotation, and RMSNorm precision bisection SOP

## Scope

Use this procedure for a `vllm-ascend` Kimi K3-family hybrid model that emits NaN/Inf logits, non-finite hidden states, repeated `!`/token 0, or a precision divergence that is not explained by a missing weight or a failed health check. The procedure assumes the exact failing request, revision, weights, CANN/torch stack, parallelism, dtype, and graph mode have been frozen. It localizes the first non-finite producer before changing kernels; it is not a license to average away a failing case.

Run step 1 with graph replay disabled (`enforce_eager` or the equivalent), because hooks and per-layer reads are not reliable inside a captured region. Once the first bad layer and sub-op are known, steps 2 to 4 can be run on the captured activation and on a standalone operator harness.

## Procedure

1. **Bisect the first non-finite layer and sub-op with forward hooks.**

   Attach a probe to every decoder layer first. When the first bad layer is known, re-attach to that layer's `q_proj`/`k_proj`, QuaRot/Hadamard transform, `fused_rms_norm`, KDA recurrent update, and MoE router. The probe must check both NaN and Inf, and must print `amax`, `amin`, and L2 norm so a silent magnitude collapse is visible before it becomes NaN.

   ```python
   import torch

   def finite_probe(name):
       def hook(_module, _inputs, output):
           x = output[0] if isinstance(output, tuple) else output
           if not torch.is_tensor(x) or not x.is_floating_point():
               return
           xf = x.detach().float()
           bad = bool((torch.isnan(xf).any() | torch.isinf(xf).any()).item())
           print(
               f"{name:64s} finite={not bad} "
               f"amax={xf.amax().item():.6g} "
               f"amin={xf.amin().item():.6g} "
               f"l2={xf.norm().item():.6g}"
           )
           if bad:
               raise RuntimeError(f"first non-finite tensor: {name}")
       return hook

   handles = []
   for name, module in model.named_modules():
       if name.startswith("model.layers.") or name in TARGET_SUBMODULES:
           handles.append(module.register_forward_hook(finite_probe(name)))
   ```

   Run one greedy request at temperature 0 with a fixed seed. The first layer whose output is non-finite is the boundary; its input is the last known-finite tensor. Record the name, shape, dtype, `amax`, `amin`, and L2 norm for the last finite and first non-finite tensors. Do not start from the final logits: the first NaN may be many layers earlier, and a later layer can turn a finite but wrong state into NaN.

   Verification: the probe report names exactly one first bad module in a deterministic run, and the same module is bad when the request is repeated with the same frozen inputs. If the bad module changes between runs, treat it as a race and go to step 3 plus the graph/stream checks, not as a pure arithmetic bug.

2. **Check QuaRot/Hadamard and online activation rotation numerics.**

   Reconstruct the Hadamard matrix used by the checkpoint and verify that the implementation applies an orthonormal transform exactly once. For an online activation rotation applied at runtime rather than folded into the weights, the same check runs on the captured activation immediately before the fused transform. For a raw `+1/-1` Hadamard `H_raw`, the normalized matrix is `H = H_raw / sqrt(H.shape[-1])`; then `H @ H.T` is the identity. A transform that omits the `1/sqrt(H)` scale, applies it twice, or transposes the multiplication changes the activation norm even when every individual multiply looks correct.

   ```python
   import math
   import torch

   H_raw = build_hadamard(size)                 # +1/-1 matrix
   H = H_raw.float() / math.sqrt(H_raw.shape[-1])
   eye = torch.eye(H.shape[-1], dtype=H.dtype, device=H.device)
   assert torch.allclose(H @ H.T, eye, atol=1e-5), "Hadamard is not orthonormal"

   x = captured_pre_rotation_activation.detach().float()
   y_ref = x @ H.T                              # check the layout the kernel declares
   y_fused = fused_rotation(x)
   assert torch.allclose(y_fused.float(), y_ref, rtol=1e-4, atol=1e-4)

   norm_ratio = y_ref.norm(dim=-1) / x.norm(dim=-1)
   print("norm ratio min/max:", norm_ratio.min().item(), norm_ratio.max().item())
   print("pre-rotation abs max:", x.abs().max().item())
   print("fp16 max:", torch.finfo(torch.float16).max)   # 65504
   ```

   Accumulate the rotation and the following projection in fp32 before casting to bf16 or int8. In fp16, an activation outlier above `65504` becomes infinity before any downstream clamp can help; in bf16 the exponent range is wider, but the 8-bit significand can still lose small channels and produce underflow-like drift. If a pre-rotation `amax` exceeds the target dtype's finite range, keep that stage in fp32 or bf16, or compute the quantization scale after the rotation rather than before it. For int8, the per-token scale must be derived from the post-rotation tensor that the quantizer actually sees.

   Also verify the anti-rotation direction. When the loader skips QuaRot for a shared `embedding`/`lm_head`, the corresponding un-rotation must be skipped for the same tensor; a flag that is read by one consumer and not another silently rotates a shared weight twice or zero times. Compare a captured fused output against the explicit `x @ H.T` reference and record both the max absolute error and the cosine similarity.

   Verification: `H @ H.T` is the identity, the norm ratio is 1 within tolerance, the fused/reference cosine is greater than `0.9999`, and no captured activation exceeds the chosen cast's finite range. A norm jump localizes the defect to the scale or transpose, not to the KDA recurrence.

3. **Audit KDA/gated delta rule recurrent-state precision.**

   Inspect the actual state tensors, not just their initialization code. `recurrent_state` and `conv_state` must be `torch.float32` (or have an explicit fp32 master) even when the surrounding activations are bf16/fp16; the cast to activation dtype belongs at the kernel boundary. Check the state after prefill, after each decode step, and after a state copy/restore.

   ```python
   for name, state in recurrent_states.items():
       assert state.dtype == torch.float32, (name, state.dtype)
       finite = bool(torch.isfinite(state).all().item())
       print(name, tuple(state.shape), state.abs().max().item(), finite)
       assert finite, f"non-finite recurrent state: {name}"

   assert torch.isfinite(beta).all(), "beta is non-finite"
   assert (beta >= 0).all() and (beta <= 1).all(), "beta outside [0, 1]"

   g = decay_log_space                      # must be <= 0 for a stable decay
   assert torch.isfinite(g).all(), "log decay is non-finite"
   decay = torch.exp(g.clamp(max=0.0))
   assert torch.isfinite(decay).all()
   assert decay.max().item() <= 1.0
   ```

   Check the state norm over the full prefill, not only the final token. A norm that decays to zero or grows without bound across a 16K-token sequence points to a log-space/clamp defect or an fp16 accumulator, even if the final logits happen to be finite. A state copy that is elementwise identical but followed by a NaN hidden state is evidence that the metadata or the entry-point classification is wrong, not that the copy is wrong; in the recorded 385-token K3 case all 138 checked copy elements were identical while the decode hidden state went all-NaN.

   Verification: every checked state is fp32 and finite, `beta` is in range, `exp(g)` never exceeds 1, and the state norm is stable across the longest prefill in the matrix. If the state is finite but the next hidden state is not, return to step 1 and inspect the metadata classification and the kernel entry point.

4. **Compare the fused Ascend operator with a CPU/NPU eager golden oracle.**

   Build the oracle from the same weights, same inputs, same dtype-rounded values, and same epsilon. Prefer a CPU fp32 reference for the formula and an NPU eager fp32 reference for the kernel path; compare both when the fused kernel's accumulation order is in question.

   For `torch_npu.npu_rms_norm`, verify the epsilon placement. The reference is `x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps) * gamma`, not `x * rsqrt(mean(x^2) ) * gamma + eps` and not an epsilon added after the normalization. Compare the fused output with the explicit formula at `eps` values around the model's configured value, including a very small `eps` where a misplaced addition is most visible.

   For `add_rms_norm`, test the aliasing case explicitly. If `x` and `residual` share storage, an in-place sum can overwrite the residual before the normalization reads it; a correct implementation must either document and order that mutation or require non-aliased inputs. Run both arms:

   ```python
   # Non-aliased control.
   x1 = x.clone()
   r1 = residual.clone()
   y_control = fused_add_rms_norm(x1, r1, gamma, eps)

   # Aliased adversarial arm: same storage for input and residual.
   buf = x.clone()
   y_alias = fused_add_rms_norm(buf, buf, gamma, eps)

   # Explicit fp32 reference for the non-aliased arm.
   s = x.float() + residual.float()
   y_ref = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps) * gamma.float()

   print("control max error:", (y_control.float() - y_ref).abs().max().item())
   print("alias max error:", (y_alias.float() - y_ref).abs().max().item())
   ```

   A control that matches and an alias arm that does not is an aliasing defect in the fused path. A control that also mismatches is an epsilon, reduction-order, or gamma-dtype defect. Check that `gamma` is not silently cast to a lower precision and that the returned residual is the value the next layer expects.

   Verification: max absolute and relative error are within the declared tolerance, cosine similarity for the operator output is greater than `0.99999`, no NaN/Inf is produced, and the aliased arm either matches or fails loudly with a documented contract. Do not accept a fused operator because the full model's final logits happen to look acceptable.

5. **Run the length and batch acceptance matrix and gate the fix.**

   Re-run the same code state, weights, and greedy decoding over prompt lengths `128`, `4096`, and `16384` and batch sizes `1` and `16`, with prefill and decode both exercised. Include the page/chunk boundaries that are relevant to the model (for example, the 383/384/385/386 neighborhood of a 384-token page) because a boundary case can fail while every round number passes.

   ```python
   CASES = [(128, 1), (4096, 1), (16384, 1),
            (128, 16), (4096, 16), (16384, 16)]

   for prompt_len, batch_size in CASES:
       logits, tokens, states = run_greedy_case(prompt_len, batch_size, seed=0)
       assert not torch.isnan(logits).any(), (prompt_len, batch_size, "NaN")
       assert not torch.isinf(logits).any(), (prompt_len, batch_size, "Inf")
       for name, state in states.items():
           assert torch.isfinite(state).all(), (prompt_len, batch_size, name)

       cosine = torch.nn.functional.cosine_similarity(
           logits_ref.flatten().float(), logits.flatten().float(), dim=0
       )
       assert cosine.item() > 0.999, (prompt_len, batch_size, cosine.item())
       assert logits.argmax(-1).equal(logits_ref.argmax(-1)), "top-1 mismatch"
   ```

   Verification: the reference is the frozen eager/fp32 oracle from step 4, not a previous failing run. A case passes only when all of the following hold: zero NaN/Inf in logits and checked states, top-1 token agreement for greedy decoding, cosine similarity over the next-token logit vector greater than `0.999`, and no repeated `!`/token 0 degeneration. Batch 16 must include a mixed prefill-decode batch, not sixteen identical prefill requests, because the state cross-talk failure mode appears when one long prefill shares a step with speculative decodes.

   If any case fails, return to step 1 with the smallest failing case and keep the failing prompt, seed, revision, and state dump. Do not exclude the case, average it with passing runs, or enable graph mode to see whether the failure disappears.

## Exit gate

The SOP is complete only when steps 1 to 5 pass on one frozen code state and the evidence record is updated with the first bad layer/sub-op, the QuaRot norm/range check, the recurrent-state dtype and decay check, the fused/oracle deltas, and the full length/batch matrix. A pass at one prompt length, one batch size, or one graph mode is a localization result, not a fix. If the failure is intermittent, record the run count and the synchronization changes separately; a probe that changes timing is not proof that the original path is correct.

## Boundaries

This procedure does not establish model accuracy, throughput, or a universal QuaRot scale. It distinguishes a first non-finite producer from a downstream symptom and gives a reproducible operator-level comparison. The `1/sqrt(H)` and `65504` checks are numerical contract checks; the archived K3 evidence establishes QuaRot rotation/anti-rotation failure modes, not a measured threshold sweep. The `add_rms_norm` aliasing arm is an adversarial contract test whose exact kernel revision must be recorded with the result.
