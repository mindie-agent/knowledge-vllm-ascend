---
category: sop
conditions:
  repository: vllm-project/vllm-ascend
  revision: 175ef4d49c2072f9233619636f0309f4f88b4e2c
content_digest: sha256:025b36835f18326dda7ef09ea380c8a4da9fd8d34eb3f39841010f302e3501ba
domain_path: integration/custom-ops
evidence_strength: single_observation
id: unit-vllm-ascend-custom-op-gates-import-retry-and-nz
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-vllm-ascend-custom-op-gates-import-retry-and-nz-review.json
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/utils.py#L397-L456
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/utils.py#L299-L316
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/utils.py#L656-L786
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/utils.py#L249-L286
  source_digest: b04fd7d5698c9a1ad9cb1470fe4ce593e1e21bbdd00747c2d797850becc43aa9
  source_locator: docs/evidence/unit-vllm-ascend-custom-op-gates-import-retry-and-nz-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Ascend custom-op imports, registration and NZ selection have distinct gates
triggers:
  error_signatures: []
  keywords:
  - vllm-ascend
  - ascend
  - custom-op
  - imports
  - registration
  - and
  - nz
  - selection
  - have
  - distinct
  - gates
---
At vLLM-Ascend revision 175ef4d49c2072f9233619636f0309f4f88b4e2c, enable_custom_op lazily memoizes its result. Batch-invariant mode or an A5 device sets it false before extension import. Otherwise the extension and meta-registration imports are attempted even while torch.compiler.is_compiling is true; only the initial environment bootstrap is skipped while compiling. An ImportError containing libcust_opapi.so permits one non-compiling retry after adding the vendor library path. That retry imports meta_registration before vllm_ascend_C, reversing the initial order. Other ImportErrors set the result false; exceptions of other types propagate. register_ascend_customop separately registers the operation map, adding GateLinear only for a resolved DeepSeek MLA model and substituting selected implementations on 310P. Its completion flag is set after the registration loop, so an exception can leave earlier registrations applied while the flag remains false. NZ selection is another independent rule: float32 and meta tensors are excluded; 310P then selects NZ; otherwise weight_nz_mode disables it at zero, permits bf16/fp16 only at two, and permits other dtypes when truthy. The adjacent VLLM_ASCEND_ENABLE_NZ comment does not name the value this function actually reads.

Reviewed limits and corrections:

- Remove the blanket fail-closed description: non-ImportError exceptions propagate, and registration can partially apply.
- Compiler mode skips bootstrap, not the initial imports.
- Classify this adaptation mechanism once in vllm-ascend; do not duplicate the unit in infer-ops-compat.
- NZ policy reads weight_nz_mode rather than the adjacent comment's environment-variable name.

Adoption boundary: Use these source gates to diagnose an Ascend integration at this revision. This unit does not establish extension availability, operator correctness or performance on any device.

Evidence scope: Root accepted only the rewritten mechanism. Independent Cursor reading and its broader findings remain in the immutable author packet. Root source reading is enumerated separately.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [vllm_ascend/utils.py:397-456](https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/utils.py#L397-L456)
- [vllm_ascend/utils.py:299-316](https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/utils.py#L299-L316)
- [vllm_ascend/utils.py:656-786](https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/utils.py#L656-L786)
- [vllm_ascend/utils.py:249-286](https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/utils.py#L249-L286)
