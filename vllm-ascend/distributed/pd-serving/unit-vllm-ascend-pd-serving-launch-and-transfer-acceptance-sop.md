---
category: sop
conditions:
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  - Ascend950
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
domain_path: vllm-ascend/distributed/pd-serving
evidence_strength: software_contract
id: unit-vllm-ascend-pd-serving-launch-and-transfer-acceptance-sop
inherits_from:
- unit-vllm-ascend-pd-kv-connector-layout-and-handoff-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/b61c9b207f72e7765bf096a5dff2fc928a19b03f/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L819-L973
  - https://github.com/vllm-project/vllm/blob/7913c96894cacc348397a377411acde96b63e68a/vllm/v1/core/sched/scheduler.py#L1244-L2185
  - https://github.com/vllm-project/vllm-ascend/blob/b61c9b207f72e7765bf096a5dff2fc928a19b03f/vllm_ascend/ascend_config.py#L657
  - https://github.com/vllm-project/vllm-ascend/blob/b61c9b207f72e7765bf096a5dff2fc928a19b03f/vllm_ascend/batch_invariant.py#L102
  source_locator: vllm-project/vllm-ascend:b61c9b207f72e7765bf096a5dff2fc928a19b03f:examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
  source_type: upstream_code_audit_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: PD disaggregated serving launch contract, proxy forwarding, and real-transfer acceptance SOP
triggers:
  error_signatures:
  - PD proxy returns HTTP 200 but Decode recomputed prefill locally after remote KV pull failed
  - Appending Prefill 1-token output to Decode prompt duplicates or shifts the first generated token
  - P and D workers hang during KV handshake due to mismatched HCCL_IF_IP, ports, or post-init config
  keywords:
  - PD serving launch
  - kv_transfer_config
  - kv_producer
  - kv_consumer
  - load_balance_proxy_server_example
  - WAITING_FOR_REMOTE_KVS
  - kv_transfer_params
  - enable_fused_mc2
  - batch_invariant
  - acceptance gate
---

# PD disaggregated serving launch contract, proxy forwarding, and real-transfer acceptance SOP

## When to Use

Apply this SOP when launching or validating a `vllm-ascend` PD disaggregated cluster (`1P1D` / `xPyD`) to prevent handshake hangs, proxy token-duplication bugs, and silent local-recompute fallbacks.

## Step 1 — Align P/D Launch Contract and Post-Init Config

1. Match model weights, tokenizer, `dtype`, `block_size`, and `kv_connector` across Prefill (`kv_role="kv_producer"`) and Decode (`kv_role="kv_consumer"`).
2. Assign disjoint `ASCEND_RT_VISIBLE_DEVICES`, HTTP ports, and KV handshake / Mooncake ports while keeping `HCCL_IF_IP` / `HCCL_SOCKET_IFNAME` on the same reachable fabric.
3. Inspect **post-init effective config** on both P and D instead of raw CLI flags:
   - `vllm_ascend/ascend_config.py:657` normalizes `enable_fused_mc2` and MegaMoE settings after init.
   - `vllm_ascend/batch_invariant.py:102` overrides weight layout, HCCL/LCCL determinism, and operator dispatch together.

## Step 2 — Bring Up Workers Before Proxy and Check `max_tokens=1` Forwarding

1. Wait until both P and D pass `/health` (or `/v1/models`) and bind KV transfer listeners before starting the proxy (`load_balance_proxy_server_example.py:819,973`).
2. **Proxy forwarding rule:** The reference v1 proxy sends `max_tokens=1` to P only to collect `kv_transfer_params`, then forwards the **original prompt** plus `kv_transfer_params` to D. Never append P's generated token to D's prompt unless your connector disables `N-1` rewind, or D will duplicate/shift the first token.

## Step 3 — Prove Real Remote KV Transfer (Reject HTTP 200 Recompute Fallback)

A `200 OK` response is not proof of PD transfer because a failed D pull can fall back to local prefill recomputation (`scheduler.py:2170`). On a single greedy smoke request (`temperature=0`), verify:
1. P completed `save_kv_layer` / `wait_for_save` for the request blocks.
2. D entered `WAITING_FOR_REMOTE_KVS` (`scheduler.py:1244,2185`) and pulled non-zero KV bytes.
3. D recorded zero failed-load recomputations and ran only the `1`-token (`N-1`) handoff step (or bounded SWA replay).

## Step 4 — Output Morphology Gate

Run 5–10 greedy prompts (`temperature=0`, `max_tokens=128`) against a colocated baseline and reject if outputs show `empty`, `repeated_char_run`, `repeated_ngram`, `punctuation_run`, `replacement_char`, or `immediate_eos`.
