---
category: mutator
conditions:
  framework:
  - vllm-ascend
  - any
domain_path: vllm-ascend/speculative-decoding
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-dspark-mtp-aclgraph-d2h-elimination-mutator
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-step3p5-and-hybrid-mtp-cache-rollback-workflow
provenance:
  evidence_refs:
  - docs/evidence/l4-dspark-mtp-aclgraph-d2h-elimination-ablation.json
  experiment_status: ablation_verified
  source_commit: d8c9e3a673af918c94a7794ef6f1f69c024d9c62
  source_digest: 99555432d52fc8846ea1870c65a293030914209d0795103f90f35b3e2ef89a33
  source_locator: docs/evidence/l4-dspark-mtp-aclgraph-d2h-elimination-ablation.json
  source_repository: https://github.com/vllm-project/vllm-ascend
  source_type: ablation_design_receipt
  supersedes: []
revision: 5
schema: mindie-graded-unit/1
tier: L4
title: v2 DSpark ACLGraph bucket alignment, metadata hoisting, and device-resident
  D2H elimination (`vllm_ascend/worker/v2/spec_decode/dspark/speculator.py`)
triggers:
  error_signatures:
  - FIA TND contract violation queryT (12) != last element of actualSequenceLengthQ
    (9)
  - FIA rejects a replayed v2 DSpark graph because actual_seq_lengths_q[-1] differs
    from the captured query-token count
  - num_accepted_tokens.item() or .cpu() or a host seq_lens read appears inside the
    captured speculative step
  keywords:
  - dspark
  - speculator.py
  - dspark_aclgraph_step
  - dspark_aclgraph_speculative_step
  - run_ablation_suite
  - speculative_decode
  - aclgraph
  - num_query_per_req
  - actual_seq_lengths_q
  - actualSequenceLengthQ
  - FIA
  - num_accepted_tokens
  - d2h
  - bucket
---
# v2 DSpark ACLGraph bucket alignment, metadata hoisting, and device-resident D2H elimination (`vllm_ascend/worker/v2/spec_decode/dspark/speculator.py`)

## Architectural Specification (`Ascend 950DT`)

On `Ascend 950DT`, v2 `DSpark` full-graph speculative decoding (`dspark_aclgraph_speculative_step` in `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py`) requires three coupled invariants across arbitrary batch sizes `num_reqs`:

1. **`align_buckets` (`bool = True`) — Independent Draft (`num_query_per_req`) vs Target (`1 + num_speculative_tokens`) Bucket Selection**:
   - Compute `draft_tokens_raw = num_reqs * num_query_per_req` and `target_tokens_raw = num_reqs * (1 + num_speculative_tokens)`.
   - **When `align_buckets=True`**: Select `draft_tokens_padded = next(b for b in sorted(bucket_sizes) if b >= draft_tokens_raw)` (e.g. `12` when `num_reqs=3, Q=3` and `bucket_sizes=(4, 8, 12, 16, 24, 32)`; `16` when `num_reqs=5, Q=3`) and independently select `target_tokens_padded = next(b for b in sorted(bucket_sizes) if b >= target_tokens_raw and b > draft_tokens_padded)` (so `target_tokens_padded` is `16` when `num_reqs=3, K=3` and `24` when `num_reqs=5, K=3`). Pad `slot_mapping` to length `draft_tokens_padded` with `-1` in tail slots `[draft_tokens_raw : draft_tokens_padded]`.
   - **When `align_buckets=False` (ablation arm `minus_bucket_alignment`)**: Set `target_tokens_padded = draft_tokens_padded` (misaligning the target graph token budget).

2. **`hoist_metadata` (`bool = True`) — Device-Resident Cumulative TND `actual_seq_lengths_q` Tail Padding**:
   - Build base cumulative query lengths on device: `base_q = torch.arange(1, num_reqs + 1, dtype=torch.int32, device=req_seq_lens.device) * num_query_per_req`.
   - When `align_buckets=True` and `draft_tokens_padded > draft_tokens_raw`, append a 1-element int32 device tensor `[draft_tokens_padded]` via `torch.cat` so the final element of `actual_seq_lengths_q` equals `draft_tokens_padded` (e.g. `[3, 6, 9, 12]` for `num_reqs=3`, `[3, 6, 9, 12, 15, 16]` for `num_reqs=5`) without calling `.item()` or `.tolist()` to check it.
   - **When `hoist_metadata=True`**: Set `metadata_allocs_in_capture = 0` with zero `.clone()` calls on `base_q`.
   - **When `hoist_metadata=False` (ablation arm `minus_metadata_hoist`)**: Clone `actual_seq_lengths_q = actual_seq_lengths_q.clone()` inside capture and record `metadata_allocs_in_capture = 2`.

3. **`device_resident_accepted` (`bool = True`) — Zero-D2H Device Reduction of Accepted Tokens & Sequence Lengths**:
   - **When `device_resident_accepted=True`**: Reduce `accepted_counts = accepted_mask.sum(dim=-1).to(torch.int32)` (if 2D) or `accepted_mask.to(torch.int32)` (if 1D) purely on device, compute `updated_seq_lens = req_seq_lens.to(torch.int32) + accepted_counts + 1`, and compute `target_query_start_loc` via `torch.cumsum(accepted_counts + 1, dim=0)` with **zero `.item()`, `.cpu()`, or `.tolist()` calls** (`captured_step_d2h_syncs = 0`, `scalar_sync_ops == 0`). Also avoid calling `req_seq_lens.max().item()`.
   - **When `device_resident_accepted=False` (ablation arm `minus_device_resident_accepted_tokens`)**: Perform host `.item()` read (`_ = int(accepted_counts.sum().item())`) and record `captured_step_d2h_syncs = 2`.
   - Return `(updated_seq_lens, metrics_dict)` where `metrics_dict` includes `"updated_seq_lens"`, `"actual_seq_lengths_q"`, `"padded_slot_mapping"`, `"draft_tokens_padded"`, `"target_tokens_padded"`, `"captured_step_d2h_syncs"`, and `"metadata_allocs_in_capture"`.
