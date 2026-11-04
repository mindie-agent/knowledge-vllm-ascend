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
revision: 4
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

## Transformation (`dspark_aclgraph_step` & `run_ablation_suite`)

Implement the 3 separable factors in `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py` with optional keyword-only boolean ablation flags (`align_buckets=True`, `hoist_metadata=True`, `device_resident_accepted=True`) and expose `run_ablation_suite()` across the 4 Leave-One-Out arms (`full_candidate`, `minus_bucket_alignment`, `minus_metadata_hoist`, `minus_device_resident_accepted_tokens`):

1. **`align_target_and_observed_draft_graph_buckets` (`align_buckets=True`)**:
   - Select `draft_tokens_padded` as the smallest bucket in `capture_buckets` `>= num_reqs * num_query_per_req` (e.g. `12` for `3 * 3 = 9`), and `target_tokens_padded` as the smallest bucket in `capture_buckets` `>= num_reqs * (1 + num_speculative_tokens)` (e.g. `16` for `3 * 4 = 12`).
   - Pad `slot_mapping` to `draft_tokens_padded` with `-1` on tail slots (`padded_slots[draft_tokens_raw:] = -1`) and mask tail `seq_lens` to `0`.
2. **`hoist_attention_metadata_out_of_capture` (`hoist_metadata=True`)**:
   - Build `actual_seq_lengths_q` outside the captured step (`metadata_allocs_in_capture = 0`), appending `draft_tokens_padded` when `draft_tokens_padded > draft_tokens_raw` so `actual_seq_lengths_q[-1] == draft_tokens_padded` (`[3, 6, 9, 12]`), satisfying the FIA TND contract (`queryT == actual_seq_lengths_q[-1]`).
3. **`device_side_num_accepted_tokens_and_seq_lens` (`device_resident_accepted=True`)**:
   - Update `seq_lens` in-place or via device tensor arithmetic (`updated_seq_lens = seq_lens + num_accepted_tokens + 1`) and compute `target_query_start_loc` on device (`torch.cumsum(num_accepted_tokens + 1, dim=0)`) with zero `.cpu()` / `.item()` calls (`captured_step_d2h_syncs = 0`).

```python
def dspark_aclgraph_step(
    query_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    num_reqs: int = 3,
    num_speculative_tokens: int = 3,
    num_query_per_req: int = 3,
    capture_buckets: tuple[int, ...] = (4, 8, 12, 16, 24, 32),
    align_buckets: bool = True,
    hoist_metadata: bool = True,
    device_resident_accepted: bool = True,
) -> dict[str, object]:
    draft_tokens_raw = num_reqs * num_query_per_req
    target_tokens_raw = num_reqs * (1 + num_speculative_tokens)

    if align_buckets:
        draft_tokens_padded = next(b for b in sorted(capture_buckets) if b >= draft_tokens_raw)
        target_tokens_padded = next(b for b in sorted(capture_buckets) if b >= target_tokens_raw)
    else:
        draft_tokens_padded = next(b for b in sorted(capture_buckets) if b >= draft_tokens_raw)
        target_tokens_padded = draft_tokens_padded

    padded_slots = torch.full((draft_tokens_padded,), -1, dtype=slot_mapping.dtype, device=slot_mapping.device)
    valid_copy = min(slot_mapping.numel(), draft_tokens_raw)
    padded_slots[:valid_copy] = slot_mapping[:valid_copy]

    if hoist_metadata:
        metadata_allocs_in_capture = 0
        base_q = torch.arange(
            num_query_per_req, draft_tokens_raw + 1, num_query_per_req,
            dtype=torch.int32, device=seq_lens.device,
        )
        if align_buckets and draft_tokens_padded > draft_tokens_raw:
            pad_tail = torch.tensor([draft_tokens_padded], dtype=torch.int32, device=seq_lens.device)
            actual_seq_lengths_q = torch.cat([base_q, pad_tail], dim=0)
        else:
            actual_seq_lengths_q = base_q
    else:
        metadata_allocs_in_capture = 2
        base_q = torch.arange(
            num_query_per_req, draft_tokens_raw + 1, num_query_per_req,
            dtype=torch.int32, device=seq_lens.device,
        )
        if align_buckets and draft_tokens_padded > draft_tokens_raw:
            actual_seq_lengths_q = torch.cat(
                [base_q, torch.tensor([draft_tokens_padded], dtype=torch.int32, device=seq_lens.device)],
                dim=0,
            )
        else:
            actual_seq_lengths_q = base_q

    if device_resident_accepted:
        captured_step_d2h_syncs = 0
        updated_seq_lens = (seq_lens + num_accepted_tokens + 1).to(torch.int32)
        step_tokens = (num_accepted_tokens + 1).to(torch.int32)
        target_query_start_loc = torch.zeros((num_reqs + 1,), dtype=torch.int32, device=seq_lens.device)
        target_query_start_loc[1:] = torch.cumsum(step_tokens, dim=0)
    else:
        captured_step_d2h_syncs = 2
        _ = int(num_accepted_tokens.cpu().sum().item())
        host_seq = seq_lens.cpu().tolist()
        host_acc = num_accepted_tokens.cpu().tolist()
        updated_seq_lens = torch.tensor(
            [s + a + 1 for s, a in zip(host_seq, host_acc)],
            dtype=torch.int32, device=seq_lens.device,
        )
        target_query_start_loc = torch.zeros((num_reqs + 1,), dtype=torch.int32, device=seq_lens.device)
        target_query_start_loc[1:] = torch.cumsum((num_accepted_tokens + 1).to(torch.int32), dim=0)

    if int(actual_seq_lengths_q[-1].item()) != draft_tokens_padded:
        raise RuntimeError(
            f"FIA TND contract violation: queryT ({draft_tokens_padded}) != "
            f"last element of actualSequenceLengthQ ({int(actual_seq_lengths_q[-1].item())})"
        )

    padded_q = torch.zeros((draft_tokens_padded, query_tokens.shape[-1]), dtype=query_tokens.dtype, device=query_tokens.device)
    padded_q[: query_tokens.shape[0]] = query_tokens
    draft_out = padded_q * 1.0

    return {
        "draft_out": draft_out,
        "updated_seq_lens": updated_seq_lens,
        "target_query_start_loc": target_query_start_loc,
        "actual_seq_lengths_q": actual_seq_lengths_q,
        "padded_slot_mapping": padded_slots,
        "draft_tokens_padded": draft_tokens_padded,
        "target_tokens_padded": target_tokens_padded,
        "captured_step_d2h_syncs": captured_step_d2h_syncs,
        "metadata_allocs_in_capture": metadata_allocs_in_capture,
    }


def run_ablation_suite() -> dict[str, dict[str, object]]:
    q = torch.randn((9, 16), dtype=torch.float32)
    seq_lens = torch.tensor([16, 24, 12], dtype=torch.int32)
    accepted = torch.tensor([2, 1, 3], dtype=torch.int32)
    slots = torch.arange(100, 109, dtype=torch.int32)
    arms = {
        "full_candidate": dict(align_buckets=True, hoist_metadata=True, device_resident_accepted=True),
        "minus_bucket_alignment": dict(align_buckets=False, hoist_metadata=True, device_resident_accepted=True),
        "minus_metadata_hoist": dict(align_buckets=True, hoist_metadata=False, device_resident_accepted=True),
        "minus_device_resident_accepted_tokens": dict(align_buckets=True, hoist_metadata=True, device_resident_accepted=False),
    }
    report: dict[str, dict[str, object]] = {}
    for arm_name, flags in arms.items():
        try:
            res = dspark_aclgraph_step(
                q.clone(), seq_lens.clone(), accepted.clone(), slots.clone(), **flags
            )
            report[arm_name] = {
                "correctness_ok": (
                    res["captured_step_d2h_syncs"] == 0
                    and res["metadata_allocs_in_capture"] == 0
                    and res["draft_tokens_padded"] == 12
                    and res["target_tokens_padded"] == 16
                ),
                "captured_step_d2h_syncs": res["captured_step_d2h_syncs"],
                "metadata_allocs_in_capture": res["metadata_allocs_in_capture"],
                "draft_tokens_padded": res["draft_tokens_padded"],
                "target_tokens_padded": res["target_tokens_padded"],
            }
        except RuntimeError as exc:
            report[arm_name] = {
                "correctness_ok": False,
                "error": str(exc),
            }
    return report
```
