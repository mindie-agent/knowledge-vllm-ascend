# Review packet: DSpark ACLGraph/D2H finding 1

Status: pending independent review. Nothing here is approved, applied or published, and no NPU workload was executed.

## Artifacts

- Corrected evidence record: `docs/evidence/l4-dspark-mtp-aclgraph-d2h-elimination-ablation.json`
  - byte SHA-256: `75a8ce2460f3154c7988f661e9dc0e177655d4dad8fad0c775b93fe8d28baf05`
- Proposed unit revision 2: `docs/evidence/review-7168-dspark/proposed-unit-vllm-ascend-dspark-mtp-aclgraph-d2h-elimination-mutator.md`
  - byte SHA-256: `73cf54b01cc07c6bc759ac70eb393e36c6539641c8e8f30d7c4a733775ea0b57`
  - semantic `content_sha256`: `1ff527c83d582b2b4559f9b6b127ae28adfaac3a0efd6442e58ba812c108e8f5`
- Reviewed-change plan for validation only: `docs/evidence/review-7168-dspark/reviewed-change-plan.json`
  - byte SHA-256: `48284be66ef522a4145c1966cbfe9f2f27805a56b50edeb212f3f4e9a74311f8`
  - `expected_revision: 1` and `expected_content_sha256: d595f148d188e55b84b4bae10b118a098b28298350420db28cd89b003bbfb8cf` are the compare-and-swap values for the current revision-1 unit.
  - The embedded `content` is the complete rendered unit and matches the proposed-unit file byte for byte.

The live unit remains at revision 1 until a reviewer applies the plan. The evidence correction is already in the working tree because it is within this worker's owned scope.

## Validation performed

```bash
PYTHONPATH=<shared-knowledge-checkout> \
  python3 scripts/apply_reviewed_change.py --root . \
  --plan docs/evidence/review-7168-dspark/reviewed-change-plan.json
```

Observed result: `status: validated`, `base_generation: c197b86caf684d80f9281094fb540e5c3dae5e7f63947cb464682d782c8a4095`, projected `generation: db2dce1d9f069ee883e609a0f65b9a273e3fc3442029951c4465e6f2caf7b3ea`, one write, no removals, `publication: not_requested`. No `--apply` was run.

```bash
PYTHONPATH=<shared-knowledge-checkout> \
  python3 -m mindie_knowledge.redact --check \
  docs/evidence/l4-dspark-mtp-aclgraph-d2h-elimination-ablation.json \
  docs/evidence/review-7168-dspark
```

Observed result: `4 file(s), 0 finding(s)`.

```bash
gitleaks dir --no-banner --report-format json --report-path - --redact=20 \
  docs/evidence/review-7168-dspark
gitleaks dir --no-banner --report-format json --report-path - --redact=20 \
  docs/evidence/l4-dspark-mtp-aclgraph-d2h-elimination-ablation.json
```

Observed result: 0 findings for each targeted scan with the pinned Gitleaks 8.30.1. A repo-wide scan also reports a pre-existing `generic-api-key` false positive in the dirty derived `index/routing-manifest.json` on the `token_estimator` string; this packet does not edit or allowlist it.

## Review boundaries

- The derived `index/routing-manifest.json` and `index/shard-topology.json` are stale because other workers added unindexed units. They were not rebuilt here; the parent owns the shared rebuild.
- `base_generation` is computed from the current unit graph. If any unit is added or changed before application, this plan is stale and must be revalidated against the new generation. Do not bypass that check.
- The evidence record now has `experiment_status: not_run_source_only`, `evidence_strength: hypothesis`, no run identifiers and no measurement claims. The retracted assertions are retained as an audit trail, not as observations.
- The proposed unit keeps the L4 mutator tier, increments the revision to 2 and preserves the source-derived hypothesis, fallback boundaries, missing experiment and future measured-acceptance gates. Its body is 7229 bytes (<= 8192).
- Source-boundary correction: the v2 `AscendDSparkSpeculator` consumes `num_query_per_req` but delegates initialization to the upstream `DSparkSpeculator` via `super().__init__`; that initializer is not vendored in the pinned tree. The `sample_from_anchor` default and K-vs-K+1 formula are v1-only observations. The unit now requires the observed v2 `num_query_per_req` and makes `dspark_graph_path=v2_ascend_dspark_full_graph` a declared pre-use condition, excluding v1/eager and unrelated methods.
- Schema validation is not semantic approval. The reviewer must inspect the exact proposed bytes, the source anchors and the evidence record before applying the plan and rebuilding the index.
- Before applying, confirm the live unit is still at revision 1 with semantic `content_sha256` `d595f148d188e55b84b4bae10b118a098b28298350420db28cd89b003bbfb8cf`. The live unit currently binds `source_digest` `1f91bb...` (the previous evidence bytes); this plan replaces it with revision 2 and the new evidence digest. The installed validator compare-and-swaps on `expected_revision`; `expected_content_sha256` is included for the documented contract, but the plan alone would not catch a concurrent body-only edit that kept revision 1.
