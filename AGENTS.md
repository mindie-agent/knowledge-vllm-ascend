# `knowledge-vllm-ascend` — Repository Guidelines for AI Agents & Maintainers

This repository is the canonical domain knowledge feed for `vllm-ascend` in [MindIE Agent](https://github.com/mindie-agent/mindie-agent). It contains **graded knowledge units** (`mindie-graded-unit/1`), their provenance and applicability conditions, plus the derived routing index and public feedback. Runtime retrieval, capture, redaction and publication live in [`mindie-agent/knowledge`](https://github.com/mindie-agent/knowledge).

The old `tasks/` material layout is not part of `mindie-publication-contract/2`. The material-v1 bytes are preserved under `docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/` and described by `docs/evidence/material-v1-migration.json`. Do not mix that layout back into the active corpus or re-label it as graded knowledge.

---

## 1. Ownership and allowed paths

- **Units** live at exactly `<domain_path>/<unit_id>.md`. The path is part of routing, not identity: the stable `id` survives moves.
- **Derived indexes** live at `index/routing-manifest.json` and `index/shard-topology.json`. They are rebuilt from units; do not hand-edit them.
- **Public feedback** lives at `feedback/votes-<id>.json` and uses `mindie-graded-feedback/1`.
- **Metadata and documentation** live at the root (`README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `LICENSE`, `publication-contract.json`) and under `docs/**`.
- **Archived evidence** lives under `docs/archive/**` and `docs/evidence/**`. It is preserved historical data, not active retrieval content.
- **Out of scope for this repository's release claim:** other domains. They use the same runtime interfaces through explicit `mindie://<domain>/...` routes, not copies of this corpus.

Do not introduce a `tasks/` tree, a second unit layout, or a private/reserved directory (`index`, `feedback`, `docs`, `private`) as a `domain_path`.

---

## 2. Unit invariants

A unit is valid only when all of the following hold:

1. **Stable identity.** `id` matches `unit-[a-z0-9]+(-[a-z0-9]+)*` and does not change when the file moves. `domain_path` is a normalized repository-relative directory; the file path must equal `<domain_path>/<unit_id>.md`.
2. **Use-appropriate tier.** `tier` and `category` describe the same use:
   - `L1` → `axiom` or `measurement`
   - `L2` → `sop`
   - `L3` → `workflow`
   - `L4` → `mutator`
   Tier is not confidence and must not be raised to make a unit look stronger.
3. **Evidence strength.** `evidence_strength` is one of `hardware_spec`, `reproduced_empirical`, `single_observation`, `hypothesis`. `L1` accepts only `hardware_spec` or `reproduced_empirical`. Failed or unfinished executions (`compile_failed`, `infra_failed`, `measurement_invalid`, `in_flight`) cannot establish hardware or reproduced evidence.
4. **Explicit conditions.** `conditions` is a non-empty mapping of premises that can be checked before use. Readiness state, a known fix, a reviewer conclusion or a desired outcome is not a condition.
5. **Provenance.** `provenance.source_type`, `source_locator` and an exact `source_digest` (64 lowercase hex) are required. `evidence_refs` must be non-empty for `reproduced_empirical` and `L4`. `supersedes` lists stable IDs, not prose.
6. **Non-empty body.** The body states the rule, procedure or transformation and its limits. The frontmatter is metadata, not a substitute for the body.
7. **Exact content digest.** `content_digest` is `sha256:` plus the semantic digest over the canonical metadata-and-body projection, excluding `domain_path`, `revision` and `content_digest` itself. The routing manifest's `content_sha256` is the same digest without the prefix.
8. **Revision discipline.** A new unit starts at revision `1`. A semantic change increases the revision by one. A location-only move keeps the same revision and content hash. Never rewrite bytes under an existing revision.

`revision` is an integer. References pin the bare 64-hex `content_sha256` after `@`; do not confuse it with the integer revision or with a legacy block revision.

---

## 3. Conditions, evidence and claims

- Every declared condition must hold. Missing facts, explicit `unknown`, and uninterpretable versions do **not** match. They are reported as `condition_unknown` and excluded by default.
- `framework` conditions may be a version mapping compared recursively by package key, or a list of requirement strings such as `torch>=2.10,<3`. The list form is conjunctive; a missing package or conditional requirement is unknown, not satisfied.
- Ordinary lists name allowed alternatives. A scalar observed value must be one of them; when the observed value is a list, every member must be allowed.
- `exact_set` binds a complete unordered workload collection: collection order does not matter, while complete per-member identities and associations do. Do not pair a shape from one case with a dtype or digest from another. A partial set, an unknown member, or a changed member identity does not match.
- `any` explicitly declares no restriction for that key. It is not a substitute for recording a real condition.
- Conditions come from the consuming task's observed facts, never from gold patches, oracle verdicts, fix commits, held-out answers or a reviewer's conclusion.
- `single_observation` does not become `reproduced_empirical` because a later summary repeats it. One successful run does not establish a general rule.
- Source-scoped claims bind repository and revision. Measurement claims bind hardware, software, input and measurement conditions. Do not generalize beyond the recorded workload, model, task or development-time claim.

---

## 4. Inheritance and routing

- `inherits_from` names stable unit IDs in the same corpus. The runtime resolves the complete closure at index build time and rejects missing parents and cycles.
- Each inherited unit must independently match the task conditions. A child does not make a non-matching parent applicable.
- Inherited bodies are included complete or omitted with a visible reason; never truncate a premise silently. Budget exclusion is reported.
- Cross-domain reuse uses an explicit configured route and a pinned `mindie://<domain>/units/<id>@<sha256>` reference. Inheritance does not cross domains, and one domain's consent or unit is never copied into another.
- `index/routing-manifest.json` is the live inventory. `generation` identifies the exact reviewed corpus state; `index/shard-topology.json` describes shard pressure. Both are derived and must be rebuilt after unit changes.

---

## 5. Retrieval, explain and feedback contract

- `knowledge_query(query?, limit?, conditions?, continuation?)` searches current units. `limit` is 1–20. It returns a digest-pinned `ref`, stable `unit_ref`, separate `feedback_ref`, match status and reasons. Excerpts are budgeted; inherited rules are complete.
- `knowledge_explain(ref)` returns the full body for one current unit reference, plus frontmatter and closure. A pinned reference must match the current content hash; a replaced or withdrawn revision fails explicitly rather than returning different bytes.
- `knowledge_feedback(ref, rating, reason?)` accepts `up` or `down` only for a revision actually observed by the caller. The reason is optional, single-line and bounded. It is a usefulness signal, not a correctness judgment, vote weight or promotion gate.
- Query and explain observe a revision so that later feedback can bind it. Importing a feed does not mark anything observed.
- Reading a reference does not establish applicability. The caller must still compare conditions with task facts.

---

## 6. Reviewed change (C6) and publication

Authorized capture/redaction preserves selected material; it does not create or promote a graded unit. A draft, model proposal or review note is not a unit either. A C6 reviewed projection becomes canonical only when `python3 scripts/apply_reviewed_change.py` validates the complete authored change and, after review, writes the exact unit bytes and rebuilds the index.

The plan uses `mindie-reviewed-change/1` and must include:

- `review_reference`: the review basis. It records where approval came from; it is not an authentication signature and does not prove independent review.
- `base_generation`: the exact current routing generation. If the corpus changed after review, the change is rejected.
- `changes`: complete rendered units (not excerpts) or withdrawals, each with `id`, `expected_content_sha256`, `path` and `content`.

Validation covers the full inheritance graph, duplicate IDs/paths, revision rules and the destination path before any write. Application writes only the declared units, rebuilds the routing index and reports `applied`, `validated`, `failed` or `partial`; it does not commit, push or publish. There is no automatic semantic acceptance, no promotion of a proposal by a successful format check and no implicit retry after a partial local write. If a write fails, report the concrete state rather than retrying blindly.

`.github/workflows/check.yml` reads the publication contract from the trusted base, installs the pinned validator revision and validates candidate blobs without executing candidate code. In `publication-contract/2`:

- **content** changes are unit files, feedback files and derived index files;
- **development** changes are root documentation/configuration, `docs/**`, `.github/**` and `publication-contract.json`.

Content validation can be `success`; development validation is `neutral` and requires maintainer review. Neither outcome is proof of semantic truth, remote `main` readback or installed native acceptance. Those are separate release results.

---

## 7. Privacy and security

- All public text, unit metadata, feedback reasons, PR titles and PR descriptions must pass the runtime's deterministic redaction profile (`r3`) and the pinned Gitleaks scan with zero findings.
- Never commit credentials, private keys, internal addresses, machine identifiers, personal home paths, ticket IDs or non-public source identifiers. Use placeholders such as `/home/<user>/...` when a path shape matters.
- Do not suppress, allowlist or rewrite a scanner finding to make a check pass. If a documented example triggers a false positive, preserve the finding record and use equivalent harmless Markdown syntax that communicates the same fact.
- Do not add executable candidate code to a content PR. The publication check treats candidate files as Git blobs.
- Archived material and evidence may contain historical identifiers already reviewed for publication. Do not copy those identifiers into new prose or examples.

---

## 8. Validation commands

Run these with the pinned `mindie-knowledge` validator available. They are local checks; they do not publish anything.

```bash
# Validate the current unit graph and confirm the derived index is fresh.
python3 scripts/build_routing_index.py --root . --check

# Validate a reviewed change without writing (add --apply only after review).
python3 scripts/apply_reviewed_change.py --root . --plan reviewed-change.json

# Validate the committed publication tree with the pinned validator.
python3 -m mindie_knowledge.publication_check \
  --repo . --revision "$(git rev-parse HEAD)" --domain vllm-ascend

# Deterministic privacy scan over changed public text.
python3 -m mindie_knowledge.redact --check README.md AGENTS.md CONTRIBUTING.md docs
```

Use the installed pinned Gitleaks 8.30.1 binary for the second privacy gate. Do not download a different scanner, create a new environment for a documentation review, or treat a passing local check as installed native acceptance.

---

## 9. Preservation rules

- Keep source files, archived bytes, original Git blob identities and historical revisions intact when authoring documentation or units.
- Do not erase a failed receipt, an earlier public state or a superseded unit to make the current tree look cleaner.
- A documentation change must not silently change a unit's claim, conditions, evidence or revision. Use the reviewed-change path for unit edits.
- Do not copy content between domains or force a same-topic merge; use explicit routes.
