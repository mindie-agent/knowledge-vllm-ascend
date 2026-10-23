# Graded Unit Format Specification

[中文](package-format.md) | **English** · [Back to Homepage](../README.en.md)

This document defines the `mindie-graded-unit/1` layout, frontmatter, revision and reference rules, condition matching, inheritance, routing indexes, public feedback and reviewed changes used by `knowledge-vllm-ascend` on the `main` branch. The current C5 unit contract is `mindie-graded-unit/1`; the C6 reviewed projection uses `mindie-reviewed-change/1` (section 9). The format identifiers and validator revision are declared in [`publication-contract.json`](../publication-contract.json).

The `main` branch uses `mindie-publication-contract/2`. The legacy `tasks/` material layout is not part of that contract and is not active corpus content. Its archived bytes remain under `docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/`, on [`main`](https://github.com/mindie-agent/knowledge-vllm-ascend/tree/main) and at immutable historical commit [`85ab134f333b06ac3873df6affbf3bf6ec71978c`](https://github.com/mindie-agent/knowledge-vllm-ascend/tree/85ab134f333b06ac3873df6affbf3bf6ec71978c). The archive is not active unit content.

---

## 1. Why conditional units instead of long excerpts

A unit makes one reusable engineering conclusion searchable, checkable and revisable while retaining the premises under which it holds:

- the **body** carries the rule, procedure, code transformation, failure modes and limits;
- the **frontmatter** carries stable identity, tier, evidence strength, conditions, triggers, inheritance and provenance;
- the **revision** records semantic evolution under a stable ID and is the public reference key;
- the **content hash** is derived by the runtime for change detection and exact Git checks; it is not stored in frontmatter.

Retrieval checks conditions first and lexical relevance second. A hit returns a budgeted excerpt; the caller reads the full body through a revision-pinned reference. This avoids both dumping a whole document into context and letting a model summary replace the underlying conclusion.

A unit may be distilled from reviewed public material, source code, specifications or experiment records, including with model assistance. A model proposal is not knowledge: it must retain provenance and evidence limits and pass deterministic format/privacy checks and semantic review before it becomes a current unit. The unit body is not a verbatim transcript; byte preservation belongs to the old material on `main`/85ab.

---

## 2. Directory and file topology

```text
knowledge-vllm-ascend/
├── publication-contract.json                 # mindie-publication-contract/2
├── index/
│   ├── routing-manifest.json                 # mindie-routing-manifest/1 (derived)
│   └── shard-topology.json                   # mindie-shard-topology/1 (derived)
├── <domain_path>/
│   └── <unit_id>.md                          # mindie-graded-unit/1
├── feedback/
│   └── votes-<id>.json                       # mindie-graded-feedback/1
├── README.md
├── README.en.md
├── LICENSE
└── docs/
    ├── CONTRIBUTING.en.md
    ├── package-format.md
    ├── package-format.en.md
    └── evidence/
```

### Path constraints

- A unit path is exactly `<domain_path>/<unit_id>.md`. `main` publishes contract/2 content only; the old `material-v1` archive under `docs/archive/material-v1/` is evidence, not active corpus content.
- `<domain_path>` is a normalized repository-relative directory: forward slashes only, not absolute, no `.`/`..`, backslash or control characters. Its first segment cannot be `tasks`, `feedback`, `index`, `docs` or `private`.
- `<unit_id>` matches `unit-[a-z0-9]+(-[a-z0-9]+)*`. The stable ID does not change when the file moves.
- A unit is a plain non-executable Git blob (`100644`), UTF-8, LF line endings, at most 256 KiB.
- `index/` and `feedback/` are reserved directories. Documentation and evidence live under `docs/` and are not scanned as units.

---

## 3. Unit frontmatter

### 3.1 Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Fixed to `mindie-graded-unit/1` |
| `id` | string | Stable unit ID (`unit-` plus lowercase words); matches the file name |
| `revision` | integer | Positive integer; `1` for a new unit; increments on semantic change; unchanged on a move |
| `title` | string | Concise title naming the subject or conclusion |
| `category` | string | Must agree with `tier`: `axiom`, `measurement`, `sop`, `workflow`, `mutator` |
| `tier` | string | `L1`, `L2`, `L3`, `L4`; describes intended use, not confidence |
| `evidence_strength` | string | `hardware_spec`, `reproduced_empirical`, `single_observation`, `hypothesis` |
| `domain_path` | string | Normalized relative directory; path must equal `<domain_path>/<id>.md` |
| `conditions` | object | Non-empty mapping of checkable applicability premises |
| `triggers` | object | `error_signatures` and `keywords`, both string lists; may be empty |
| `inherits_from` | list | Stable IDs in the same corpus; empty when there is no inheritance |
| `provenance` | object | Source type, locator, exact digest, evidence references and supersession |

### 3.2 Tier and evidence strength

| Tier | Category | Intended content |
| --- | --- | --- |
| `L1` | `axiom` / `measurement` | A specification-backed constraint or a measurement reproduced under stated conditions |
| `L2` | `sop` | Diagnostic, development or verification steps under explicit premises |
| `L3` | `workflow` | A multi-step task process with dependencies |
| `L4` | `mutator` | A code transformation with source and verification evidence |

`L1` accepts only `hardware_spec` or `reproduced_empirical`. `reproduced_empirical` and `L4` require non-empty `provenance.evidence_refs`. Failed or unfinished executions (`compile_failed`, `infra_failed`, `measurement_invalid`, `in_flight`) cannot establish hardware or reproduced evidence.

Tier is not a confidence score. Do not raise it to make a unit easier to retrieve or seem more authoritative; evidence strength is not a substitute for tier.

### 3.3 Provenance

`provenance` includes at least:

| Field | Meaning |
| --- | --- |
| `source_type` | Source class, such as `ecosystem_repo`, `session_distill`, `ablation_receipt`, `run_manifest` |
| `source_locator` | Reviewable locator, either an `owner/repo@<40-hex>:path` Git locator or a repository evidence path |
| `source_digest` | Exact 64 lowercase hex SHA-256 of the source |
| `evidence_refs` | String list of evidence records, source line ranges or public links |
| `supersedes` | Stable IDs semantically superseded by this unit; empty when none |
| `experiment_status` | Optional; failed or incomplete states cannot support hardware/reproduced evidence |

Provenance records sources and evidence, not an "accepted" claim. A `review_reference`, a PR link or a passing format check never substitutes for the evidence.

### 3.4 Example

The following illustrates the field structure only. Angle-bracket values are placeholders and are not publishable; real units are parsed and validated by the runtime.

```yaml
---
schema: mindie-graded-unit/1
id: unit-example-sop
revision: 1
title: Example source-scoped procedure
category: sop
tier: L2
evidence_strength: single_observation
domain_path: integration/example
conditions:
  repository: owner/repo
  revision: <40-hex-commit>
triggers:
  error_signatures: []
  keywords:
    - example
    - procedure
inherits_from: []
provenance:
  source_type: ecosystem_repo
  source_locator: owner/repo@<40-hex-commit>:path/to/source.py
  source_digest: <64-hex-source-digest>
  evidence_refs:
    - docs/evidence/example-review.json
  supersedes: []
---
```

The body starts after the closing frontmatter delimiter. State the conclusion, procedure, failure conditions and limits of use. Do not repeat the field table in prose or replace executable detail with "see above".

---

## 4. Revision, content hash and references

### 4.1 Revision and content hash

The public identity is the integer `revision`, not a content digest. A new unit starts at `revision: 1`; a semantic change increments it; a location-only move keeps it unchanged.

The runtime derives a content hash from the unit bytes: take the frontmatter metadata (excluding `domain_path` and `revision`) and the body (trim outer whitespace, then add one trailing newline), hash canonical JSON (sorted keys, UTF-8, no extra whitespace) with SHA-256, and use the result for change detection and exact Git checks. The hash is not stored in unit frontmatter and is not the public reference key. Do not hand-write a digest or add the legacy `content_digest` field.

### 4.2 Revision

- New unit: `revision: 1`.
- Semantic change: increment `revision`; the runtime content hash changes with the content.
- Location-only move: keep the revision and content hash; only routing changes.
- Superseded revisions remain in Git history; the current corpus has one current file per stable ID.

Never overwrite bytes under an existing revision or silently redirect an old reference to new content. A replaced or withdrawn pinned reference must fail explicitly.

### 4.3 Reference form

```text
mindie://<domain>/units/<unit_id>@<revision>
```

For example:

```text
mindie://vllm-ascend/units/unit-vllm-ascend-custom-op-gates-import-retry-and-nz@3
```

- The integer after `@` is a positive revision, never a content hash.
- `knowledge_explain` reads an exact current unit revision; a stable `unit_ref` without `@` also reads current content.
- `knowledge_feedback` requires a pinned reference that the caller actually observed.
- The domain in `mindie://` selects the configured domain engine; an unknown or unconfigured domain fails instead of falling back to a default.

---

## 5. Condition matching

Every entry in `conditions` must hold before the unit applies. The runtime compares each premise with observed task facts. All hold → `matched`; any explicit contradiction → `mismatch`; missing, `unknown` or incomparable facts → `unknown`. Default queries exclude `unknown`; "not known" is never treated as "applicable".

### 5.1 Semantics

- **Scalar/list**: declared values are the allowed set. A scalar observation must be a member; every member of a list observation must be allowed.
- **Version specifier**: strings beginning with comparison operators (`<`, `>`, `=`, `!`, `~`) are compared as versions. An uninterpretable version is `unknown`, not a match.
- **`framework`**: either a version mapping (for example `{"torch": "2.10.0+cpu", "torch_npu": "2.10.0.post4"}`, compared recursively by key) or a list of requirement strings such as `torch>=2.10,<3`. The list form is conjunctive; a missing package, a conditional requirement (`marker`/`url`/`extras`) or an unparseable requirement is `unknown`.
- **Nested object**: compared recursively by the same rules; a non-object observation is `unknown`.
- **`exact_set`**: `{"exact_set": [...]}` requires the complete observed collection to equal the declared collection, unordered, member by member; collection order does not matter, while complete member identities and associations do. Members often bind shape, dtype, sample count and case digest; do not pair a shape from one case with a dtype or digest from another. A missing, extra, unknown or duplicate member does not match.
- **`any`**: explicitly declares no restriction for that key. It is not a substitute for a real condition and must not hide an unobserved fact.
- **`unknown`**: an `unknown` declared or observed value never matches.

### 5.2 Where query conditions come from

`knowledge_query` conditions contain only facts the current task actually observed. Do not:

- copy the unit's declared conditions into the query to force a hit;
- use a gold patch, oracle verdict, fixed code or held-out answer as an "observation";
- guess versions, commits, hardware or shapes to complete the set;
- treat "should apply" as "was observed".

A missing condition yielding `condition_unknown` is a correct result: the evidence is insufficient to decide applicability, not a retrieval bug. Only `matched` results enter the applicability decision; reading a full body does not establish applicability.

---

## 6. Inheritance

`inherits_from` names stable IDs in the same corpus. At index build time the runtime:

- resolves the complete closure, including indirect parents;
- rejects missing parents, duplicate IDs and cycles;
- records the closure of stable parent IDs for each unit.

At query time **every parent must independently match** the task conditions. A matching child never makes a non-matching parent applicable. A parent body is either injected complete or omitted with an explicit `inheritance_excluded` reason; premises are never silently truncated. Budget exclusion is also reported.

Inheritance does not cross domains. Cross-domain reuse uses an explicit `mindie://<domain>/...` route and that domain's configuration; do not copy another domain's unit or share its consent or inheritance graph.

---

## 7. Routing indexes (derived data)

`index/routing-manifest.json` uses `mindie-routing-manifest/1` and includes:

- `schema_version`, `generation`, `unit_count`;
- per unit: `id`, `title`, `tier`, `category`, `evidence_strength`, `domain_path`, `conditions`, `triggers`, `inherits_from`, `inherits_from_closure`;
- `path`, `revision`, `token_estimate`, `provenance`.

`generation` is the exact corpus state and the `base_generation` for a reviewed change. `index/shard-topology.json` uses `mindie-shard-topology/1` and reports leaf/subtree/federation unit counts, bytes, token estimates and over-limit signals. Those are maintenance signals, not automatic split/merge actions.

Both indexes are derived and must not be hand-edited. Rebuild or check them with:

```bash
python3 scripts/build_routing_index.py --root . --check
python3 scripts/build_routing_index.py --root .
```

`--check` validates freshness without writing. Without `--check`, the command validates the complete graph before an atomic write.

---

## 8. Public feedback (`mindie-graded-feedback/1`)

Public feedback lives at `feedback/votes-<id>.json` and contains only votes selected for publication:

```json
{
  "schema": "mindie-graded-feedback/1",
  "votes": [
    {
      "vote_id": "<opaque-id>",
      "root_id": "<opaque-root-id>",
      "unit_id": "unit-example-sop",
      "revision": 1,
      "rating": "up",
      "reason": "Optional single-line reason."
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `vote_id` | Opaque unique vote ID; non-empty, at most 256 characters |
| `root_id` | Opaque anonymous root ID; no local session identity; at most 256 characters |
| `unit_id` | Stable unit ID |
| `revision` | **Integer** revision, not a content hash |
| `rating` | `up` or `down` |
| `reason` | Optional single-line text, at most 1000 characters; must pass the privacy scan before publication |

Constraints:

- at most 500 votes per file; one current vote per `(root_id, unit_id, revision)`;
- a vote requires a revision actually observed by the caller;
- public totals count only votes for the current revision; older revisions remain local history and are not counted as current feedback;
- vote counts are descriptive, not confidence, proof of correctness or an automatic promotion gate. No feedback does not mean the unit is invalid.

---

## 9. Reviewed change (C6, `mindie-reviewed-change/1`)

C6 is not "a model proposal becomes a unit automatically". It is an explicit, reviewable local projection:

```json
{
  "schema": "mindie-reviewed-change/1",
  "review_reference": "review-record-or-pr",
  "base_generation": "<current routing generation>",
  "changes": [
    {
      "id": "unit-example-sop",
      "expected_content_sha256": null,
      "path": "integration/example/unit-example-sop.md",
      "content": "<complete rendered unit file>"
    }
  ]
}
```

Rules:

- `review_reference` records the review basis. It may be a review record or PR reference, but it is not a signature and does not prove independent review occurred.
- `base_generation` must equal the current `generation` in `index/routing-manifest.json`. If the corpus changed after review, the plan is rejected and must be re-reviewed.
- `content` is the **complete rendered unit file**, not a summary, diff or body excerpt. For a withdrawal, `content` is `null`.
- `expected_content_sha256` provides compare-and-swap: `null` for a new unit; the current `content_sha256` when editing an existing unit.
- A new unit must use `revision: 1`; a semantic edit must equal the old revision plus one; a location-only move keeps revision and content hash.
- Validation precedes the first write: the complete inheritance graph, duplicate IDs/paths, destination conflicts and revision rules are checked first.
- `--apply` writes the declared units, removes declared old files and rebuilds the routing index. It does not commit, push or publish.
- There is no automatic semantic acceptance, automatic promotion or "format passed means published". If application returns `partial` or `failed`, retain the actual state; do not retry blindly.

```bash
# Validate only; do not write.
python3 scripts/apply_reviewed_change.py --root . --plan reviewed-change.json

# Apply after semantic review.
python3 scripts/apply_reviewed_change.py --root . --plan reviewed-change.json --apply
```

---

## 10. Validation, privacy and publication facts

### 10.1 Deterministic checks

- `scripts/build_routing_index.py --check`: validates the unit graph and derived index freshness.
- `python3 -m mindie_knowledge.publication_check --repo . --revision <40-hex-commit> --domain vllm-ascend`: validates the complete publication tree with the installed validator; CI revalidates with the contract-pinned validator.
- `python3 -m mindie_knowledge.redact --check ...`: deterministic privacy profile `r3`.
- Pinned Gitleaks 8.30.1: the second privacy gate.

Privacy checks require zero findings. Do not add an allowlist, `.gitleaksignore`, `gitleaks:allow` or a modified scanner to suppress a real finding. If a documentation example triggers a false positive, keep the finding record and use equivalent harmless Markdown syntax.

### 10.2 `publication-head` split

On `main`, under `publication-contract/2`:

- **content**: unit Markdown, `feedback/**`, `index/**`; `success` after checks, handled by the content review bot under `mindie-content-review/2`.
- **development**: `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `docs/**`, `.github/**`, `publication-contract.json`; `neutral`, requiring maintainer review under the engineering design principles, never bot auto-merge.

### 10.3 Three separate facts

1. **Candidate tree validation**: format, content hash, inheritance, index consistency, privacy scans.
2. **Remote publication**: actual readback of `main`; `main` carries the current contract/2 declaration and does not change automatically with a candidate branch.
3. **Installed native acceptance**: whether a specific host can actually use query, explain, feedback and routing after installing that revision.

Passing candidate validation does not prove the latter two. The contract, CI checks and release/validation evidence each record the facts they cover. `main` may receive content updates within this compatibility version without changing the current `publication-contract/2` declaration. Formal assets, remote `main` publication and paid native acceptance are separate results; locally prepared content is not published proof.

---

## 11. Old material location

The old `material-v1` corpus is archived under `docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/` on [`main`](https://github.com/mindie-agent/knowledge-vllm-ascend/tree/main) and at immutable historical commit [`85ab134f333b06ac3873df6affbf3bf6ec71978c`](https://github.com/mindie-agent/knowledge-vllm-ascend/tree/85ab134f333b06ac3873df6affbf3bf6ec71978c). It is not moved or rewritten here.

The historical migration metadata records:

- `task_count`: 663
- `archived_file_count`: 2,387 (`retained_file_count`: 48 on `main`; `pruned_unreferenced_file_count`: 2,339 preserved at `85ab134f`)
- `block_count`: 1,723
- `feedback_count`: 1
- each old file retains its original `git_blob_oid`, on `main`/85ab.

The old material is not current unit content and is not re-labeled as graded knowledge. It preserves source traceability and history; the old `tasks/` layout does not re-enter the active `main` contract/2 corpus. `main` history and the immutable bytes at 85ab stay as they are; they are not regenerated content.
