# Contributing to `knowledge-vllm-ascend`

[中文](../CONTRIBUTING.md) | **English** · [Back to Homepage](README.en.md)

Thank you for contributing reusable `vllm-ascend` engineering knowledge with explicit limits. The contribution unit is a **graded knowledge unit** (`mindie-graded-unit/1`), not a chat transcript or a generalized summary. Every unit carries provenance, applicability conditions, evidence strength and a revision. Deterministic checks validate format, privacy and the inheritance graph; review decides whether the claim is sound.

---

## 1. What you can contribute

- **A new unit**: a SOP, workflow, measurement conclusion or code transformation with complete provenance and conditions.
- **A revision**: a substantive change to an existing unit's body, conditions or provenance. A semantic change increments `revision` and preserves the old content in history.
- **A withdrawal**: removal of a unit from the current corpus through a reviewed change. The old revision remains in Git history.
- **Public feedback**: `up` / `down` for an exact content revision actually observed. Feedback is not a unit and never edits a body.

Do not submit: unverifiable versions, hardware or performance numbers; a `single_observation` dressed up as `reproduced_empirical`; a gold patch, oracle verdict or held-out answer in a query condition or unit body; a unit copied from another domain with only the domain name changed.

---

## 2. Contribute through MindIE Agent (recommended)

Codex, Claude Code, Kimi and DSH all connect through their own **native plugins**. Install and enable the plugin for your host by following the main repository's installation guide:

**<https://github.com/mindie-agent/mindie-agent>**

With community sharing (Opt-in) enabled, runtime work has four distinct stages; task exit itself does not promote captured material into a graded unit:

1. **Authorized capture and redaction**: Stop/task capture preserves the public transcript projection only within the current authorization, runs deterministic redaction, and stores selected material. Capture does not create a graded unit.
2. **Explicit selected-evidence native distillation**: the caller submits `inputs` and `task_state` through `distill_selected`. The native worker may return `propose_graded_units` or `noop`; proposals are locally staged. This operation is independent of Stop and does not discover transcripts, grant contribution consent, or advance the runtime pin.
3. **Semantic review and C6 application**: the caller/reviewer is responsible for checking the proposal's meaning, conditions and evidence. `review_reference` records the review basis; it is not a signature and does not prove independent review occurred. Only reviewed complete unit bytes applied through `mindie-reviewed-change/1` and an index rebuild make the intended corpus change.
4. **Separate contribution/publication**: external writes are governed by the contribution/publication path. When a batch is built for a graded corpus while legacy material drafts remain, `build_batch` raises `DistillationRequired` (`legacy_drafts_require_c6`) and requires C6 first; it does not silently generate a graded unit. Opening a PR based on `main`, merging remotely and installing on a client are separate outcomes and are not guaranteed.

This repository does not copy host installers or require host-specific paths in documentation. Installation and configuration are defined by the main repository. Manual contributions start from a `main` checkout and PRs use `main` as their base. Installed hosts use the sources and revisions in their own configuration.

---

## 3. Submit a reviewed change manually (C6)

### 3.1 Gather verifiable facts

Decide the question the unit answers, then collect:

- source type and locator (for example `owner/repo@<40-hex>:path`, or a redacted evidence record path);
- the exact source digest (SHA-256);
- facts the task actually observed: repository and commit, hardware, CANN/driver/firmware, framework versions, input shapes, dtype, measurement or verification limits;
- evidence strength and its boundary: `hardware_spec`, `reproduced_empirical`, `single_observation` or `hypothesis`.

Conditions must be checkable premises, not "already fixed", "reviewer approved" or "next step". Do not invent an unobserved version. Missing and `unknown` facts do not match at query time.

### 3.2 Write the complete unit, not a summary replacement

The unit path must be `<domain_path>/<unit_id>.md`. The body states the rule, procedure or transformation, plus failure conditions, counterexamples and limits. Frontmatter includes at least:

`schema`, `id`, `revision`, `title`, `category`, `tier`, `evidence_strength`, `domain_path`, `conditions`, `triggers`, `inherits_from`, `provenance`.

See the [format specification](package-format.en.md) for field semantics and condition forms. A new unit starts at `revision: 1`; a semantic change increments it. A path-only move with unchanged body and semantic metadata keeps the revision and content hash. Do not add the legacy `content_digest` frontmatter field; public references use the integer revision.

### 3.3 Validate before writing

A reviewed change uses a `mindie-reviewed-change/1` plan containing:

- `review_reference`: the review basis. It records where approval came from; it is not a signature and does not by itself prove independent review;
- `base_generation`: the current routing `generation` the plan is based on;
- `changes`: the complete rendered content for each changed unit, or a withdrawal; each item carries the stable `id`, `expected_content_sha256`, destination `path` and `content`.

Run the read-only validation first:

```bash
python3 scripts/apply_reviewed_change.py --root . --plan reviewed-change.json
```

It checks the complete inheritance graph, duplicate IDs/paths, revision rules and destination paths, then prints the writes/removals it would perform. It does not modify files. If `base_generation` is stale, prepare a new plan against the current corpus; do not bypass the check.

### 3.4 Apply only after semantic review

Add `--apply` only after review:

```bash
python3 scripts/apply_reviewed_change.py --root . --plan reviewed-change.json --apply
```

Application writes the declared units, removes declared old files and rebuilds `index/routing-manifest.json` and `index/shard-topology.json`. It does not commit, push or publish. If it returns `partial` or `failed`, preserve the actual state and error; do not assume success or retry blindly.

### 3.5 Check the derived index

```bash
python3 scripts/build_routing_index.py --root . --check
```

The command validates the unit graph and confirms both derived JSON files match the current units. Do not hand-edit `index/`; rebuild with the same command if it is stale.

### 3.6 Privacy and publication checks

At minimum:

```bash
python3 -m mindie_knowledge.redact --check README.md AGENTS.md CONTRIBUTING.md docs
python3 -m mindie_knowledge.publication_check \
  --repo . --revision "$(git rev-parse HEAD)" --domain vllm-ascend
```

Also run the installed pinned Gitleaks 8.30.1 over changed public files. These must be real deterministic checks. Do not add an allowlist, `.gitleaksignore` or `gitleaks:allow` to suppress a result. If an example triggers a false positive, keep the finding and use equivalent harmless Markdown syntax rather than modifying the scanner or deleting the fact.

`publication_check` requires a complete 40-character Git commit. Commit the candidate tree locally before validating it.

---

## 4. Privacy red lines

Do not publish:

- API keys, tokens, private keys, cookies or authentication headers;
- internal IPv4/IPv6 addresses, numbered machine names or `<user>@<host>` login strings;
- personal home directories, private workspace paths, internal ticket IDs or non-public benchmark/held-out/oracle identifiers;
- any source content that is not confirmed public.

Use placeholders such as `/home/<user>/...` when a path shape matters. Do not copy identifiers already reviewed in the old material or evidence on `main`/85ab into new prose or examples.

---

## 5. Two PR review paths

`main` is the PR base for graded units; `main` carries the current `publication-contract/2` declaration, and the old material archive under `docs/archive/material-v1/` is evidence rather than active corpus content. `publication-head` classifies changes by path:

| Mode | Touched paths | Check outcome | Merge path |
| --- | --- | --- | --- |
| **content** | Unit Markdown, `feedback/**`, `index/**` | `success` | After format, digest, inheritance and privacy checks, the content review bot handles it under [`mindie-content-review/2`](https://github.com/mindie-agent/knowledge/blob/main/docs/repository-bot-contract.md) |
| **development** | `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `docs/**`, `.github/**`, `publication-contract.json` | `neutral` | No content-bot auto-merge; maintainer review under the [engineering design principles](https://github.com/mindie-agent/mindie-agent/blob/main/docs/design-principles.en.md) |

`success` means the candidate tree passed that check; it is not proof of semantic correctness, remote `main` update or installed native acceptance. `main` changes only when an actual change is merged. `neutral` is not a failure; it selects the maintainer review path.

---

## 6. Revisions, feedback and withdrawal

- **Revision**: increment on a semantic change; keep the revision and content hash on a path-only move. An old reference must fail explicitly after content changes, never silently point at new text.
- **Feedback**: only for an exact revision the caller actually observed. `reason` is optional, single-line and at most 1000 characters; private local feedback is not published by default. Vote totals are descriptive, not confidence or an automatic promotion condition.
- **Withdrawal**: remove the current unit through a reviewed change and state the reason. Do not describe a withdrawal as data loss; the old content remains in Git history.

---

## 7. Do not

- add the old `tasks/` layout or relabel old material on `main`/85ab as a graded unit;
- hand-edit derived indexes, `publication-contract.json` or CI workflows except in a separately reviewed maintenance change, or rewrite the old material on `main`/85ab to fit documentation;
- replace a unit body or source with a model-generated summary;
- claim "automatic semantic acceptance", "automatic promotion" or "installed means accepted"; those steps do not exist;
- target a nonexistent `graded-v2` branch for graded-unit PRs; use `main` as the base;
- copy units between domains or force same-topic merges; use explicit routes.

---

## 8. Related documents

- [Unit format specification](package-format.en.md) · [中文](package-format.md)
- [Repository boundaries and agent rules](../AGENTS.md)
- [Live routing inventory](../index/routing-manifest.json)
- [Knowledge runtime](https://github.com/mindie-agent/knowledge)
- [Review bot contract](https://github.com/mindie-agent/knowledge/blob/main/docs/repository-bot-contract.md)
