<h1 align="center">MindIE Knowledge · vLLM-Ascend</h1>

<p align="center">
  <strong>Conditional engineering knowledge for Ascend NPUs and <code>vLLM-Ascend</code></strong><br>
  <sub>Part of <a href="https://github.com/mindie-agent/mindie-agent">MindIE Agent</a> · Local retrieval, exact full-body reads and revision feedback via <a href="https://github.com/mindie-agent/knowledge"><code>mindie-knowledge</code></a></sub>
</p>

<p align="center">
  <a href="https://github.com/mindie-agent/mindie-agent"><img src="https://img.shields.io/badge/MindIE%20Agent-Ecosystem-7045f5.svg" alt="MindIE Agent Ecosystem"></a>
  <a href="https://github.com/mindie-agent/knowledge"><img src="https://img.shields.io/badge/runtime-mindie--knowledge-0284c7.svg" alt="mindie-knowledge"></a>
  <a href="../publication-contract.json"><img src="https://img.shields.io/badge/domain-vllm--ascend-0f766e.svg" alt="Domain vllm-ascend"></a>
  <a href="../AGENTS.md"><img src="https://img.shields.io/badge/schema-mindie--graded--unit%2F1-0f766e.svg" alt="Graded unit schema"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/corpus-CC--BY--4.0-0d9488.svg" alt="License CC-BY-4.0"></a>
</p>

<p align="center">
  <a href="../README.md">中文</a> · <strong>English</strong> · <a href="#-connect">Connect</a> · <a href="#-why-graded-units">Why</a> · <a href="#-query-and-read">Query</a> · <a href="#-current-units">Current units</a> · <a href="package-format.en.md">📐 Format</a> · <a href="CONTRIBUTING.en.md">🤝 Contributing</a>
</p>

---

## What this repository is

The `graded-v2` branch of this repository is the `vllm-ascend` domain graded knowledge source for MindIE Agent `0.0.3`. It stores **graded knowledge units** (`mindie-graded-unit/1`): a reusable rule, procedure or code transformation together with its provenance, applicability conditions, evidence strength and revision.

A unit is not a chat summary and not a universal best practice. It answers two questions:

- under which **checkable premises** does this conclusion hold;
- which source and evidence support it, and how strong that evidence is.

At query time the runtime compares the task's observed facts with each declared condition. A condition that is unmet, missing or unknown is never treated as "close enough".

> The `graded-v2` branch uses `mindie-publication-contract/2` and `mindie-graded-unit/1`. `main` retains the old material-v1 corpus and old contract bytes for 0.0.2 consumers and is not a graded tree. [`index/routing-manifest.json`](../index/routing-manifest.json) is the live inventory; the table below lists the units present in the manifest at this check, not a fixed count.

---

## ⚡ Connect

MindIE Agent supports **Codex, Claude Code, Kimi and DSH**, each through its own native plugin. Follow the main repository's installation guide for your host:

**<https://github.com/mindie-agent/mindie-agent>**

Once installed, the plugin connects to the [`mindie-knowledge`](https://github.com/mindie-agent/knowledge) runtime, syncs this domain and builds a local index. Read-only `knowledge_query` / `knowledge_explain` does not require community sharing; capture and contribution remain explicit user choices.

This repository does not copy host installers or local paths. Host installation, plugin configuration and upgrades are defined by the main repository. 0.0.3 plugins select the `graded-v2` branch and bind the exact digest of this domain's `publication-contract.json`; contribution PRs use `graded-v2` as their base.

---

## ✨ Why graded units

| Summaries alone | Units in this repository |
| --- | --- |
| Conclusions lose versions, premises and failure branches | The body keeps concrete conditions, steps, counterexamples and limits; provenance binds the source |
| A search hit does not tell you whether it fits your environment | Query compares observed task facts first; mismatch or unknown is reported explicitly |
| Old conclusions are silently overwritten | Revision, content digest and pinned references preserve old observations; replaced references fail explicitly |
| You see only an excerpt and cannot verify the whole claim | `knowledge_query` returns summarized candidates; `knowledge_explain` reads the full body by exact digest |

A unit may be distilled from reviewed public material, source code, specifications or experiment records. Model assistance is allowed, but **model output is only a proposal**: it must retain provenance and evidence limits and pass deterministic format/privacy checks and semantic review before a reviewed change admits it. A failed experiment cannot be relabeled `reproduced_empirical`; hardware, versions and performance data cannot be invented; model rewriting cannot replace the source.

The original `material-v1` task material is not in the `graded-v2` branch; it remains on `main` and at immutable historical commit 85ab, described under [Old material and history](#-old-material-and-history). That material is evidence, not active retrieval content, and is never relabeled as a graded unit.

---

## 🔎 Query and read

The normal flow is: query first, read the full body by pinned reference, then optionally record feedback.

### 1. Query with facts the task actually observed

`knowledge_query` accepts a query string, an optional condition mapping and a result limit:

```python
knowledge_query(
    query="custom op import registration NZ selection",
    conditions={
        "repository": "vllm-project/vllm-ascend",
        "revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    },
)
```

`conditions` contains only facts the current task **actually observed**: software versions, source commits, hardware, input shapes and similar. Do not copy a unit's own declared conditions into the query to force a match, and do not use a gold patch, oracle verdict, held-out answer or post-fix conclusion as an observed fact.

Missing conditions, `unknown` values and incomparable versions produce `condition_unknown` and are excluded by default. That is correct behavior, not a retrieval failure. All declared conditions holding is `matched`; any explicit contradiction is `mismatch`.

### 2. Read the full body through the returned reference

A query result carries a digest-pinned `ref`, a stable `unit_ref` and a separate `feedback_ref`. `knowledge_explain` returns the full body, frontmatter, inheritance closure and applicability note for that exact revision:

```python
knowledge_explain(
    ref="mindie://vllm-ascend/units/unit-vllm-ascend-custom-op-gates-import-retry-and-nz@025b36835f18326dda7ef09ea380c8a4da9fd8d34eb3f39841010f302e3501ba",
)
```

The hash after `@` is the **bare 64-character content digest** (`content_sha256`), not the revision number. After a semantic edit, an old reference fails explicitly instead of silently returning new bytes. A location-only move keeps the digest and revision.

Reading a reference only means you received that content; it does not make the unit applicable. Compare its conditions with the task facts.

### 3. Record local usefulness feedback

After testing a unit in real work, optionally submit `up` / `down`:

```python
knowledge_feedback(
    ref="mindie://vllm-ascend/units/unit-vllm-ascend-custom-op-gates-import-retry-and-nz@025b36835f18326dda7ef09ea380c8a4da9fd8d34eb3f39841010f302e3501ba",
    rating="up",
    reason="Recorded after reproducing the import gate difference in the observed task.",
)
```

- Feedback binds the **exact content revision actually returned**; an unobserved revision cannot be rated.
- `reason` is optional, single-line and at most 1000 characters. Private local feedback is not published by default.
- Feedback is a descriptive signal, not proof of correctness, confidence or an automatic promotion gate. No feedback does not mean a unit is invalid or unused.
- Public feedback uses `mindie-graded-feedback/1` and counts only votes for the current content digest. Votes for older digests remain distinct local history.

### 4. Explicit domain routes

Cross-domain knowledge is read through an explicit reference such as `mindie://operators/units/<unit-id>@<sha256>`. The domain in the reference selects the configured domain engine; an unknown or unconfigured domain fails explicitly instead of falling back to a default.

The runtime can host arbitrary domains. The `0.0.3` release scope covers `vllm-ascend` and `operators`, but that is not a hardcode of the runtime to exactly two domains. This repository does not copy `operators` units or force same-topic merges across domains.

---

## 📚 Current units

[`index/routing-manifest.json`](../index/routing-manifest.json) is the only live inventory. It records stable IDs, paths, tiers, evidence strength, conditions, content digests, inheritance closures and token estimates. The table lists the units present in the manifest at this check; the manifest remains authoritative. A new unit becomes current only when its Markdown file and rebuilt manifest both exist.

| Unit | Tier / category | Evidence strength | Pinned reference |
| --- | --- | --- | --- |
| [`unit-vllm-ascend-custom-op-gates-import-retry-and-nz`](../integration/custom-ops/unit-vllm-ascend-custom-op-gates-import-retry-and-nz.md) | L2 / `sop` | `single_observation` | `mindie://vllm-ascend/units/unit-vllm-ascend-custom-op-gates-import-retry-and-nz@025b36835f18326dda7ef09ea380c8a4da9fd8d34eb3f39841010f302e3501ba` |
| [`unit-vllm-ascend-idle-dp-gdn-state`](../vllm-ascend/runtime/idle-dp-gdn-state/unit-vllm-ascend-idle-dp-gdn-state.md) | L2 / `sop` | `hypothesis` | `mindie://vllm-ascend/units/unit-vllm-ascend-idle-dp-gdn-state@72f2bfadc7899c951e711b81719ce6b5854cbfe6dd343b00bc35d75b221deb00` |
| [`unit-vllm-ascend-qwen35-gated-deltanet-decode-adaptation`](../vllm-ascend/runtime/qwen35-gated-deltanet-decode/unit-vllm-ascend-qwen35-gated-deltanet-decode-adaptation.md) | L2 / `sop` | `single_observation` | `mindie://vllm-ascend/units/unit-vllm-ascend-qwen35-gated-deltanet-decode-adaptation@33661ebd978f03c8355b1728e082fb8937bb5bf245bea38557d069def80ef14c` |

These are source-scoped L2 SOPs, not hardware performance results. The custom-op unit binds source revision `175ef4d49c2072f9233619636f0309f4f88b4e2c` of `vllm-project/vllm-ascend`; the idle-DP GDN unit is a `hypothesis` with base commit `c3394206bdd00b132580ae439555b4f1a70d5bf6` and model/state context; the Qwen3.5 unit is a `single_observation` of the historical PR7109 and binds model and source revisions. Do not upgrade a hypothesis or single observation to reproduced measurement; exact conditions are in each unit's frontmatter and body.

---

## 🗂️ Repository layout

```text
knowledge-vllm-ascend/
├── publication-contract.json           # contract/2, formats and validator pin
├── AGENTS.md                           # maintainer and AI-agent boundaries
├── CONTRIBUTING.md                     # contribution and review flow
├── index/
│   ├── routing-manifest.json           # derived live routing inventory
│   └── shard-topology.json             # derived shard topology
├── <domain_path>/
│   └── <unit_id>.md                    # body + frontmatter; path bound to stable ID
├── feedback/
│   └── votes-<id>.json                 # mindie-graded-feedback/1 public feedback
└── docs/
    ├── README.en.md
    ├── CONTRIBUTING.en.md
    ├── package-format.md
    ├── package-format.en.md
    └── evidence/                       # sources and review records
```

A unit path must be `<domain_path>/<unit_id>.md`. `domain_path` is a normalized repository-relative directory used for routing and sharding; it is not part of stable identity. `unit_id` starts with `unit-` and uses lowercase words and hyphens. The two JSON files under `index/` are derived and rebuilt by the runtime; do not hand-edit them. This branch publishes contract/2 content only; the old `material-v1` local archive copy is not in `graded-v2`.

The complete field, digest, condition, inheritance, feedback and reviewed-change rules are in the [format specification](package-format.en.md).

---

## 🗃️ Old material and history

The old `material-v1` corpus is not in the `graded-v2` branch. It remains on [`main`](https://github.com/mindie-agent/knowledge-vllm-ascend/tree/main) for 0.0.2 consumers and at immutable historical commit [`85ab134f333b06ac3873df6affbf3bf6ec71978c`](https://github.com/mindie-agent/knowledge-vllm-ascend/tree/85ab134f333b06ac3873df6affbf3bf6ec71978c). `graded-v2` does not move, rewrite or relabel that material, and it does not keep a local archive copy.

The historical migration metadata records 663 task packages, 2,387 files, 1,723 blocks and 1 feedback file; their original blob identities remain on `main` and 85ab.

- The old material is **not** active graded-unit content; the `mindie-graded-unit/1` corpus does not mix in the old `tasks/` layout.
- The old material is not graded knowledge and is not presented as a current conclusion.
- `main` history and the immutable bytes at 85ab stay as they are; they are not regenerated content.
- Do not edit the old material on `main`/85ab, the migration metadata, or blob identities to make the current branch look tidier.

---

## ✅ Publication and validation

On `graded-v2`, [`publication-contract.json`](../publication-contract.json) declares `mindie-publication-contract/2`, the data formats and the immutable validator revision used by publication checks. `.github/workflows/check.yml` reads that contract from the branch trusted base, installs the exact validator and validates candidate content as Git blobs without executing candidate code.

Keep three facts separate:

1. **Local/candidate tree validation**: whether format, digests, inheritance, index consistency and privacy scans pass;
2. **Remote publication**: the actual readback of `graded-v2`; `main` retains the old material corpus and old contract bytes and does not change with this branch.
3. **Installed native acceptance**: whether a specific host can actually use query, explain, feedback and routing after installing that revision.

A validated candidate tree does not prove the latter two, and a documentation change does not become an automatic content merge. `graded-v2` may receive content updates within this compatibility version while keeping the same `publication-contract/2` bytes; `main` does not change with them. Formal assets, remote `graded-v2` publication and paid native acceptance are separate results; locally prepared content is not published proof.

Documentation and engineering files follow the development review path; units, feedback and derived indexes follow the content path. The two paths have different check outcomes and merge rules; see the [contributing guide](CONTRIBUTING.en.md).

---

## 📖 Documentation and related repositories

| Topic | English | 中文 |
| --- | --- | --- |
| **Unit format specification** | [package-format.en.md](package-format.en.md) | [package-format.md](package-format.md) |
| **Contributing and review** | [CONTRIBUTING.en.md](CONTRIBUTING.en.md) | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| **Repository boundaries** | [AGENTS.md](../AGENTS.md) | — |
| **Live unit inventory** | [index/routing-manifest.json](../index/routing-manifest.json) | — |
| **Knowledge runtime** | [English README](https://github.com/mindie-agent/knowledge/blob/main/README.en.md) | [mindie-agent/knowledge](https://github.com/mindie-agent/knowledge) |
| **Review bot contract** | [repository-bot-contract.md](https://github.com/mindie-agent/knowledge/blob/main/docs/repository-bot-contract.md) | [repository-bot-contract.zh.md](https://github.com/mindie-agent/knowledge/blob/main/docs/repository-bot-contract.zh.md) |
| **Main installation guide** | [mindie-agent](https://github.com/mindie-agent/mindie-agent) | — |

---

## 📄 License

Public knowledge units and documentation in this repository are licensed under [Creative Commons Attribution 4.0 International (CC-BY-4.0)](../LICENSE). Units are engineering reference material, not a substitute for verification in your current environment; applicability always depends on observed task facts and the unit's declared conditions.
