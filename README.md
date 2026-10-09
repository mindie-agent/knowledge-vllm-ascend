# MindIE Agent · vLLM-Ascend knowledge

Part of [MindIE Agent](https://github.com/mindie-agent/mindie-agent).

This repository distributes mechanically redacted task material as ordinary
Markdown files. Each `tasks/<task_id>/` package contains an `index.md` navigation
manifest and ordered `blocks/*.md` files. The manifest binds every block hash and
carries fallible retrieval titles and summaries. Failed attempts, later
corrections and unresolved claims remain part of the reference material.

Consumers use the published files and their existing summaries directly. The
runtime builds a local ReMe search index without another model summarization
pass. Summaries and retrieval scores do not certify factual correctness.

The 20 entries already public at the recorded source commit were explicitly
[repackaged once](docs/material-package-migration-2026-10-04.md) into 20 task
manifests and 22 blocks. Their 110,289 body bytes, titles, summaries, conditions
and entry identities were preserved. Existing feedback retains its original
revision attribution. No new model summary or business verification was added;
`status: complete` records completed index packaging only. Private material and
older Git history are not automatically imported. Optional explicit feedback
belongs under `feedback/`; an empty corpus remains valid. Public Git history is
retained for provenance, and consumers follow this repository's main branch. The retired
`knowledge/vllm-ascend` distribution branch is not a supported source.

Publication checks validate the exact Git commit, canonical package structure,
all referenced blocks and hashes, indexing readiness and public-data redaction.
The trusted workflow reads the immutable runtime validator from the base commit's
[publication contract](publication-contract.json). It distinguishes ordinary
content from development changes and attaches its result to the exact head.
Contract and Bot-rule changes require development review. Implementation and
local tests remain distinct from remote CI, delivery and merge acceptance.

Runtime implementation: [mindie-agent/knowledge](https://github.com/mindie-agent/knowledge).
Codex plugin: [mindie-agent/mindie-agent-codex](https://github.com/mindie-agent/mindie-agent-codex).

License: MIT.

The [repository Bot operating contract](https://github.com/mindie-agent/knowledge/blob/main/docs/repository-bot-contract.md)
uses `mindie-content-review/2`. A correction that changes body bytes needs a new
block identity and a recomputed manifest; metadata-only corrections preserve the
body identity. Withdrawal removes the complete package. Later contributions use
the confirmed remote version and unsent blocks, preserving maintainer corrections.

A consumer pins this declaration's SHA256 as part of its product combination.
Content may advance under the same declaration; a changed declaration requires a
matching plugin combination. Missing or mismatched declarations are explicit
failures, not empty corpora. Release order is runtime, content and Bot contract,
then adapter. Local checks do not establish external Bot adoption or native use.
