# Public material repackaging · 2026-10-04

This one-time, explicitly authorized conversion covers only the 20 entries
already public at commit `9aab5e1e2f1298f992b1b41e403c2b1e0232d12c`.
It imports no private transcript or earlier Git history and adds no runtime
compatibility path for the retired entry format.

Each entry keeps its identity, title, summary, conditions and exact published
body UTF-8 bytes, including the final newline. The new tree has 20 task
manifests and 22 blocks: 19 entries fit one block and one entry needs three.
Concatenating each manifest's ordered block bodies reproduces all 110,289
original body bytes. Public source paths, source commit and byte ranges are
recorded in the blocks. Every package passed the current schema, hash and
public-data rule checks.

Existing titles and summaries are reused as fallible indexes, including for
the three parts of the longest entry. There were **zero model calls** and no
new summary-quality or business-acceptance claim. `status: complete` means
index packaging is complete; the original successes, failures, corrections
and uncertainty remain exactly as written in the source body.

The single feedback file is byte-for-byte unchanged. Its vote still belongs
to the original entry revision, which was verified against the source entry.
It is not transferred to the new package revision.

The [machine-readable record](material-package-migration-2026-10-04.json)
maps each original path/revision to the new path/revision, records source-file,
new-manifest and exact-body SHA256 values, and includes feedback hashes. Git
retains the source files and the conversion diff for independent verification.
