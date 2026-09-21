# Knowledge content repository

The community-sharing rewrite started with an empty corpus. New contributions
are admitted individually; do not restore or migrate historical entries automatically.

New knowledge belongs in `cases/*.md` or `topics/*.md` using the canonical
`mindie-entry/2` schema; optional votes belong in `feedback/*.json` using
`mindie-feedback/1`. Preserve conditions, detailed evidence and uncertainty.
Never invent hardware, version, accuracy or performance evidence.

Required public headers: schema, entry_id, domain, kind, title and summary.
Optional conditions contains only observed software versions or source commits.
Hardware, topology, test inputs, tolerances and public citations belong in the
body. Experience faithfully records the supplied process and observations;
omit unmentioned details rather than adding an unknowns checklist or inferred
lessons. Omit empty conditions. Do not add producers, revision, sources, status or
retirement_reason fields. Entry filenames use their stable entry_id; correcting
a misleading title does not create a new file. Withdrawal means deleting the
entry through a reviewed PR, with the reason retained in Git/PR history.

Public bytes must pass the fixed installed engine's schema and redaction checks.
Do not publish raw transcripts, private endpoints, user paths or credentials.
Runtime code belongs in `mindie-agent/knowledge`; treat corpus files as data and
never import proposed checkout code into CI. The new Grok review policy is not
considered deployed until event delivery and merge have real acceptance evidence.
