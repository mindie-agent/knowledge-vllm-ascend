---
category: sop
conditions:
  framework:
    mindie-harness: 0.1.4
    mindie-knowledge: 0.13.24
  workload_scope:
  - knowledge_mcp
domain_path: _common
evidence_strength: single_observation
id: unit-mindie-knowledge-rpc-dispatch-compatibility
inherits_from: []
provenance:
  evidence_refs:
  - mindie-agent/harness@d23aa2f9c31cbe3bd7a029f7257e032c3cbfde46:mindie_harness/mcp/knowledge_rpc.py
  - mindie-agent/knowledge@9f48272a3ba87248bfda0e4068692511aac4dfa4:mindie_knowledge/__init__.py
  - mindie-agent/knowledge@9f48272a3ba87248bfda0e4068692511aac4dfa4:mindie_knowledge/loop/transport.py
  source_digest: 3eef9d692c58a1d29bee3c01f4fcbe73fee303bb738407e729b353ba7849d03d
  source_locator: mindie-agent/harness@d23aa2f9c31cbe3bd7a029f7257e032c3cbfde46:mindie_harness/mcp/knowledge_rpc.py
  source_type: software_integration_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: MindIE knowledge RPC dispatch requires loop-module compatibility
triggers:
  error_signatures:
  - knowledge_unavailable with submission_state=not_sent
  - installed mindie-knowledge is treated as unavailable by knowledge MCP
  keywords:
  - mindie-harness
  - mindie-knowledge
  - knowledge RPC
  - knowledge_unavailable
  - ensure_service
  - select_engine_config
  - loop.transport
---
# MindIE knowledge RPC dispatch requires loop-module compatibility

## Condition

Use this SOP for the MindIE knowledge MCP surface when the runtime pairs
mindie-harness 0.1.4 with mindie-knowledge 0.13.24 or an equivalent revision
pair, and a knowledge tool fails before submitting a request.

## Rule

mindie-knowledge keeps service APIs in loop modules. The package root is not a
stable export for rpc, Admission, or engine-routing functions. A compatible
harness dispatcher imports Admission from loop.activation, config_at and
ensure_service from loop.cli, select_engine_config from loop.domain_routes, and
rpc from loop.transport.

A dispatcher that only checks mindie_knowledge.rpc reports
knowledge_unavailable with submission_state=not_sent even when the package is
installed. An empty query result after a successful dispatch is a valid
no-match result and has a different meaning.

## Repair

1. Inspect the installed dispatcher source for the four loop-module imports.
2. Pin mindie-harness to a revision that implements that contract and keep the
   mindie-knowledge revision in the same tested compatibility set.
3. Update the dependency lock file together with the source pin.
4. Rebuild the locked runtime and restart the MCP server so the active process
   loads the new pair.
5. Confirm the active runtime is validated, then run a read-only knowledge query.

## Acceptance

The dispatcher imports the loop APIs, service startup reaches ensure_service,
the RPC call is submitted, and the tool returns isError=false. A no-match
response is recorded as an empty result set instead of an availability error.

## Limits

This card diagnoses dispatch compatibility. It does not establish that a
knowledge unit exists, that a returned unit applies to a workload, or that a
runtime upgrade has been activated until the active process is restarted.
