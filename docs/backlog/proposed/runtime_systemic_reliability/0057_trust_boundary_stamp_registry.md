# Proposed: Declarative trust-boundary argument-stamp registry

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
Two trust-boundary argument stamps exist, both correct, both single-seam,
both HAND-ROLLED in the tool-calls planning loop:
- shell `_registry_namespace` (effect_handlers.py:~3000) — run-id stamped,
  model-supplied values overwritten;
- agora `_agora_agent` (effect_handlers.py:~3009-3020) — run-var alias
  stamped, strip-vs-stamp semantics, schema-hidden via hide_args.
Workspace path rewriting is a third authority seam (one call site,
workspace_scoped_tools.rewrite_tool_arguments). Approval-resume replays the
STAMPED calls (pinned both for shell and agora).

## Current code reality
The pattern is copy-paste-with-variations; nothing asserts that every
schema-hidden argument HAS a stamp entry.

## Problem or opportunity
The risk is additive: the NEXT namespace/identity-sensitive toolset must
remember this block exists. The fourth toolset shipping without its stamp is
a cross-run identity or namespace leak discovered in production.

## Proposed direction
A declared table `{tool_name → (arg_name, source, on_missing)}` consumed by
the planning loop (source ∈ {run_id, run_var(<path>), ...}); the two
existing stamps become entries; PLUS a test asserting every hide_args-hidden
argument across registered toolsets has a stamp entry (the un-forgettable
half).

## Why it might matter
Makes the third and fourth stamps one-line additions with inherited pins;
converts a review-memory rule into a checked invariant.

## Promotion criteria
The next toolset with a trust-boundary argument (MCP per-run credentials,
a second comms identity, etc.) — build the registry WITH it.

## Validation ideas
Existing shell + agora pins re-run against the registry path unchanged;
the completeness test fails when a hidden arg lacks an entry.

## Non-goals
No change to stamp SEMANTICS (overwrite vs strip stays per-entry policy).

## Guidance for future agents
Read the agora blank-alias and approval-resume pins first — the registry
must reproduce those exact behaviors, not approximate them.
