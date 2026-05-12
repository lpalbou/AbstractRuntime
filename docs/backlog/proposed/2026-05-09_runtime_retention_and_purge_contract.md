# Proposed: Runtime Retention And Purge Contract

## Metadata
- Created: 2026-05-09
- Status: Proposed
- Completed: N/A

## Context

AbstractRuntime is the durable graph ledger. Durability is still the default, but hosts such as
AbstractGateway need a principled way to handle temporary/private runs created by authoring tools
like AbstractFlow.

Current Runtime reality:

- `RunStore` exposes `save()` and `load()`.
- `QueryableRunStore` exposes listing and child traversal.
- `LedgerStore` is append-only with `append()` and `list()`.
- `ArtifactStore` implementations already have delete helpers such as `delete_by_run()`.
- There is no generic retention class, purge protocol, or host-neutral cleanup helper for a run tree.

## Why this might matter

Gateway can implement cleanup itself, but the concepts are generic:

- deleting or archiving a root run and all child/subworkflow runs;
- deleting ledgers and artifacts for a run tree;
- preserving append-only production ledgers while allowing explicitly temporary authoring runs to
  expire;
- hiding private runs from default indexes.

If every host invents its own cleanup behavior, run history, artifacts, and child runs can drift.

## Proposed direction

Add optional runtime-level retention/purge contracts:

- Run metadata convention:
  - `run.vars["_runtime"]["purpose"]`;
  - `run.vars["_runtime"]["visibility"]`;
  - `run.vars["_runtime"]["retention"]`;
  - `expires_at` or `ttl_s` where appropriate.
- Optional storage protocols:
  - `DeletableRunStore.delete(run_id)`;
  - `DeletableLedgerStore.delete(run_id)` or `delete_by_run(run_id)`;
  - query helpers for expired/private/draft runs.
- Runtime helper:
  - collect a run tree from a root;
  - purge or archive run checkpoints, ledgers, and artifacts;
  - return a structured cleanup report.
- Documentation:
  - durability remains default;
  - purge is allowed only for explicitly temporary/private runs unless a host chooses otherwise.

## Evidence to gather before promotion

- Determine whether JSON file and SQLite stores can implement deletion safely without weakening
  provenance guarantees for production runs.
- Decide whether "private" means hidden, encrypted, or merely excluded from default listings in v0.
- Decide whether purge should be hard-delete or archive-to-artifact for auditability.
- Check Gateway's needs for draft Flow tests and backlog/UAT workspaces before finalizing fields.

## Validation ideas

- Store contract tests for JSON, SQLite, and in-memory implementations.
- Runtime helper tests:
  - root + child run tree cleanup;
  - artifact cleanup;
  - no accidental cleanup of non-temporary runs unless explicitly forced.
- Gateway integration tests once promoted.

## Non-goals

- Do not make Runtime responsible for Gateway's retention policy defaults.
- Do not remove append-only ledger semantics for normal production runs.
- Do not require every store backend to implement deletion immediately; keep protocols optional and
  capability-detected.

## Guidance for future agents

Promote this only when Gateway starts implementing first-class draft/private run cleanup. Keep the
boundary clean: Runtime defines reusable storage mechanics; Gateway decides policy; Flow decides UX.
