## 025_runtime_retention_and_purge_contract (planned)

**Status**: Partially implemented
**Priority**: Medium
**Depends on**: 009_artifact_store (completed), 011_subworkflow_support (completed), 012_run_store_query_and_scheduler_support (completed), 024_runtime_owned_run_scoped_media_execution (completed)
**Related ADRs**: 0003_provenance_tamper_evident_hash_chain, 0004_runtime_owns_run_scoped_media_execution_truth

---

## Goal

Define capability-detected runtime purge mechanics for explicitly ephemeral run trees without weakening the default durable, append-only posture of AbstractRuntime.

---

## Current code reality

- `RunStore` still exposes only `save()` / `load()` as its mandatory base interface.
- `QueryableRunStore` already exposes `list_runs()`, `list_due_wait_until()`, and `list_children()`.
- `QueryableRunIndexStore` already exposes `list_run_index(...)`, including `session_id` and `root_only`, so hosts can already hide child runs from top-level listings without inventing purge semantics first.
- `LedgerStore` is still append-only as its mandatory base interface with `append()` / `list()`.
- Optional deletion protocols now exist for host-owned cleanup paths:
  - `DeletableRunStore.delete(run_id)`
  - `DeletableLedgerStore.delete(run_id)`
  - `DeletableCommandStore.delete_by_run(run_id)`
- In-memory, JSON-file/JSONL, SQLite, and storage wrapper implementations now support those deletion protocols where applicable.
- `ArtifactStore` already supports `delete()` and `delete_by_run()`.
- Runtime already has private tree-walk precedents via `history_bundle._list_descendant_run_ids(...)` and scheduler child-run traversal, but no public host-facing purge helper.
- Gateway now uses these optional deletion capabilities in `abstractgateway.run_retention` to purge expired ephemeral draft-test run trees. Runtime still does not own the Gateway/Flow draft taxonomy.
- Runtime-owned child runs now include both classic subworkflows and media child runs, so cleanup is no longer only a Flow/Gateway concern.
- Provenance hash chains are implemented, which makes purge/archive semantics an explicit audit decision, not just a storage API detail.

---

## Problem

Hosts need consistent cleanup mechanics for draft/private/ephemeral runs. If each host invents its own run-tree deletion logic, child runs, artifacts, and ledger retention behavior will drift.

At the same time, Runtime should not silently become a policy engine that decides which runs are purgeable. The durable boundary should stay:

- Runtime defines reusable mechanics and capability detection.
- Hosts decide policy defaults and when purge is allowed.

---

## Planned scope

### 1. Minimal durable retention metadata

Define a small runtime-owned metadata contract instead of a broad taxonomy:

- `run.vars["_runtime"]["retention_class"]` with values like `durable` or `ephemeral`
- optional `run.vars["_runtime"]["expires_at"]`

Do not introduce broader fields like `purpose` or `visibility` unless a real host consumer proves they are needed.

### 2. Optional storage capabilities

Add explicit optional protocols rather than widening every base interface:

- `DeletableRunStore.delete(run_id)`
- `DeletableLedgerStore.delete_by_run(run_id)` or equivalent
- optional artifact-GC capability for stores that dedupe blobs and need a separate reclamation pass

Deletion support must remain capability-detected. Not every backend needs to implement it immediately.

### 3. Host-facing purge helper

Add a runtime helper or utility for:

- collecting a root run plus descendants
- refusing `RUNNING` / `WAITING` runs by default
- refusing `durable` runs unless forced
- deleting run checkpoints, run-owned ledgers, and run-owned artifacts when supported
- reporting unsupported capabilities explicitly
- returning a structured dry-run / execute report

This should be a host-facing utility, not a new always-on `Runtime.tick(...)` behavior.

### 4. Explicit v1 boundary

Version 1 should cover:

- root run
- descendant runs
- run checkpoints
- run-owned ledgers
- run-owned artifacts

Version 1 should explicitly exclude:

- session attachment pseudo-runs / session-memory artifacts
- archive-to-artifact behavior
- automatic expiry schedulers

### 5. Provenance and documentation

Document the consequences clearly:

- durability remains the default
- append-only ledgers remain the normal production posture
- purge is for explicitly ephemeral runs or explicit operator force paths
- artifact deletion may not reclaim deduped blob bytes until store-specific GC runs

---

## Acceptance criteria

- [ ] Runtime documents a minimal retention metadata contract centered on `retention_class` and optional `expires_at`.
- [x] Optional delete capabilities are defined without widening every storage base class.
- [ ] A host-facing purge helper can dry-run and execute run-tree cleanup when the configured stores support it.
- [ ] The helper refuses active runs by default and refuses durable runs unless explicitly forced.
- [ ] The helper reports descendant runs, deleted checkpoints, deleted ledgers, deleted artifacts, unsupported capabilities, and GC caveats in a structured result.
- [ ] Session attachment cleanup is either explicitly excluded from v1 or tracked as a separate follow-up.
- [ ] Provenance docs explain the interaction between purge and hash-chained ledgers.

---

## Validation

1. Contract tests for in-memory, JSON-file, and SQLite stores covering supported and unsupported delete capabilities.
2. Run-tree tests covering root + descendant cleanup, including media child runs and subworkflow child runs.
3. Refusal tests for `RUNNING`, `WAITING`, and `durable` runs.
4. Artifact tests covering per-run deletion vs deduped blob GC behavior.
5. Documentation review to ensure hosts understand the policy/mechanics boundary.

---

## ADR note

No new ADR is required yet. If this item graduates into an accepted retention taxonomy or a durable archive policy, capture that boundary in an ADR at implementation time.

---

## Non-goals

- Runtime-owned retention policy defaults for Gateway, Flow, or other hosts
- mandatory delete support for every storage backend
- changing normal production ledgers from append-only to purge-first
- automatic cleanup of session attachment registries in v1
- background expiry daemons or scheduler behavior in this first item

---

## Priority note

This matters more now that Runtime owns more child-run patterns, but it is still a medium-priority mechanics item, not the highest-priority runtime correctness gap.

---

## 2026-05-25 implementation note

The AbstractFlow draft/published lifecycle slice added the storage-level deletion mechanics needed
by Gateway-owned draft-test purge:

- `abstractruntime.storage.base.DeletableRunStore`
- `abstractruntime.storage.base.DeletableLedgerStore`
- `abstractruntime.storage.commands.DeletableCommandStore`
- concrete deletion support for memory, JSON/JSONL, SQLite, offloading, observable, and hash-chain
  store wrappers
- focused coverage in `tests/test_storage_deletion.py`

The broader Runtime-owned host-facing purge helper remains open. Gateway currently owns the only
concrete purge policy because `purpose: draft_test` and `retention.mode: ephemeral` are Gateway/Flow
control-plane semantics, not Runtime taxonomy.
