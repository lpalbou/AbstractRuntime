# API reference

This document summarizes the **public Python API** of AbstractRuntime and points to the **source of truth in code**.

Public exports live in `src/abstractruntime/__init__.py`. If you are unsure what is supported for external use, start there.

## Recommended imports

Core kernel:

```python
from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
```

Storage helpers (common stores):

```python
from abstractruntime.storage import (
    InMemoryLedgerStore,
    InMemoryRunStore,
    JsonFileRunStore,
    JsonlLedgerStore,
)
```

Scheduler convenience wrapper:

```python
from abstractruntime import create_scheduled_runtime
```

Optional integration (requires `abstractruntime[abstractcore]`):

```python
from abstractruntime.integrations.abstractcore import create_local_runtime
```

See also: `getting-started.md` (end-to-end runnable examples).

## Core types (durable workflow semantics)

Implementation: `src/abstractruntime/core/models.py`, `src/abstractruntime/core/spec.py`.

- `WorkflowSpec`: in-memory workflow graph (`workflow_id`, `entry_node`, `nodes`).
- `StepPlan`: node return value (what happens next): `effect`, `next_node`, or `complete_output`.
- `Effect` / `EffectType`: durable side-effect request protocol (the runtime mediates execution).
- `RunState` / `RunStatus`: durable checkpoint for a run, persisted by a `RunStore`.
- `WaitState` / `WaitReason`: durable pause metadata for `WAIT_*` / `ASK_USER` / passthrough tool waits.

Durability invariant: `RunState.vars` must remain JSON-serializable (`src/abstractruntime/core/models.py`). For large payloads use artifacts/offloading (`src/abstractruntime/storage/artifacts.py`, `src/abstractruntime/storage/offloading.py`).

## Runtime (start / tick / resume)

Implementation: `src/abstractruntime/core/runtime.py`.

- `Runtime.start(workflow, vars=..., actor_id=..., session_id=...) -> run_id`
  - creates and persists a new `RunState`
- `Runtime.tick(workflow, run_id, max_steps=...) -> RunState`
  - executes node handlers and effects until the run becomes `WAITING`, `COMPLETED`, `FAILED`, or `CANCELLED`
- `Runtime.resume(workflow, run_id, wait_key, payload, override_node=...) -> RunState`
  - validates the `wait_key`, writes `payload` to `WaitState.result_key` (if set), and continues from `WaitState.resume_to_node`

For the execution model (ledger records, effect outcomes, waits), see `architecture.md`.

## Scheduler convenience API

Implementation: `src/abstractruntime/scheduler/*`.

Use `create_scheduled_runtime()` for a zero-config wrapper that bundles `Runtime` + an in-process polling `Scheduler`:
- `ScheduledRuntime.run(workflow, vars=..., actor_id=..., max_steps=...) -> (run_id, state)` (`src/abstractruntime/scheduler/convenience.py`)
- `ScheduledRuntime.respond(run_id, payload) -> RunState` (resumes a waiting run using its stored `wait_key`)
- `ScheduledRuntime.stop()` (stops the scheduler thread/loop)

For time-based waits, the scheduler polls due runs via `QueryableRunStore.list_due_wait_until(...)` (`src/abstractruntime/storage/base.py`, `src/abstractruntime/scheduler/scheduler.py`).

## Storage layer (durability backends)

Interfaces: `RunStore`, `LedgerStore`, and `QueryableRunStore` are defined in `src/abstractruntime/storage/base.py`.

Included backends:
- In-memory (tests/dev): `InMemoryRunStore`, `InMemoryLedgerStore` (`src/abstractruntime/storage/in_memory.py`)
- Filesystem:
  - checkpoints: `JsonFileRunStore` (`src/abstractruntime/storage/json_files.py`)
  - append-only ledger: `JsonlLedgerStore` (`src/abstractruntime/storage/json_files.py`)
- SQLite:
  - `SqliteRunStore`, `SqliteLedgerStore` (`src/abstractruntime/storage/sqlite.py`)

Common decorators:
- `ObservableLedgerStore` for subscriptions (`src/abstractruntime/storage/observable.py`)
- `HashChainedLedgerStore` + `verify_ledger_chain(...)` for tamper-evidence (`src/abstractruntime/storage/ledger_chain.py`)
- `OffloadingRunStore` / `OffloadingLedgerStore` to store large values by artifact reference (`src/abstractruntime/storage/offloading.py`)

## Artifacts (store by reference)

Implementation: `src/abstractruntime/storage/artifacts.py`.

Key types:
- `ArtifactStore` (interface), `InMemoryArtifactStore`, `FileArtifactStore`
- helpers: `artifact_ref(...)`, `resolve_artifact(...)`, `is_artifact_ref(...)`

Artifacts are used by:
- offloading wrappers (`src/abstractruntime/storage/offloading.py`)
- evidence capture (`docs/evidence.md`, `src/abstractruntime/evidence/recorder.py`)

## Snapshots / bookmarks

Implementation: `src/abstractruntime/storage/snapshots.py`.

- `SnapshotStore` interface + `InMemorySnapshotStore`, `JsonSnapshotStore`
- `Snapshot` model (a named bookmark of run state)

Docs: `snapshots.md`.

## Effect policies (retries + idempotency)

Implementation: `src/abstractruntime/core/policy.py`.

- `EffectPolicy` protocol and implementations: `DefaultEffectPolicy`, `RetryPolicy`, `NoRetryPolicy`
- `compute_idempotency_key(...)` helper

Docs: `architecture.md` (reliability section).

## WorkflowBundles (`.flow`) and VisualFlow distribution

Implementation:
- bundles: `src/abstractruntime/workflow_bundle/*`
- compiler: `src/abstractruntime/visualflow_compiler/*`

Public bundle APIs are exported from `src/abstractruntime/workflow_bundle/__init__.py` and re-exported in `src/abstractruntime/__init__.py`:
- open: `open_workflow_bundle(...)`
- registry: `WorkflowBundleRegistry`
- pack/unpack: `pack_workflow_bundle(...)`, `unpack_workflow_bundle(...)`

Docs: `workflow-bundles.md`.

## Run history bundle export (portable replay artifact)

Implementation: `src/abstractruntime/history_bundle.py`.

- `export_run_history_bundle(...)`
- `persist_workflow_snapshot(...)`

This produces a portable record of a run’s state + ledger + artifacts suitable for debugging/review.

## Optional integrations

### AbstractCore (LLM + tools)

Requires: `pip install "abstractruntime[abstractcore]"`.

Implementation: `src/abstractruntime/integrations/abstractcore/*`.

Entry points:
- `create_local_runtime(...)`, `create_remote_runtime(...)`, `create_hybrid_runtime(...)` (`src/abstractruntime/integrations/abstractcore/factory.py`)
- effect handler wiring: `build_effect_handlers(...)` (`src/abstractruntime/integrations/abstractcore/effect_handlers.py`)

Docs: `integrations/abstractcore.md`.

### AbstractMemory bridge (KG effects)

Implementation: `src/abstractruntime/integrations/abstractmemory/effect_handlers.py`.

This provides handlers for `MEMORY_KG_*` effects (opt-in wiring layer).

## See also

- `../README.md` — install + quick start
- `getting-started.md` — first durable workflow
- `architecture.md` — component map + durability invariants
- `faq.md` — common questions and gotchas
- `integrations/abstractcore.md` — `LLM_CALL` / `TOOL_CALLS` wiring

