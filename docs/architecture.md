# AbstractRuntime — Architecture (Current)

> Updated: 2026-01-07  
> Scope: this describes **what is implemented today** in this monorepo.

AbstractRuntime is the **durable execution kernel** of the AbstractFramework. It runs workflow graphs as a persisted state machine:
- **tick** a run forward until it completes or blocks
- persist **checkpoints** (RunState) + an append-only **ledger** (StepRecord)
- represent blocking as durable **WaitState** (USER / EVENT / UNTIL / SUBWORKFLOW)
- resume with an external payload (human input, event envelope, job completion)

AbstractRuntime is intentionally dependency-light in the core (`abstractruntime/core/*`):
- higher-level integrations (LLM providers, tool execution, event bus bridging) live in `abstractruntime/integrations/*`

## High-level runtime loop (data flow)

```
WorkflowSpec (in-memory handlers)
   │
   ▼
Runtime.tick(run_id)
   │ loads + updates
   ▼
RunStore  (checkpoint RunState.vars + waiting)
   │ append
   ▼
LedgerStore (StepRecords: started/completed/waiting/failed)
   │ large payloads
   ▼
ArtifactStore (JSON blobs referenced from vars/ledger)
```

## Core Types (Durable, JSON-Safe)

Defined in `abstractruntime/src/abstractruntime/core/models.py`:

- `WorkflowSpec`
  - `workflow_id`, `entry_node`, `nodes: Dict[node_id, handler]`
- `RunState`
  - `run_id`, `workflow_id`, `status`, `current_node`
  - `vars` (**JSON-safe** dictionary, persisted by RunStore)
  - `waiting: WaitState | None`
  - `output` / `error`
  - optional provenance: `actor_id`, `session_id`, `parent_run_id`
- `StepPlan`
  - returned by node handlers: `{effect?, next_node?, complete_output?}`
- `Effect` + `EffectType`
  - requests for side effects / waits (LLM_CALL, TOOL_CALLS, ASK_USER, WAIT_EVENT, …)
- `WaitState` + `WaitReason`
  - durable blocking representation
- `StepRecord`
  - append-only audit log entries (STARTED/COMPLETED/WAITING/FAILED)

## Runtime Semantics (start / tick / resume)

Implemented in `abstractruntime/src/abstractruntime/core/runtime.py`:

- `Runtime.start(workflow, vars, actor_id?, session_id?, parent_run_id?) -> run_id`
  - creates a `RunState` checkpoint (RUNNING, current_node=entry)
- `Runtime.tick(workflow, run_id, max_steps=...) -> RunState`
  - executes node handlers in a loop:
    - handler returns `StepPlan`
    - if `StepPlan.effect` exists → dispatch to the effect handler registry
    - if effect outcome blocks → persist `RunState.waiting` and return WAITING
    - if `next_node` exists → advance cursor and continue
    - if `complete_output` exists → set run output and finish
- `Runtime.resume(workflow, run_id, wait_key, payload, max_steps=...) -> RunState`
  - validates the run is waiting for `wait_key`
  - writes `payload` to the waiting node’s `result_key`
  - clears `waiting`, transitions to RUNNING, continues via `tick`

### Run Control (pause/resume/cancel)
Run control is runtime-owned (also in `abstractruntime/core/runtime.py`):
- pause uses a synthetic WAIT_USER (`wait_key = pause:<run_id>`) and a durable flag in `vars["_runtime"]["control"]["paused"]`
- resume clears the pause wait and continues ticking
- cancel marks the run CANCELLED (and can cancel subflows via host orchestration)

## Persistence Layer (RunStore / LedgerStore / ArtifactStore)

Storage interfaces are defined in `abstractruntime/src/abstractruntime/storage/base.py` and implemented in:
- `abstractruntime/src/abstractruntime/storage/in_memory.py` (in-memory stores)
- `abstractruntime/src/abstractruntime/storage/json_files.py` (file-based stores)
  - `JsonFileRunStore`: `run_<run_id>.json`
  - `JsonlLedgerStore`: `ledger_<run_id>.jsonl`
- `abstractruntime/src/abstractruntime/storage/artifacts.py`
  - `ArtifactStore` + `FileArtifactStore` + helpers (`artifact_ref`, `resolve_artifact`, …)
- `abstractruntime/src/abstractruntime/storage/observable.py`
  - `ObservableLedgerStore` adds subscriptions for live UI streaming

**Key invariant:** `RunState.vars` must remain JSON-safe. Large payloads should be stored in `ArtifactStore` and referenced.

## Effect System (Handlers)

The runtime executes side effects by dispatching `EffectType` to handler callables. The core runtime ships with the effect protocol; hosts wire concrete handlers.

### Memory effects (runtime-owned, durable)
The runtime ships a small, provenance-first memory surface (no embeddings):
- `EffectType.MEMORY_COMPACT`: archive older `context.messages` into an artifact + insert a summary handle
- `EffectType.MEMORY_NOTE`: store a durable note with tags + sources
  - supports `payload.scope = run|session|global` (scope is routing: it selects the index-owner run)
- `EffectType.MEMORY_QUERY`: recall spans/notes by id/tags/time/query
  - supports `payload.scope = run|session|global|all`
  - supports `payload.return = rendered|meta|both` (meta enables deterministic workflows without parsing text)
- `EffectType.MEMORY_TAG`: apply/merge tags onto existing span index entries
- `EffectType.MEMORY_REHYDRATE`: rehydrate archived `conversation_span` artifacts into `context.messages` deterministically (deduped)

Global scope uses a stable run id (default `global_memory`, override via `ABSTRACTRUNTIME_GLOBAL_MEMORY_RUN_ID`, validated as filesystem-safe).

### AbstractCore Integration (LLM + Tools)
`abstractruntime/src/abstractruntime/integrations/abstractcore/*` provides:
- an AbstractCore-backed LLM client (`llm_client.py`)
- tool execution boundary (`tool_executor.py`)
  - `MappingToolExecutor` (recommended)
  - `AbstractCoreToolExecutor` (legacy adapter path)
  - `PassthroughToolExecutor` (host executes tools externally)
- effect handler wiring (`effect_handlers.py`)
- convenience factories (`factory.py`)
  - `create_local_runtime(...)` (local execution)
  - `create_remote_runtime(...)` (passthrough tools / host-mediated)

This keeps the kernel independent of AbstractCore while enabling LLM/tool workflows when a host opts in.

## Observability (Ledger + Runtime-Owned Node Traces)

### Ledger (Source of Truth)
Every node transition produces `StepRecord` entries appended to the `LedgerStore`.
When using `ObservableLedgerStore`, hosts can subscribe to appends for real-time UI updates.

### Runtime-Owned Node Traces
In addition to the ledger, the runtime stores bounded, JSON-safe per-node traces under:
`run.vars["_runtime"]["node_traces"]` (see `abstractruntime/src/abstractruntime/core/runtime.py:_record_node_trace` and helpers in `abstractruntime/src/abstractruntime/core/vars.py`).

These traces support host UX needs (scratchpad/trace views) without inventing host-specific persistence formats.

### Optional Event Bus Bridge
`abstractruntime.integrations.abstractcore.observability` can bridge ledger events to `abstractcore.events.GlobalEventBus` when desired.

## Scheduling (Time-Based Waits)

Time-based waits (`WAIT_UNTIL`) are durable. To advance them, a host needs to tick due runs. AbstractRuntime provides:
- `create_scheduled_runtime(...)` and a `Scheduler` (`abstractruntime/src/abstractruntime/scheduler/*`)

Hosts can run a scheduler loop to periodically:
- query due `WAIT_UNTIL` runs (requires a queryable run store)
- tick them forward

## Eventing (WAIT_EVENT / EMIT_EVENT)

Custom eventing is implemented as:
- listeners block on `WAIT_EVENT` (`WaitReason.EVENT`)
- emitters request `EffectType.EMIT_EVENT`, handled by the runtime (`Runtime._handle_emit_event`) which resumes matching `WAIT_EVENT` runs via `Runtime.resume(...)`
  - requires a `QueryableRunStore` to find waiting runs and a `workflow_registry` to load target `WorkflowSpec`s

Notes:
- The emitted event **payload is normalized to a dict** for network-safe stability. If a non-dict is provided, it is wrapped as `{ "value": <payload> }`.
- `WAIT_EVENT` waits can optionally include UX metadata (`prompt`, `choices`, `allow_free_text`) so hosts can render interactive prompts while remaining fully durable.

For **host-driven external signals**, use the scheduler API:
- `Scheduler.emit_event(...)` (finds waiting runs and resumes them)

Event envelopes (scope/session/workflow) are encoded via stable wait keys (see `abstractruntime/src/abstractruntime/core/event_keys.py`).

## Deviations / near-term work
- **Client-agnostic run history contract**: today, some hosts implement bespoke “ledger → UI events” replay logic. A runtime-owned, versioned `RunHistoryBundle` contract (planned: backlog 311) should become the shared format so any client can render consistent history and achieve stronger reproducibility.

