# AbstractRuntime — Overview (v0.4.x)

**AbstractRuntime** is a low-level *durable workflow runtime*:
- execute workflow graphs (state machines)
- support **interrupt → checkpoint → resume** without keeping Python stacks alive
- record an append-only **execution journal** (“ledger”) for observability/audit/debug

**Scope boundary:** AbstractRuntime is not a UI builder and not an agent framework. It is the execution substrate that higher-level orchestration (e.g., visual authoring hosts) and agent loops can build on.

## What problem this solves

Once a workflow can:
- ask a user and wait hours/days
- wait until a scheduled time
- wait for an external job/event

…you need durable semantics: persisted checkpoints + a journal. Keeping a Python stack alive is not reliable across restarts.

## Core concepts (durable model)

All core durable types are stdlib-only and live in `src/abstractruntime/core/models.py`:

- `WorkflowSpec`: in-memory workflow graph (node handlers keyed by id) (`src/abstractruntime/core/spec.py`)
- `RunState`: durable run checkpoint (`run_id`, `status`, `current_node`, `vars`, `waiting`, `output`, `error`, provenance fields)
- `StepPlan`: what a node returns (`effect?`, `next_node?`, `complete_output?`)
- `Effect` + `EffectType`: request for side effects (LLM, tools, waits, memory ops, etc.)
- `WaitState`: durable blocking state (`reason`, `wait_key`/`until`, `resume_to_node`, `result_key`)
- `StepRecord`: append-only ledger entry

**Non-negotiable constraint:** values stored in `RunState.vars` must be JSON-serializable. For large payloads, use `ArtifactStore` references (`src/abstractruntime/storage/artifacts.py`) or offloading wrappers (`src/abstractruntime/storage/offloading.py`).

## Minimal runtime API

Implemented in `src/abstractruntime/core/runtime.py`:
- `start(workflow, vars, actor_id, session_id) -> run_id`
- `tick(workflow, run_id) -> RunState` (progress until waiting/completed/failed)
- `resume(workflow, run_id, wait_key, payload) -> RunState`
- `get_state(run_id) -> RunState`
- `get_ledger(run_id) -> list[dict]`

Resume semantics (important):
- when a run blocks, the runtime stores `WaitState.resume_to_node`
- on resume, execution continues **from that node** (it does not re-run the waiting node)

## Persistence

Interfaces live in `src/abstractruntime/storage/base.py`.

Included backends:
- `InMemory*` (tests/dev): `src/abstractruntime/storage/in_memory.py`
- file-based JSON/JSONL: `src/abstractruntime/storage/json_files.py`
- SQLite: `src/abstractruntime/storage/sqlite.py`

Related features:
- snapshots/bookmarks: `src/abstractruntime/storage/snapshots.py` (`docs/snapshots.md`)
- tamper-evident ledger: `src/abstractruntime/storage/ledger_chain.py` (`docs/provenance.md`)
- in-process ledger subscriptions: `src/abstractruntime/storage/observable.py`

## Scheduling (driver loop)

AbstractRuntime ships a simple in-process scheduler:
- `Scheduler`, `ScheduledRuntime`, `create_scheduled_runtime()` (`src/abstractruntime/scheduler/*`)

This is a driver loop (polls due waits, resumes runs). It is not a distributed orchestrator.

## Integrations (optional)

AbstractRuntime stays dependency-light at the kernel level; concrete integrations are opt-in:
- AbstractCore (LLM + tools): `src/abstractruntime/integrations/abstractcore/*` (`integrations/abstractcore.md`)
- AbstractMemory bridge (KG assertions/queries): `src/abstractruntime/integrations/abstractmemory/*`

## Status (implemented in this repository)

As of v0.4.1 (`pyproject.toml`):
- durable kernel: `RunState`, `WaitState`, `Runtime.start/tick/resume`
- built-in waits + events: `WAIT_EVENT`, `WAIT_UNTIL`, `ASK_USER`, `EMIT_EVENT`
- persistence backends: in-memory, JSON/JSONL, SQLite
- artifacts/offloading: store large payloads by reference
- retries/idempotency policy hooks: `src/abstractruntime/core/policy.py`
- snapshots, tamper-evident ledger chain, ledger subscriptions
- VisualFlow compiler + WorkflowBundles (`src/abstractruntime/visualflow_compiler/*`, `src/abstractruntime/workflow_bundle/*`)
- evidence capture helpers (`src/abstractruntime/evidence/recorder.py`, `Runtime.list_evidence/load_evidence`)
- run history bundle export (`src/abstractruntime/history_bundle.py`)

## See also

- `../README.md` — install + quick start
- `getting-started.md` — first steps
- `architecture.md` — full component map and diagrams
- `integrations/abstractcore.md` — AbstractCore wiring
- `limits.md` — `_limits` / RuntimeConfig
