# FAQ

## What is AbstractRuntime (in one sentence)?

AbstractRuntime is a **durable workflow runtime**: it runs workflow graphs as a persisted state machine with explicit waits (pause → resume) and an append-only execution ledger.  
Code: `src/abstractruntime/core/runtime.py`, `src/abstractruntime/core/models.py`.

## Is AbstractRuntime an agent framework?

No. AbstractRuntime is the **execution substrate**. Agent logic (ReAct/CodeAct loops, prompt policies, etc.) is built *on top* of it.  
Docs: `proposal.md`. Code: `src/abstractruntime/core/*`.

## How does AbstractRuntime relate to AbstractCore / AbstractFramework?

AbstractRuntime is the **durable execution kernel**. In the AbstractFramework ecosystem, it is commonly paired with:
- **AbstractCore** for LLM + tool execution (`EffectType.LLM_CALL`, `EffectType.TOOL_CALLS`)  
  Code: `src/abstractruntime/integrations/abstractcore/*`. Repo: [lpalbou/abstractcore](https://github.com/lpalbou/abstractcore)

AbstractFramework umbrella: [lpalbou/AbstractFramework](https://github.com/lpalbou/AbstractFramework)

## Where is the public API documented?

- API guide: `api.md`
- Canonical export list: `src/abstractruntime/__init__.py`

## How do pause/resume work?

- A node returns a `StepPlan` with an `Effect` (e.g. `ASK_USER`, `WAIT_UNTIL`, `WAIT_EVENT`).
- The runtime persists a `WaitState` into `RunState.waiting` and returns `status=waiting`.
- You resume by calling `Runtime.resume(...)` (or `ScheduledRuntime.respond(...)`) with the matching `wait_key`.

Docs: `getting-started.md`, `architecture.md`. Code: `src/abstractruntime/core/runtime.py` (`tick`, `resume`) and `src/abstractruntime/core/models.py` (`WaitState`).

## Does time-based waiting (`WAIT_UNTIL`) progress automatically?

Only if **something drives the runtime**:
- `Runtime.tick(...)` will auto-unblock a due `WAIT_UNTIL` run *when called*.
- The built-in `Scheduler` provides a driver loop that polls due waits and ticks runs.

Docs: `getting-started.md`, `architecture.md`. Code: `src/abstractruntime/core/runtime.py` (`tick`), `src/abstractruntime/scheduler/scheduler.py`.

## How do I resume a waiting run?

- If you have the `WorkflowSpec`: call `Runtime.resume(workflow=..., run_id=..., wait_key=..., payload=...)`.
- If you use `create_scheduled_runtime()`: call `sr.respond(run_id, payload)` (it uses `state.waiting.wait_key`).

Docs: `getting-started.md`. Code: `src/abstractruntime/core/runtime.py`, `src/abstractruntime/scheduler/convenience.py`.

## Why is my `ASK_USER` answer a dict?

`Runtime.resume(..., payload=...)` always takes a **dict** payload. If the wait has a `result_key`, the runtime stores that dict into `RunState.vars` at `result_key`.  
Code: `src/abstractruntime/core/runtime.py` (`Runtime.resume`) and `src/abstractruntime/core/models.py` (`WaitState.result_key`).

Common pattern:
- resume with `{"text": "..."}` (host-side)
- read `run.vars["my_result_key"]["text"]` (node-side)

## What storage backends are included?

AbstractRuntime includes:
- in-memory: `InMemoryRunStore`, `InMemoryLedgerStore`
- filesystem: `JsonFileRunStore` (checkpoints), `JsonlLedgerStore` (append-only JSONL ledger)
- SQLite: `SqliteRunStore`, `SqliteLedgerStore`

Docs: `architecture.md`. Code: `src/abstractruntime/storage/*`.

## What must be JSON-serializable (and why)?

Everything stored in `RunState.vars` must be JSON-serializable because it is persisted as durable state.  
Code: `src/abstractruntime/core/models.py` (`RunState`) and store implementations under `src/abstractruntime/storage/`.

For large values, use:
- `ArtifactStore` references (`src/abstractruntime/storage/artifacts.py`)
- offloading wrappers (`OffloadingRunStore`, `OffloadingLedgerStore`) (`src/abstractruntime/storage/offloading.py`)

Docs: `architecture.md`.

## How do I run LLM calls and tools?

LLM and tool execution are wired via the **AbstractCore integration**:
- `EffectType.LLM_CALL`
- `EffectType.TOOL_CALLS`

Docs: `integrations/abstractcore.md`. Code: `src/abstractruntime/integrations/abstractcore/*`.

## What are “local / remote / hybrid” execution modes?

They refer to where LLM and tools execute:
- **Local**: in-process LLM + local tool execution
- **Remote**: HTTP to an AbstractCore server + tools typically passthrough
- **Hybrid**: remote LLM + local tools

Docs: `integrations/abstractcore.md`, `../docs/adr/0002_execution_modes_local_remote_hybrid.md`. Code: `src/abstractruntime/integrations/abstractcore/factory.py`.

## What does passthrough tool mode mean?

In passthrough mode, tool calls are **not executed** in-process:
- the `TOOL_CALLS` handler returns `WAITING` with tool call details
- an external worker/operator executes the tools
- the host resumes the run with the tool results

Docs: `integrations/abstractcore.md`. Code: `src/abstractruntime/integrations/abstractcore/tool_executor.py` (`PassthroughToolExecutor`).

## Does AbstractRuntime retry effects (LLM/tools)? Is it idempotent?

Retry and idempotency are controlled via `EffectPolicy`:
- idempotency keys are used to reuse prior completed results after restarts
- retry behavior is configurable (e.g. `RetryPolicy`)

Docs: `architecture.md`. Code: `src/abstractruntime/core/policy.py`, `src/abstractruntime/core/runtime.py` (effect execution + reuse).

## Is the ledger tamper-proof?

No. The built-in provenance feature is **tamper-evident** (hash chain), not signature-backed non-forgeability.

Docs: `provenance.md`. Code: `src/abstractruntime/storage/ledger_chain.py`.

## How do I stream progress updates?

If your `LedgerStore` supports subscriptions (or is wrapped with `ObservableLedgerStore`), you can subscribe in-process:
- `Runtime.subscribe_ledger(callback, run_id=...)`

Docs: `architecture.md`. Code: `src/abstractruntime/core/runtime.py` (`subscribe_ledger`), `src/abstractruntime/storage/observable.py`.

## What is “evidence capture”?

Evidence capture records durable, artifact-backed evidence for selected external-boundary tools:
- `web_search`, `fetch_url`, `execute_command`

It runs best-effort after successful `TOOL_CALLS` and requires an `ArtifactStore`.  
Docs: `evidence.md`. Code: `src/abstractruntime/evidence/recorder.py`, `src/abstractruntime/core/runtime.py` (`_maybe_record_tool_evidence`, `list_evidence`, `load_evidence`).

## What are snapshots and are they safe to restore?

Snapshots are named bookmarks of run state. Restoring a snapshot is a host-level operation (load + write back into your RunStore).  
Safety depends on whether workflow code/spec has changed since the snapshot was taken.

Docs: `snapshots.md`. Code: `src/abstractruntime/storage/snapshots.py`.

## How do WorkflowBundles (`.flow`) relate to `WorkflowSpec`?

`WorkflowSpec` is an in-memory graph of Python callables (not portable). WorkflowBundles (`.flow`) distribute **VisualFlow JSON** plus a manifest; hosts compile VisualFlow JSON into `WorkflowSpec` using the VisualFlow compiler.

Docs: `workflow-bundles.md`, `architecture.md`. Code: `src/abstractruntime/workflow_bundle/*`, `src/abstractruntime/visualflow_compiler/*`.

## How do I run the MCP worker?

Use the `abstractruntime-mcp-worker` CLI (from the `mcp-worker` extra) and select toolsets explicitly.

Docs: `mcp-worker.md`. Code: `src/abstractruntime/integrations/abstractcore/mcp_worker.py`.

## Where should I look for runnable examples?

- `../examples/README.md` (runnable scripts)
- `manual_testing.md` (smoke tests)
