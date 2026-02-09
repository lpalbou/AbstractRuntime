# AbstractRuntime Roadmap

## Current status (v0.4.2)

AbstractRuntime provides a durable workflow kernel plus optional integrations:
- durable execution: `Runtime.start/tick/resume`, explicit `WaitState` (`src/abstractruntime/core/runtime.py`)
- append-only ledger (`StepRecord`) + persistent stores (JSON/JSONL, SQLite) (`src/abstractruntime/storage/*`)
- built-in scheduler (`Scheduler`, `ScheduledRuntime`) (`src/abstractruntime/scheduler/*`)
- snapshots/bookmarks (`src/abstractruntime/storage/snapshots.py`)
- tamper-evident hash-chained ledger (`src/abstractruntime/storage/ledger_chain.py`)
- artifacts + offloading for large payloads (`src/abstractruntime/storage/artifacts.py`, `src/abstractruntime/storage/offloading.py`)
- retries/idempotency hooks (`src/abstractruntime/core/policy.py`)
- VisualFlow compiler + WorkflowBundles (`src/abstractruntime/visualflow_compiler/*`, `src/abstractruntime/workflow_bundle/*`)
- AbstractCore integration for `LLM_CALL` / `TOOL_CALLS` (`docs/integrations/abstractcore.md`)

## Near-term priorities

These are tracked in `docs/backlog/planned/`:

1. **Signatures and keys** — non-forgeable provenance (beyond tamper-evidence)  
   `docs/backlog/planned/008_signatures_and_keys.md`

2. **Remote tool worker executor** — first-class worker boundary for tool execution  
   `docs/backlog/planned/014_remote_tool_worker_executor.md`

3. **Limit warnings + observability events** — surface `_limits` warnings durably/streaming  
   `docs/backlog/planned/017_limit_warnings_and_observability.md`

4. **Agent integration improvements** — reduce friction for external agent loops building on runtime  
   `docs/backlog/planned/015_agent_integration_improvements.md`

## Longer-term (not scheduled)

- distributed scheduling primitives (beyond in-process polling)
- workflow versioning/migration patterns for long-lived runs and snapshot restore
- stronger reproducibility contracts for replays (workflow snapshotting + run history bundles)
