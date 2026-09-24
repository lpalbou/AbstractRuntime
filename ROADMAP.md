# AbstractRuntime Roadmap

## Current status

What changed in each release is in [CHANGELOG.md](CHANGELOG.md); planned work is tracked in the [AbstractFramework backlog](https://github.com/lpalbou/abstractframework/tree/main/docs/backlog).

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

## Longer-term (not scheduled)

- distributed scheduling primitives (beyond in-process polling)
- workflow versioning/migration patterns for long-lived runs and snapshot restore
- stronger reproducibility contracts for replays (workflow snapshotting + run history bundles)
