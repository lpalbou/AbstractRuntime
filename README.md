# AbstractRuntime

**AbstractRuntime** is a durable workflow runtime (interrupt → checkpoint → resume) with an append-only execution ledger.

It is designed for long-running workflows that must survive restarts and explicitly model blocking (human input, timers, external events, subworkflows) without keeping Python stacks alive.

**Version:** 0.4.0 (`pyproject.toml`) • **Python:** 3.10+

## Install

Core runtime:

```bash
pip install abstractruntime
```

AbstractCore integration (LLM + tools):

```bash
pip install "abstractruntime[abstractcore]"
```

MCP worker entrypoint (default toolsets over stdio):

```bash
pip install "abstractruntime[mcp-worker]"
```

## Quick start (pause + resume)

```python
from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.storage import InMemoryLedgerStore, InMemoryRunStore


def ask(run, ctx):
    return StepPlan(
        node_id="ask",
        effect=Effect(
            type=EffectType.ASK_USER,
            payload={"prompt": "Continue?"},
            result_key="user_answer",
        ),
        next_node="done",
    )


def done(run, ctx):
    return StepPlan(node_id="done", complete_output={"answer": run.vars.get("user_answer")})


wf = WorkflowSpec(workflow_id="demo", entry_node="ask", nodes={"ask": ask, "done": done})
rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

run_id = rt.start(workflow=wf)
state = rt.tick(workflow=wf, run_id=run_id)
assert state.status.value == "waiting"

state = rt.resume(
    workflow=wf,
    run_id=run_id,
    wait_key=state.waiting.wait_key,
    payload={"text": "yes"},
)
assert state.status.value == "completed"
```

## What’s included (v0.4.0)

Kernel (dependency-light):
- workflow graphs: `WorkflowSpec` (`src/abstractruntime/core/spec.py`)
- durable execution: `Runtime.start/tick/resume` (`src/abstractruntime/core/runtime.py`)
- durable waits/events: `WAIT_EVENT`, `WAIT_UNTIL`, `ASK_USER`, `EMIT_EVENT`
- append-only ledger (`StepRecord`) + node traces (`vars["_runtime"]["node_traces"]`)
- retries/idempotency hooks: `src/abstractruntime/core/policy.py`

Durability + storage:
- stores: in-memory, JSON/JSONL, SQLite (`src/abstractruntime/storage/*`)
- artifacts + offloading (store large payloads by reference)
- snapshots/bookmarks (`docs/snapshots.md`)
- tamper-evident hash-chained ledger (`docs/provenance.md`)

Drivers + distribution:
- scheduler: `create_scheduled_runtime()` (`src/abstractruntime/scheduler/*`)
- VisualFlow compiler + WorkflowBundles (`src/abstractruntime/visualflow_compiler/*`, `src/abstractruntime/workflow_bundle/*`)
- run history export: `export_run_history_bundle(...)` (`src/abstractruntime/history_bundle.py`)

Optional integrations:
- AbstractCore (LLM + tools): `docs/integrations/abstractcore.md`
- comms toolset gating (email/WhatsApp/Telegram): `docs/tools-comms.md`

## Built-in scheduler (zero-config)

```python
from abstractruntime import create_scheduled_runtime

sr = create_scheduled_runtime()
run_id, state = sr.run(my_workflow)

if state.status.value == "waiting":
    state = sr.respond(run_id, {"answer": "yes"})

sr.stop()
```

For persistent storage:

```python
from abstractruntime import create_scheduled_runtime, JsonFileRunStore, JsonlLedgerStore

sr = create_scheduled_runtime(
    run_store=JsonFileRunStore("./data"),
    ledger_store=JsonlLedgerStore("./data"),
)
```

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Install + first durable workflow |
| [Docs Index](docs/README.md) | Full docs map (guides + reference) |
| [Architecture](docs/architecture.md) | Component map + diagrams |
| [Overview](docs/proposal.md) | Design goals, core concepts, and scope |
| [Integrations](docs/integrations/) | Integration guides (AbstractCore) |
| [Snapshots](docs/snapshots.md) | Named checkpoints for run state |
| [Provenance](docs/provenance.md) | Tamper-evident ledger documentation |
| [Evidence](docs/evidence.md) | Artifact-backed evidence capture for web/command tools |
| [Limits](docs/limits.md) | `_limits` namespace and RuntimeConfig |
| [WorkflowBundles](docs/workflow-bundles.md) | `.flow` bundle format (VisualFlow distribution) |
| [MCP Worker](docs/mcp-worker.md) | `abstractruntime-mcp-worker` CLI |
| [ROADMAP](ROADMAP.md) | Prioritized next steps |

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[abstractcore,mcp-worker]"
python -m pytest -q
```
