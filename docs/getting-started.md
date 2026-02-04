# Getting started

This guide gets you from install to a durable **pause → resume** workflow quickly.

If you only read one doc after `README.md`, read this one.

## Install

Core runtime:

```bash
pip install abstractruntime
```

Optional (LLM + tools via AbstractCore):

```bash
pip install "abstractruntime[abstractcore]"
```

## Mental model (source of truth)

- Workflows are in-memory graphs: `WorkflowSpec` (`src/abstractruntime/core/spec.py`)
- Runs are durable checkpoints: `RunState` (`src/abstractruntime/core/models.py`)
- Nodes return “what to do next”: `StepPlan` (`src/abstractruntime/core/models.py`)
- Side effects are requested, not executed directly: `Effect` / `EffectType` (`src/abstractruntime/core/models.py`)
- Blocking is explicit and durable: `WaitState` (`src/abstractruntime/core/models.py`)
- Every step is append-only in the ledger: `StepRecord` (`src/abstractruntime/core/models.py`)

The execution loop is implemented in `Runtime.start/tick/resume` (`src/abstractruntime/core/runtime.py`).

## Quick start: pause + resume

```python
from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.storage import InMemoryLedgerStore, InMemoryRunStore


def ask(run, ctx):
    return StepPlan(
        node_id="ask",
        effect=Effect(type=EffectType.ASK_USER, payload={"prompt": "Continue?"}, result_key="answer"),
        next_node="done",
    )


def done(run, ctx):
    return StepPlan(node_id="done", complete_output={"answer": run.vars.get("answer")})


wf = WorkflowSpec(workflow_id="demo", entry_node="ask", nodes={"ask": ask, "done": done})
rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

run_id = rt.start(workflow=wf)
state = rt.tick(workflow=wf, run_id=run_id)
assert state.status.value == "waiting"

state = rt.resume(workflow=wf, run_id=run_id, wait_key=state.waiting.wait_key, payload={"text": "yes"})
assert state.status.value == "completed"
print(state.output)
```

## Recommended: use the built-in scheduler wrapper

For most apps, use `create_scheduled_runtime()` which bundles:
- `Runtime`
- `Scheduler` (in-process polling driver)
- `WorkflowRegistry` (maps `workflow_id` → `WorkflowSpec`)

Implementation: `src/abstractruntime/scheduler/*`.

```python
from datetime import datetime, timedelta, timezone
import time

from abstractruntime import create_scheduled_runtime, Effect, EffectType, StepPlan, WorkflowSpec, RunStatus

def wait(run, ctx):
    until = (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()
    return StepPlan(
        node_id="wait",
        effect=Effect(type=EffectType.WAIT_UNTIL, payload={"until": until}),
        next_node="done",
    )

def done(run, ctx):
    return StepPlan(node_id="done", complete_output={"ok": True})

wf = WorkflowSpec(workflow_id="demo_wait_until", entry_node="wait", nodes={"wait": wait, "done": done})

sr = create_scheduled_runtime(poll_interval_s=0.2)  # in-memory stores, scheduler auto-starts
run_id, state = sr.run(wf)
assert state.status == RunStatus.WAITING

time.sleep(3)
state = sr.get_state(run_id)
print(state.status.value, state.output)

sr.stop()
```

## Persist runs + ledgers (survive restarts)

Use file-backed stores:

```python
from abstractruntime import create_scheduled_runtime, JsonFileRunStore, JsonlLedgerStore

sr = create_scheduled_runtime(
    run_store=JsonFileRunStore("./data"),
    ledger_store=JsonlLedgerStore("./data"),
)
```

Notes:
- `JsonFileRunStore` stores `run_<run_id>.json`
- `JsonlLedgerStore` stores `ledger_<run_id>.jsonl`
- Durable state must be JSON-serializable; for large payloads use `ArtifactStore`/offloading (see `architecture.md`)

## Optional: LLM + tools (AbstractCore)

AbstractRuntime’s kernel is dependency-light; LLM/tool execution is wired via the AbstractCore integration:
- docs: `integrations/abstractcore.md`
- code: `src/abstractruntime/integrations/abstractcore/*`

Typical local mode:

```python
from abstractruntime.integrations.abstractcore import create_local_runtime

rt = create_local_runtime(provider="ollama", model="qwen3:4b")
```

## Next reading

- `faq.md` — common questions and gotchas
- `api.md` — public API surface (imports + pointers)
- `architecture.md` — component map + durability invariants (with diagrams)
- `manual_testing.md` — smoke tests and how to run `pytest`
- `../examples/README.md` — runnable scripts
- `integrations/abstractcore.md` — `LLM_CALL` / `TOOL_CALLS`
