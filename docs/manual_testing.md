# Manual testing

This guide is a small set of **manual smoke tests** you can run to verify the durable runtime loop (start/tick/wait/resume), scheduler resumption, and persistence.

## Prerequisites

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

## Test 1: Zero-config hello world

```python
from abstractruntime import create_scheduled_runtime, StepPlan, WorkflowSpec


def greet(run, ctx):
    name = run.vars.get("name", "World")
    return StepPlan(node_id="greet", complete_output={"message": f"Hello, {name}!"})


workflow = WorkflowSpec(
    workflow_id="hello",
    entry_node="greet",
    nodes={"greet": greet},
)

sr = create_scheduled_runtime()
run_id, state = sr.run(workflow, vars={"name": "Alice"})

print(state.status.value)
print(state.output)

sr.stop()
```

Expected:
- status is `completed`
- output contains `{"message": "Hello, Alice!"}`

---

## Test 2: Ask user (pause + resume)

```python
from abstractruntime import create_scheduled_runtime, Effect, EffectType, StepPlan, WorkflowSpec, RunStatus


def ask_name(run, ctx):
    return StepPlan(
        node_id="ask",
        effect=Effect(
            type=EffectType.ASK_USER,
            payload={"prompt": "What is your name?"},
            result_key="user_input",
        ),
        next_node="greet",
    )


def greet(run, ctx):
    name = run.vars.get("user_input", {}).get("text", "Unknown")
    return StepPlan(node_id="greet", complete_output={"greeting": f"Hello, {name}!"})


workflow = WorkflowSpec(
    workflow_id="ask_and_greet",
    entry_node="ask",
    nodes={"ask": ask_name, "greet": greet},
)

sr = create_scheduled_runtime()
run_id, state = sr.run(workflow)
assert state.status == RunStatus.WAITING
print(state.waiting.prompt)

state = sr.respond(run_id, {"text": "Bob"})
assert state.status == RunStatus.COMPLETED
print(state.output)

sr.stop()
```

Expected:
- first run blocks with `status=waiting` and a `prompt`
- after `respond`, run completes with a greeting

---

## Test 3: Wait until (scheduler auto-resume)

```python
from datetime import datetime, timedelta, timezone
import time

from abstractruntime import create_scheduled_runtime, Effect, EffectType, StepPlan, WorkflowSpec, RunStatus


def schedule_task(run, ctx):
    until = (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()
    return StepPlan(
        node_id="schedule",
        effect=Effect(type=EffectType.WAIT_UNTIL, payload={"until": until}),
        next_node="execute",
    )


def execute_task(run, ctx):
    return StepPlan(node_id="execute", complete_output={"ok": True})


workflow = WorkflowSpec(
    workflow_id="scheduled_task",
    entry_node="schedule",
    nodes={"schedule": schedule_task, "execute": execute_task},
)

sr = create_scheduled_runtime(poll_interval_s=0.2)
run_id, state = sr.run(workflow)
assert state.status == RunStatus.WAITING
print("waiting until:", state.waiting.until)

for _ in range(20):
    time.sleep(0.2)
    state = sr.get_state(run_id)
    if state.status == RunStatus.COMPLETED:
        break

print(state.status.value, state.output)
sr.stop()
```

Expected:
- run first blocks with `wait_reason=until`
- within a few seconds, scheduler resumes and the run completes

---

## Test 4: Persistence (survive restart)

```python
import tempfile
from pathlib import Path

from abstractruntime import (
    create_scheduled_runtime,
    Effect,
    EffectType,
    StepPlan,
    WorkflowSpec,
    JsonFileRunStore,
    JsonlLedgerStore,
    RunStatus,
)


def ask(run, ctx):
    return StepPlan(
        node_id="ask",
        effect=Effect(type=EffectType.ASK_USER, payload={"prompt": "Continue?"}, result_key="answer"),
        next_node="done",
    )


def done(run, ctx):
    return StepPlan(node_id="done", complete_output={"answer": run.vars.get("answer")})


workflow = WorkflowSpec(workflow_id="persistent_wf", entry_node="ask", nodes={"ask": ask, "done": done})
data_dir = Path(tempfile.mkdtemp())

# Session 1: start + block
sr1 = create_scheduled_runtime(run_store=JsonFileRunStore(data_dir), ledger_store=JsonlLedgerStore(data_dir))
run_id, state = sr1.run(workflow)
assert state.status == RunStatus.WAITING
sr1.stop()

# Session 2: “restart”, reload + resume
sr2 = create_scheduled_runtime(
    run_store=JsonFileRunStore(data_dir),
    ledger_store=JsonlLedgerStore(data_dir),
    workflows=[workflow],  # re-register
)
state = sr2.get_state(run_id)
assert state.status == RunStatus.WAITING
state = sr2.respond(run_id, {"text": "yes"})
assert state.status == RunStatus.COMPLETED
print(state.output)
sr2.stop()
```

Expected:
- run id remains valid after a restart
- ledger and checkpoint files exist under `data_dir`

---

## Test 5: Find waiting runs

```python
from abstractruntime import create_scheduled_runtime, Effect, EffectType, StepPlan, WorkflowSpec, WaitReason, RunStatus


def wait_for_event(run, ctx):
    return StepPlan(node_id="wait", effect=Effect(type=EffectType.WAIT_EVENT, payload={"wait_key": f"event_{run.run_id[:8]}"}))


workflow = WorkflowSpec(workflow_id="event_wf", entry_node="wait", nodes={"wait": wait_for_event})

sr = create_scheduled_runtime()
ids = [sr.run(workflow)[0] for _ in range(3)]

waiting = sr.find_waiting_runs()
waiting_events = sr.find_waiting_runs(wait_reason=WaitReason.EVENT)

print("waiting:", len(waiting), "events:", len(waiting_events))
assert all(r.status == RunStatus.WAITING for r in waiting_events)

sr.stop()
```

Expected:
- at least 3 waiting runs are listed
- filtering by `WaitReason.EVENT` works

## Run the automated tests

```bash
python -m pytest -q
```

Expected: the test suite passes.

## See also

- `getting-started.md` — first steps
- `../examples/README.md` — runnable examples
- `architecture.md` — where these behaviors come from
