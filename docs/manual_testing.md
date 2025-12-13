# Manual Testing Guide

This guide shows how to manually test AbstractRuntime's core functionality.

## Prerequisites

```bash
cd /workspaces/workspaces/abstractruntime
source .venv/bin/activate  # or: .venv/bin/python
```

## Test 1: Zero-Config Hello World

```python
from abstractruntime import create_scheduled_runtime, Effect, EffectType, StepPlan, WorkflowSpec

# Define a simple workflow
def greet(run, ctx):
    name = run.vars.get("name", "World")
    return StepPlan(node_id="greet", complete_output={"message": f"Hello, {name}!"})

workflow = WorkflowSpec(
    workflow_id="hello",
    entry_node="greet",
    nodes={"greet": greet},
)

# Zero-config runtime
sr = create_scheduled_runtime()

# Run the workflow
run_id, state = sr.run(workflow, vars={"name": "Alice"})

print(f"Status: {state.status.value}")
print(f"Output: {state.output}")

sr.stop()
```

Expected output:
```
Status: completed
Output: {'message': 'Hello, Alice!'}
```

---

## Test 2: Ask User (Pause and Resume)

```python
from abstractruntime import create_scheduled_runtime, Effect, EffectType, StepPlan, WorkflowSpec

def ask_name(run, ctx):
    return StepPlan(
        node_id="ask",
        effect=Effect(
            type=EffectType.ASK_USER,
            payload={"prompt": "What is your name?", "wait_key": "name_prompt"},
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

# Start the workflow - it will pause waiting for user input
run_id, state = sr.run(workflow)
print(f"Status: {state.status.value}")
print(f"Waiting for: {state.waiting.reason.value}")
print(f"Prompt: {state.waiting.prompt}")

# Simulate user responding
state = sr.respond(run_id, {"text": "Bob"})
print(f"Status: {state.status.value}")
print(f"Output: {state.output}")

sr.stop()
```

Expected output:
```
Status: waiting
Waiting for: user
Prompt: What is your name?
Status: completed
Output: {'greeting': 'Hello, Bob!'}
```

---

## Test 3: Wait Until (Scheduled Resumption)

```python
from datetime import datetime, timezone, timedelta
from abstractruntime import create_scheduled_runtime, Effect, EffectType, StepPlan, WorkflowSpec

def schedule_task(run, ctx):
    # Wait for 2 seconds from now
    wait_until = (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()
    return StepPlan(
        node_id="schedule",
        effect=Effect(
            type=EffectType.WAIT_UNTIL,
            payload={"until": wait_until},
        ),
        next_node="execute",
    )

def execute_task(run, ctx):
    return StepPlan(node_id="execute", complete_output={"executed_at": datetime.now(timezone.utc).isoformat()})

workflow = WorkflowSpec(
    workflow_id="scheduled_task",
    entry_node="schedule",
    nodes={"schedule": schedule_task, "execute": execute_task},
)

sr = create_scheduled_runtime(poll_interval_s=0.5)  # Poll every 0.5s

run_id, state = sr.run(workflow)
print(f"Status: {state.status.value}")
print(f"Waiting until: {state.waiting.until}")

# Wait for the scheduler to resume it automatically
import time
print("Waiting for scheduler to resume...")
for _ in range(10):
    time.sleep(0.5)
    state = sr.get_state(run_id)
    if state.status.value == "completed":
        break

print(f"Status: {state.status.value}")
print(f"Output: {state.output}")

sr.stop()
```

Expected output:
```
Status: waiting
Waiting until: 2025-12-13T...
Waiting for scheduler to resume...
Status: completed
Output: {'executed_at': '2025-12-13T...'}
```

---

## Test 4: Persistence (Survive Restart)

```python
import tempfile
from pathlib import Path
from abstractruntime import (
    create_scheduled_runtime, Effect, EffectType, StepPlan, WorkflowSpec,
    JsonFileRunStore, JsonlLedgerStore,
)

# Create a temp directory for persistence
data_dir = Path(tempfile.mkdtemp())
print(f"Data directory: {data_dir}")

def ask_question(run, ctx):
    return StepPlan(
        node_id="ask",
        effect=Effect(
            type=EffectType.ASK_USER,
            payload={"prompt": "Continue?", "wait_key": "continue_prompt"},
            result_key="answer",
        ),
        next_node="done",
    )

def done(run, ctx):
    return StepPlan(node_id="done", complete_output={"answer": run.vars.get("answer")})

workflow = WorkflowSpec(
    workflow_id="persistent_wf",
    entry_node="ask",
    nodes={"ask": ask_question, "done": done},
)

# Session 1: Start workflow and pause
print("=== Session 1: Start and pause ===")
sr1 = create_scheduled_runtime(
    run_store=JsonFileRunStore(data_dir),
    ledger_store=JsonlLedgerStore(data_dir),
)
run_id, state = sr1.run(workflow)
print(f"Run ID: {run_id}")
print(f"Status: {state.status.value}")
sr1.stop()

# Session 2: Resume from disk (simulating restart)
print("\n=== Session 2: Resume after 'restart' ===")
sr2 = create_scheduled_runtime(
    run_store=JsonFileRunStore(data_dir),
    ledger_store=JsonlLedgerStore(data_dir),
    workflows=[workflow],  # Re-register workflow
)

# Load the state
state = sr2.get_state(run_id)
print(f"Loaded status: {state.status.value}")

# Resume
state = sr2.respond(run_id, {"text": "yes"})
print(f"Final status: {state.status.value}")
print(f"Output: {state.output}")
sr2.stop()

# Cleanup
import shutil
shutil.rmtree(data_dir)
```

Expected output:
```
Data directory: /tmp/...
=== Session 1: Start and pause ===
Run ID: ...
Status: waiting

=== Session 2: Resume after 'restart' ===
Loaded status: waiting
Final status: completed
Output: {'answer': {'text': 'yes'}}
```

---

## Test 5: Find Waiting Runs

```python
from abstractruntime import create_scheduled_runtime, Effect, EffectType, StepPlan, WorkflowSpec, WaitReason

def wait_for_event(run, ctx):
    return StepPlan(
        node_id="wait",
        effect=Effect(
            type=EffectType.WAIT_EVENT,
            payload={"wait_key": f"event_{run.run_id[:8]}"},
        ),
        next_node="done",
    )

def done(run, ctx):
    return StepPlan(node_id="done", complete_output={"done": True})

workflow = WorkflowSpec(
    workflow_id="event_wf",
    entry_node="wait",
    nodes={"wait": wait_for_event, "done": done},
)

sr = create_scheduled_runtime()

# Start multiple runs
run_id1, _ = sr.run(workflow)
run_id2, _ = sr.run(workflow)
run_id3, _ = sr.run(workflow)

# Find all waiting runs
waiting = sr.find_waiting_runs()
print(f"Total waiting: {len(waiting)}")

# Filter by wait reason
waiting_events = sr.find_waiting_runs(wait_reason=WaitReason.EVENT)
print(f"Waiting for events: {len(waiting_events)}")

# Show details
for run in waiting:
    print(f"  - {run.run_id[:8]}: waiting for {run.waiting.wait_key}")

sr.stop()
```

Expected output:
```
Total waiting: 3
Waiting for events: 3
  - abc12345: waiting for event_abc12345
  - def67890: waiting for event_def67890
  - ghi11111: waiting for event_ghi11111
```

---

## Running All Tests

To run the automated test suite:

```bash
cd /workspaces/workspaces/abstractruntime
.venv/bin/python -m pytest tests/ -v
```

Expected: 57 passed, 1 skipped
