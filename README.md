## AbstractRuntime

**AbstractRuntime** is a low-level **durable workflow runtime**:
- Execute workflow graphs (state machines)
- **Interrupt → checkpoint → resume** (hours/days) without keeping Python stacks alive
- Append-only **ledger** ("journal d’exécution") for audit/debug/provenance

**Status**: MVP kernel + file persistence + AbstractCore integration adapters are implemented.

---

### Key concepts
- **WorkflowSpec**: graph definition (node handlers keyed by id)
- **RunState**: durable state (`current_node`, `vars`, `waiting`, etc.)
- **Effect**: a side-effect request (`llm_call`, `tool_calls`, `ask_user`, `wait_event`, ...)
- **Ledger**: append-only step records (`StepRecord`) describing what happened

---

### Quick start (pause + resume)

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

---

### Built-in Scheduler

AbstractRuntime includes a zero-config scheduler for automatic run resumption:

```python
from abstractruntime import create_scheduled_runtime

# Zero-config: defaults to in-memory storage, auto-starts scheduler
sr = create_scheduled_runtime()

# run() does start + tick in one call
run_id, state = sr.run(my_workflow)

# If waiting for user input, respond (auto-finds wait_key)
if state.status.value == "waiting":
    state = sr.respond(run_id, {"answer": "yes"})

# Stop scheduler when done
sr.stop()
```

For production with persistent storage:

```python
from abstractruntime import create_scheduled_runtime, JsonFileRunStore, JsonlLedgerStore

sr = create_scheduled_runtime(
    run_store=JsonFileRunStore("./data"),
    ledger_store=JsonlLedgerStore("./data"),
)
```

---

### AbstractCore integration (LLM + tools)

AbstractRuntime’s kernel stays dependency-light; AbstractCore integration lives in:
- `src/abstractruntime/integrations/abstractcore/`

Execution modes:
- **Local**: in-process AbstractCore providers + local tool execution
- **Remote**: HTTP to AbstractCore server (`/v1/chat/completions`) + tool passthrough (default)
- **Hybrid**: remote LLM + local tool execution

See: `docs/integrations/abstractcore.md`.

---

### Snapshots / bookmarks

Snapshots are named, searchable checkpoints of a run state:
- `Snapshot(snapshot_id, run_id, name, description, tags, run_state)`

See: `docs/snapshots.md`.

---

### Provenance (tamper-evident ledger)

You can wrap any `LedgerStore` with a hash chain:

- `HashChainedLedgerStore(inner_store)`
- `verify_ledger_chain(records)`

This is **tamper-evident**, not non-forgeable (signatures are optional future work).

See: `docs/provenance.md`.

---

### Documentation index

| Document | Description |
|----------|-------------|
| [Proposal](docs/proposal.md) | Design goals, core concepts, and scope |
| [ROADMAP](ROADMAP.md) | Prioritized next steps with rationale |
| [ADRs](docs/adr/) | Architectural decisions and their rationale |
| [Backlog](docs/backlog/) | Completed and planned work items |
| [Integrations](docs/integrations/) | AbstractCore integration guide |
| [Snapshots](docs/snapshots.md) | Named checkpoints for run state |
| [Provenance](docs/provenance.md) | Tamper-evident ledger documentation |
