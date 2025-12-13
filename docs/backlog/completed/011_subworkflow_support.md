## 011_subworkflow_support (completed)

**Status**: Completed  
**Completed**: 2025-12-13  
**Final Review**: 2025-12-13

---

## Final Report

### What Was Implemented

| Feature | Description | Tests |
|---------|-------------|-------|
| `START_SUBWORKFLOW` effect | Sync and async modes for workflow composition | 5 |
| `parent_run_id` tracking | Parent-child relationship in RunState | 2 |
| `list_children()` | Query child runs by parent | 1 |
| `resume_subworkflow_parent()` | Auto-resume parent when child completes | 1 |
| `RunStatus.CANCELLED` | Explicit cancellation status | 3 |
| `cancel_run()` | Cancel a single run | 3 |
| `cancel_with_children()` | Cascading cancellation | 2 |

**Total: 18 subworkflow tests, 75 tests overall (all passing)**

### Architecture Decision

After thorough analysis, we determined that the subworkflow implementation should remain **minimal and focused**. The following were explicitly **not added** to the kernel:

| Feature | Reason Not Added |
|---------|------------------|
| Timeout as effect parameter | Achievable with WAIT_UNTIL pattern (documented) |
| PARALLEL_SUBWORKFLOWS effect | Achievable with async + polling pattern (documented) |
| Remote execution primitives | Belongs in AbstractFlow, not the kernel |

The kernel provides **primitives**; AbstractFlow will provide **patterns**.

### Files Modified

**Core:**
- `src/abstractruntime/core/models.py` - Added `parent_run_id`, `RunStatus.CANCELLED`
- `src/abstractruntime/core/runtime.py` - Subworkflow handler, `cancel_run()`

**Storage:**
- `src/abstractruntime/storage/base.py` - Added `list_children()` protocol
- `src/abstractruntime/storage/in_memory.py` - Implemented `list_children()`
- `src/abstractruntime/storage/json_files.py` - Implemented `list_children()`, `parent_run_id` persistence

**Scheduler:**
- `src/abstractruntime/scheduler/scheduler.py` - `resume_subworkflow_parent()`, `cancel_with_children()`
- `src/abstractruntime/scheduler/convenience.py` - Exposed methods on ScheduledRuntime

**Tests:**
- `tests/test_subworkflow.py` - 18 comprehensive tests

### Recommendations for Next Steps

1. **Artifact Store (009)** - High priority for multi-agent systems
2. **Effect Retries/Idempotency (013)** - Required for production reliability
3. **AbstractFlow Integration** - Build higher-level patterns on these primitives

---

## Implementation Summary

### Core Features
- `START_SUBWORKFLOW` effect handler in `runtime.py`
- **Sync mode**: Blocks parent until child completes, waits, or fails
- **Async mode**: Returns immediately with `sub_run_id` for fire-and-forget patterns
- Uses existing `WorkflowRegistry` for workflow lookup
- Child runs have independent state and ledger entries

### Parent-Child Tracking (Added in Review)
- `RunState.parent_run_id` field tracks workflow hierarchy
- `list_children()` method on run stores to query child runs
- `resume_subworkflow_parent()` on Scheduler for auto-resuming parents when children complete

### Files Modified
- `src/abstractruntime/core/models.py` - Added `parent_run_id` field
- `src/abstractruntime/core/runtime.py` - Subworkflow effect handler
- `src/abstractruntime/storage/base.py` - Added `list_children()` to QueryableRunStore
- `src/abstractruntime/storage/in_memory.py` - Implemented `list_children()`
- `src/abstractruntime/storage/json_files.py` - Implemented `list_children()`, added `parent_run_id` persistence
- `src/abstractruntime/scheduler/scheduler.py` - Added `resume_subworkflow_parent()`
- `src/abstractruntime/scheduler/convenience.py` - Exposed `resume_subworkflow_parent()` on ScheduledRuntime

### Test Coverage (13 tests)
- Sync mode: child completes, child waits, child fails
- Async mode: parent continues immediately
- Nested workflows (3 levels deep)
- Error cases: missing registry, unknown workflow, missing workflow_id
- Integration with ScheduledRuntime
- Parent-child tracking and auto-resume

---

## Architecture Analysis for Multi-Agent Systems

### Current Capabilities

The subworkflow implementation provides the foundation for:

1. **Agent Composition**: An agent is a workflow. A supervisor agent can spawn specialist agents as subworkflows.

2. **Memory Pipelines**: Background memory maintenance jobs can run as subworkflows.

3. **Reusable Building Blocks**: Common patterns (search, summarize, validate) can be packaged as workflows and composed.

### Execution Semantics

| Mode | Parent Behavior | Child Behavior | Use Case |
|------|-----------------|----------------|----------|
| Sync | Blocks until child done | Runs in-process | Sequential agent handoff |
| Async | Continues immediately | Started but not ticked | Fire-and-forget tasks |

### Distributed Execution Considerations

The current implementation runs subworkflows **in-process**. For distributed scenarios:

1. **Shared Storage**: Parent and child share the same `RunStore` and `LedgerStore`. For distributed execution, use a shared backend (e.g., PostgreSQL, Redis).

2. **Remote Subworkflow Pattern**: For cross-server execution:
   - Use async mode to start the subworkflow
   - Return `WAITING` with `WaitReason.SUBWORKFLOW`
   - A remote worker picks up the child run and executes it
   - When child completes, call `resume_subworkflow_parent()` to continue parent

3. **Artifact Passing**: For workflows that need artifacts from previous steps:
   - Store artifacts by reference (see backlog 009_artifact_store)
   - Pass artifact IDs in `vars` between workflows

---

## Cancellation Support (Added in Final Review)

### RunStatus.CANCELLED
Added `CANCELLED` status for explicit run cancellation (distinct from `FAILED`).

### cancel_run()
```python
runtime.cancel_run(run_id, reason="User requested")
```
Cancels a single run. Only RUNNING or WAITING runs can be cancelled.

### cancel_with_children()
```python
scheduler.cancel_with_children(run_id, reason="Parent cancelled")
# Returns list of all cancelled runs
```
Cascading cancellation that traverses the parent-child tree.

---

## Patterns for Advanced Use Cases

The subworkflow primitives support advanced patterns without adding kernel complexity.

### Pattern 1: Timeout with Deadline

Use `WAIT_UNTIL` combined with async subworkflow:

```python
def start_with_timeout(run: RunState, ctx) -> StepPlan:
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
    run.vars["deadline"] = deadline
    return StepPlan(
        node_id="start",
        effect=Effect(
            type=EffectType.START_SUBWORKFLOW,
            payload={"workflow_id": "slow_task", "async": True},
            result_key="task_info",
        ),
        next_node="wait_for_deadline",
    )

def wait_for_deadline(run: RunState, ctx) -> StepPlan:
    return StepPlan(
        node_id="wait",
        effect=Effect(
            type=EffectType.WAIT_UNTIL,
            payload={"until": run.vars["deadline"]},
        ),
        next_node="check_result",
    )

def check_result(run: RunState, ctx) -> StepPlan:
    sub_run_id = run.vars["task_info"]["sub_run_id"]
    child_state = runtime.get_state(sub_run_id)
    
    if child_state.status == RunStatus.COMPLETED:
        return StepPlan(node_id="check", complete_output=child_state.output)
    else:
        # Timeout - cancel child and return error
        runtime.cancel_run(sub_run_id, reason="Timeout")
        return StepPlan(node_id="check", complete_output={"error": "timeout"})
```

### Pattern 2: Parallel Fan-Out

Start multiple async subworkflows, then poll for completion:

```python
def fan_out(run: RunState, ctx) -> StepPlan:
    # Start multiple children async
    sub_ids = []
    for source in ["web", "docs", "knowledge_graph"]:
        sub_id = runtime.start(
            workflow=search_workflow,
            vars={"source": source, "query": run.vars["query"]},
            parent_run_id=run.run_id,
        )
        sub_ids.append(sub_id)
    run.vars["sub_ids"] = sub_ids
    return StepPlan(node_id="fan_out", next_node="poll_children")

def poll_children(run: RunState, ctx) -> StepPlan:
    # Check if all children completed
    all_done = True
    results = []
    for sub_id in run.vars["sub_ids"]:
        state = runtime.get_state(sub_id)
        if state.status == RunStatus.COMPLETED:
            results.append(state.output)
        elif state.status in (RunStatus.RUNNING, RunStatus.WAITING):
            all_done = False
    
    if all_done:
        return StepPlan(node_id="poll", complete_output={"results": results})
    
    # Wait briefly then poll again
    return StepPlan(
        node_id="poll",
        effect=Effect(type=EffectType.WAIT_UNTIL, payload={"until": ...}),
        next_node="poll_children",
    )
```

### Pattern 3: Distributed Execution

For cross-server execution with shared storage:

```python
# Server A: Start subworkflow async
def start_remote_task(run: RunState, ctx) -> StepPlan:
    return StepPlan(
        node_id="start",
        effect=Effect(
            type=EffectType.START_SUBWORKFLOW,
            payload={"workflow_id": "gpu_task", "async": True},
            result_key="task_info",
        ),
        next_node="wait_for_completion",
    )

def wait_for_completion(run: RunState, ctx) -> StepPlan:
    # Return WAITING - external system will resume when child completes
    sub_run_id = run.vars["task_info"]["sub_run_id"]
    return StepPlan(
        node_id="wait",
        effect=Effect(
            type=EffectType.WAIT_EVENT,
            payload={"wait_key": f"subworkflow:{sub_run_id}"},
            result_key="child_output",
        ),
        next_node="done",
    )

# Server B (GPU worker): Poll for pending child runs and execute
pending = run_store.list_runs(status=RunStatus.RUNNING, workflow_id="gpu_task")
for run in pending:
    state = runtime.tick(workflow=gpu_workflow, run_id=run.run_id)
    if state.status == RunStatus.COMPLETED:
        # Resume parent on Server A
        scheduler.resume_subworkflow_parent(
            child_run_id=run.run_id,
            child_output=state.output,
        )
```

---

## Recommendations for Next Steps

### Immediate (High Priority)

1. **Implement Artifact Store (009)**: Required for passing large payloads between workflows without bloating `RunState.vars`.

2. **Implement Effect Retries/Idempotency (013)**: Required for production reliability when LLM calls or tools fail mid-execution.

### Medium Term

3. **Subworkflow Observability**: Add helper functions for:
   - Tree visualization of parent-child relationships
   - Aggregate status of workflow trees

### Future (AbstractFlow Integration)

4. **Declarative Subworkflow Nodes**: In AbstractFlow, provide a high-level DSL for subworkflow composition.

5. **Built-in Parallel/Fan-Out**: AbstractFlow can provide higher-level patterns that compile down to the primitives shown above.

### Explicitly Deferred

The following were considered but intentionally not added to the kernel:

- **Timeout as effect parameter**: Can be achieved with WAIT_UNTIL pattern
- **PARALLEL_SUBWORKFLOWS effect**: Can be achieved with async + polling pattern
- **Remote execution primitives**: Belongs in AbstractFlow, not the kernel

---

## Effect Payload Schema

```json
{
  "workflow_id": "child_workflow_id",
  "vars": {"key": "value"},
  "async": false
}
```

### Response (Sync, Completed)
```json
{
  "sub_run_id": "uuid",
  "output": {"child_output": "..."}
}
```

### Response (Async)
```json
{
  "sub_run_id": "uuid",
  "async": true
}
```

### WaitState (Sync, Child Waiting)
```json
{
  "reason": "subworkflow",
  "wait_key": "subworkflow:{sub_run_id}",
  "details": {
    "sub_run_id": "uuid",
    "sub_workflow_id": "child_workflow_id",
    "sub_waiting": {
      "reason": "user",
      "wait_key": "child_prompt"
    }
  }
}
```

---

## Original Proposal

### Goal
Implement `EffectType.START_SUBWORKFLOW` so a workflow can compose other workflows ("workflow-as-node").

This is the kernel primitive that enables:
- multi-agent orchestration (an agent is a workflow)
- memory maintenance pipelines (memory jobs as workflows)
- reusable building blocks in AbstractFlow

### Context / problem
We already model composition in `EffectType.START_SUBWORKFLOW`, but the runtime currently has **no handler** for it.

Without a first-class subworkflow primitive, higher-level systems will re-implement composition inconsistently (and break durability/resume semantics).

### Non-goals
- No Temporal-like distributed orchestration semantics.
- No cross-process worker leasing.
- No full workflow registry product/UX (that belongs to AbstractFlow).

---

### Proposed design (minimal + durable)

#### A) Subworkflow registry interface
A subworkflow must be referenced by a stable id (string), since effects must be JSON.

**Already implemented**: `WorkflowRegistry` exists in `abstractruntime.scheduler.registry`:

```python
from abstractruntime import WorkflowRegistry

registry = WorkflowRegistry()
registry.register(my_workflow)
registry.get(workflow_id) -> WorkflowSpec
registry.get_or_raise(workflow_id) -> WorkflowSpec
```

The scheduler already uses this. For subworkflows, we can reuse the same registry.

#### B) Effect payload schema

```json
{
  "workflow_id": "deepsearch_v1",
  "vars": {"query": "..."},
  "mode": "sync" | "async",
  "result_key": "optional.override" 
}
```

Notes:
- `vars` are the input variables for the child run.
- `mode=sync` runs the subworkflow in-process until it blocks or completes.
- `mode=async` starts it and returns a WAIT state immediately.

#### C) Execution semantics

**Mode: sync**
- Start a child run (`child_run_id = runtime.start(child_workflow, vars, actor_id=parent.actor_id)`)
- Tick it until:
  - completed → return `completed({child_run_id, output})`
  - failed → return `failed("child failed: ...")`
  - waiting → return `waiting(wait=WaitState(...))` where:
    - `reason=JOB` or `EVENT`
    - `details` includes `child_run_id` and child wait state

**Mode: async**
- Start child run
- Return `waiting(wait=WaitState(reason=JOB, wait_key=f"sub:{child_run_id}", details={child_run_id}))`
- The host (scheduler/worker) drives the child run and resumes the parent when done.

#### D) Resume propagation
Two acceptable designs:

1) **Host-driven** (simplest): parent is waiting on `sub:{child_run_id}`; the host resumes parent with the child output.
2) **Runtime-driven** (more complex): parent resume handler detects `details.child_run_id` and forwards resume payload to the child, then continues.

For v0.1/v0.2, prefer **host-driven** to avoid cross-run coupling inside the kernel.

---

### Files to add / modify
- ✅ `WorkflowRegistry` already exists in `src/abstractruntime/scheduler/registry.py`
- Add effect handler for `EffectType.START_SUBWORKFLOW` in `src/abstractruntime/core/runtime.py`
- Add tests:
  - sync subworkflow completes
  - sync subworkflow waits and parent becomes waiting
  - async subworkflow starts and returns wait

---

### Acceptance criteria
- A workflow can start a subworkflow by id and receive its output deterministically.
- Subworkflow runs are durable: child run state + ledger are persisted independently.
- Waiting subworkflow does not deadlock the parent; parent WAIT state is durable and resumable.

### Test plan
- Unit tests using in-memory stores (no external services).
- One file-based persistence test ensuring both parent+child survive restart.
