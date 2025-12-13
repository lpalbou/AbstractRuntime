## 011_subworkflow_support (planned)

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

Introduce a small registry interface (can be in `abstractruntime.core` or `abstractruntime.runtime_context`):

```python
class WorkflowRegistry(Protocol):
    def get(self, workflow_id: str) -> WorkflowSpec: ...
```

A host can provide this via the `Runtime` context (DI).

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
- Add a planned registry protocol (location TBD): `src/abstractruntime/core/registry.py` or extend RunContext
- Add effect handler for `EffectType.START_SUBWORKFLOW`
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
