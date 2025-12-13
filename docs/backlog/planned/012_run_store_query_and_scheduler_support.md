## 012_run_store_query_and_scheduler_support (planned)

### Goal
Enable querying runs to support:
- scheduler/driver loops (resume `wait_until` runs)
- operational tooling (list waiting runs)
- UI backoffice views (runs by status)

### Context / problem
The current `RunStore` interface only supports `save()` and `load()`. That is sufficient for single-run manipulation, but insufficient for:
- finding *which runs are waiting*
- finding runs whose `wait_until` is due

Without query support, every host will reinvent scanning/indexing.

### Non-goals
- No distributed leasing.
- No global matching service.

---

### Proposed design

#### A) Keep the existing `RunStore` minimal
Do not break the kernel’s `RunStore` ABC.

#### B) Add an optional query interface
Introduce a second interface:

```python
class QueryableRunStore(Protocol):
    def list_runs(
        self,
        *,
        status: RunStatus | None = None,
        wait_reason: WaitReason | None = None,
        limit: int = 100,
    ) -> list[RunState]: ...

    def list_due_wait_until(
        self,
        *,
        now_iso: str,
        limit: int = 100,
    ) -> list[RunState]: ...
```

Scheduler/driver code can require `QueryableRunStore`.

#### C) Implementations
- **InMemoryRunStore**: trivial filter
- **JsonFileRunStore**:
  - scan `run_*.json`
  - parse minimal fields
  - filter by status/waiting

This is acceptable for MVP.

---

### Files to add / modify
- `src/abstractruntime/storage/queryable_run_store.py` (or add to `storage/base.py` as Protocol)
- Update `InMemoryRunStore` (optional)
- Update `JsonFileRunStore` (optional)
- Add tests:
  - list waiting runs
  - list due wait_until runs

---

### Acceptance criteria
- We can list WAITING runs without external indexing.
- We can list WAIT_UNTIL runs due at time T.
- Scheduler driver backlog `004_scheduler_driver.md` can be implemented cleanly.

### Test plan
- Unit tests for both in-memory and file-based stores.
