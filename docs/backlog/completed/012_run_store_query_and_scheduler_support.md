## 012_run_store_query_and_scheduler_support (completed)

### Goal
Enable querying runs to support:
- scheduler/driver loops (resume `wait_until` runs)
- operational tooling (list waiting runs)
- UI backoffice views (runs by status)

### What shipped

#### QueryableRunStore Protocol
`src/abstractruntime/storage/base.py`:
- `QueryableRunStore` — Protocol (structural typing) for query-capable stores
- Uses `@runtime_checkable` for isinstance() checks

#### Methods
```python
def list_runs(
    self,
    *,
    status: Optional[RunStatus] = None,
    wait_reason: Optional[WaitReason] = None,
    workflow_id: Optional[str] = None,
    limit: int = 100,
) -> List[RunState]:
    """List runs matching the given filters."""
    ...

def list_due_wait_until(
    self,
    *,
    now_iso: str,
    limit: int = 100,
) -> List[RunState]:
    """List runs waiting for a time threshold that has passed."""
    ...
```

#### Implementations
- `InMemoryRunStore` — implements QueryableRunStore
- `JsonFileRunStore` — implements QueryableRunStore (scans files)

#### Exports
- `QueryableRunStore` exported from `abstractruntime.storage` and `abstractruntime`

### Design decisions

1. **Protocol vs ABC**: Used `Protocol` (structural typing) so existing stores can add query methods without changing inheritance. This follows the backlog recommendation.

2. **Ordering**:
   - `list_runs()` returns results ordered by `updated_at` descending (most recent first)
   - `list_due_wait_until()` returns results ordered by `waiting.until` ascending (earliest due first)

3. **ISO string comparison**: For `list_due_wait_until`, we compare ISO 8601 strings directly. This works correctly for UTC timestamps because ISO 8601 is lexicographically sortable.

4. **File scanning for JsonFileRunStore**: The file-based implementation scans all `run_*.json` files. This is acceptable for MVP but may need indexing for large deployments.

### Tests
`tests/test_queryable_run_store.py` — 20 tests covering:
- Protocol implementation verification
- Empty store handling
- Filter by status, wait_reason, workflow_id
- Combined filters
- Limit parameter
- Ordering (updated_at desc, until asc)
- Persistence across store instances (JsonFileRunStore)

### Test results
```
30 passed, 1 skipped in 0.14s
```

### Next steps
This enables implementation of:
- [004_scheduler_driver.md](../planned/004_scheduler_driver.md) — Built-in scheduler

### Related
- [ADR 0001: Layered Coupling](../../adr/0001_layered_coupling_with_abstractcore.md) — Query support stays in storage layer
