## 004_scheduler_driver (completed)

### Goal
Provide a **built-in, zero-config scheduler** that automatically resumes waiting runs.

### What shipped

#### New Module: `src/abstractruntime/scheduler/`

**registry.py** — WorkflowRegistry
```python
registry = WorkflowRegistry()
registry.register(workflow)
registry.get(workflow_id) -> WorkflowSpec
registry.get_or_raise(workflow_id) -> WorkflowSpec
registry.unregister(workflow_id)
registry.list_ids() -> list[str]
```

**scheduler.py** — Scheduler
```python
scheduler = Scheduler(
    runtime=runtime,
    registry=registry,
    poll_interval_s=1.0,
    on_run_resumed=callback,
    on_run_failed=callback,
)
scheduler.start()  # Start background polling thread
scheduler.stop()   # Stop gracefully
scheduler.poll_once() -> int  # Manual poll (for testing)
scheduler.resume_event(run_id, wait_key, payload) -> RunState
scheduler.find_waiting_runs(wait_reason, workflow_id, limit) -> list[RunState]
scheduler.stats -> SchedulerStats
scheduler.is_running -> bool
```

**convenience.py** — Zero-config helpers
```python
# Zero-config: defaults to in-memory storage, auto-starts scheduler
sr = create_scheduled_runtime()

# run() does start + tick in one call
run_id, state = sr.run(my_workflow)

# respond() auto-finds wait_key - no need to specify it
if state.status.value == "waiting":
    state = sr.respond(run_id, {"answer": "yes"})

# tick() no longer needs workflow - looks it up from registry
state = sr.tick(run_id)

# Stop when done
sr.stop()
```

#### Exports
Added to `abstractruntime`:
- `WorkflowRegistry`
- `Scheduler`
- `SchedulerStats`
- `ScheduledRuntime`
- `create_scheduled_runtime`

### Design decisions

1. **Separate component (not built into Runtime)**: Keeps concerns separated. The Runtime handles execution; the Scheduler handles timing. This avoids circular dependencies and keeps the Runtime simple.

2. **WorkflowRegistry**: The scheduler needs to map `workflow_id` → `WorkflowSpec` to call `tick()`. The registry provides this mapping and can be shared across components.

3. **Thread-based polling**: Uses a background daemon thread with `threading.Event` for clean shutdown. Simpler than asyncio and works well for the MVP.

4. **QueryableRunStore requirement**: The scheduler requires a `QueryableRunStore` (checked at construction). This ensures `list_due_wait_until()` is available.

5. **ScheduledRuntime convenience wrapper**: Provides a unified API that combines Runtime + Scheduler + Registry for zero-config operation.

6. **Callbacks for observability**: Optional `on_run_resumed` and `on_run_failed` callbacks for monitoring.

7. **Stats tracking**: `SchedulerStats` tracks poll cycles, runs resumed, runs failed, and recent errors.

### Acceptance criteria (met)

- ✅ A `wait_until` run automatically resumes when its time arrives (no manual `tick()` call)
- ✅ A `wait_event` run can be resumed via `scheduler.resume_event(wait_key, payload)`
- ✅ Scheduler can be started/stopped cleanly
- ✅ Works with both in-memory and file-based stores

### Tests
`tests/test_scheduler.py` — 24 tests covering:
- WorkflowRegistry (6 tests)
- Scheduler (11 tests)
- ScheduledRuntime (7 tests)

### Test results
```
54 passed, 1 skipped in 0.35s
81% overall coverage
```

### Files created

| File | Description |
|------|-------------|
| `src/abstractruntime/scheduler/__init__.py` | Module exports |
| `src/abstractruntime/scheduler/registry.py` | WorkflowRegistry |
| `src/abstractruntime/scheduler/scheduler.py` | Scheduler class |
| `src/abstractruntime/scheduler/convenience.py` | ScheduledRuntime + factory |
| `tests/test_scheduler.py` | 24 tests |

### Future enhancements (not in scope for v0.1)
- Redis ZSET scheduler (distributed, persistent timers)
- Postgres-based scheduling (transactional guarantees)
- Webhook ingestion endpoint for external events

### Related
- Prerequisite: [012_run_store_query_and_scheduler_support.md](012_run_store_query_and_scheduler_support.md)
