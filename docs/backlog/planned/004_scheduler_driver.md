## 004_scheduler_driver (planned)

### Goal
Provide a **built-in, zero-config scheduler** that automatically resumes waiting runs.

### Rationale
The current design punts scheduling to the host: "the host is responsible for driving `tick()`". This is a UX problem — every user has to build their own scheduler, which is a significant adoption barrier.

**Our design principle**: simplify UX as much as possible. Any step we can automate, we should.

### Requirements
1. **Zero-config for simple cases**: `runtime.start_scheduler()` just works
2. **Automatic `wait_until` resumption**: runs waiting for a time threshold resume automatically
3. **Event ingestion API**: simple way to resume `wait_event` / `ask_user` runs
4. **Pluggable backends**: in-process polling (MVP), Redis ZSET, Postgres (future)

### MVP design

#### A) In-process scheduler (default)
```python
class Scheduler:
    def __init__(self, runtime: Runtime, poll_interval_s: float = 1.0):
        ...
    
    def start(self) -> None:
        """Start background polling thread."""
        ...
    
    def stop(self) -> None:
        """Stop scheduler gracefully."""
        ...
    
    def resume_event(self, wait_key: str, payload: dict) -> RunState:
        """Resume a run waiting for an event."""
        ...
```

#### B) Integration with Runtime
```python
# Option 1: Scheduler as separate component
scheduler = Scheduler(runtime)
scheduler.start()

# Option 2: Built into Runtime (simpler UX)
runtime = Runtime(..., auto_scheduler=True)
```

#### C) Polling loop
```python
def _poll_loop(self):
    while self._running:
        # Find due wait_until runs
        due_runs = self._run_store.list_due_wait_until(now_iso=utc_now_iso())
        for run in due_runs:
            self._runtime.tick(workflow=..., run_id=run.run_id)
        
        time.sleep(self._poll_interval)
```

### Dependencies
- **012_run_store_query_and_scheduler_support.md**: Must be implemented first

### Future enhancements
- Redis ZSET scheduler (distributed, persistent timers)
- Postgres-based scheduling (transactional guarantees)
- Webhook ingestion endpoint for external events
- Workflow registry for automatic workflow spec lookup

### Acceptance criteria
- A `wait_until` run automatically resumes when its time arrives (no manual `tick()` call)
- A `wait_event` run can be resumed via `scheduler.resume_event(wait_key, payload)`
- Scheduler can be started/stopped cleanly
- Works with both in-memory and file-based stores

### Related ADRs
- Consider creating ADR 0004: Built-in Scheduler (why zero-config matters)

