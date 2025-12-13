## 004_scheduler_driver (planned)

### Goal
Provide a minimal driver loop that:
- finds runs that are due for `wait_until`
- calls `tick()` to progress them

### Rationale
The kernel intentionally avoids building a Temporal-like orchestration backend.
However, production deployments need a component that:
- periodically polls waiting runs
- resumes time-based waits

### MVP design
- In-process polling loop (single process)
- Storage backend must support listing waiting runs (not implemented yet)

### Future
- Redis ZSET scheduler
- Postgres-based scheduling
- Event ingestion endpoint for `wait_event` / `ask_user`

