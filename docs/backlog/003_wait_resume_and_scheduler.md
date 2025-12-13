# 003_wait_resume_and_scheduler

> Legacy note: preserved for history. See `docs/backlog/completed/003_wait_primitives.md` and `docs/backlog/planned/004_scheduler_driver.md`.

## Goal
Support long pauses with two mechanisms:

1) `wait_event(key)`
- pauses until an external resume event arrives

2) `wait_until(when)`
- pauses until a time threshold is reached

## Deliverables
- `abstractruntime/waiters/wait_event.py`
- `abstractruntime/waiters/wait_until.py`
- minimal scheduler abstraction:
  - in-process poller (MVP)
  - later: pluggable scheduler backends (Redis ZSET, Postgres, etc.)

## Acceptance criteria
- A run can pause for N seconds and resume automatically.
- A run can pause indefinitely for an event and resume when `resume(run_id, event)` is called.

## Notes
This is **not** a Temporal clone. For v0.1 we can implement a simple in-process scheduler and document that production deployments should run a scheduler worker.


