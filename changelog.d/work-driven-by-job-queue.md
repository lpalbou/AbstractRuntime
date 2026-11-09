### Added

- **Work is driven by the job queue** (`identity/life.py`; CONTRACT v3 D1/D4, operator 2026-11-09).
  `read_job_queue(home)` / `jobs_queued(home)` read the jobs queued for him, in the order he works them: the standing
  work order, then the task inbox's (`task_inbox.jsonl`, AbstractGateway's file) `taken` and `pending` tasks (parked
  and done tasks are not queued; the fold matches the gateway's). The day gate runs a work day whenever a job is
  queued, whatever the personal grant says; the decision carries `job` and `jobs_queued` (`loop_status.day_cause.kind`
  `work_order` or `task`). A task a work day opens is marked `taken`, and his ```work verdict marks it `done` or
  `parked` (`blocked:`) with the verdict line as its note (`append_task_status`, the gateway's flock-guarded
  single-line append; a plain append where there is no `fcntl`). The work day's contract names who left a task (`work_day_text`); `build_session_factory`'s
  factory takes the gate's `job`.
- **A job interrupts his own time and returns to it** (D4). When a work day opens while his own time is granted and is
  the phase the run returns to, the home remembers it (`resume_to = personal`, written by
  `loop:job-interrupts-own-time`). When every job is done the gate returns him to his own time while the grant still
  stands; otherwise own time ends idle with `stopped_by=jobs_done`. Any idle exit cancels a remembered return held in
  the runtime bucket (own time ended without an interruption).
