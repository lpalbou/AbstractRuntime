### Added

- **The Work toggle and resume_to are runtime-owned phases.yaml buckets** (`identity/life.py`; contract-phases v2
  §6.4, §10.2). `work: {enabled, written_by, written_at}` has one writer, `write_work_toggle(home, enabled=,
  written_by=)`, read by `read_work_toggle` / `work_enabled` (default on; a malformed bucket fails closed);
  `resume_to: {phase: personal|work|null, written_by, written_at}` (the own-time phase an interruption took over,
  spec v23) has one writer, `write_resume_to`, read by `read_resume_to`. Both writers field-merge (every other key
  of phases.yaml rides through; a corrupt file or a newer schema is refused). Until a home's buckets exist the reads
  fall back to the gateway sidecar `<entities_dir>/.life_phase/<slug>.json` (`work`, `resume_to`);
  `migrate_phase_sidecar(home)` copies them once, idempotently, after which the sidecar is a projection only.

### Changed

- **Work off pauses a work day at its next effect boundary** (`identity/life.py`, `identity/chat.py`; contract §10.2).
  The day gate skips the work leg while Work is off (the order waits; own time follows while the grant stands, else
  own time ends awake with `stopped_by=work_off`). During a work day the session's `yield_check` answers "Work was
  turned off": the in-flight call finishes, the tools its reply elected do not run (`[tool call not run: X - Work
  was turned off]`), the day closes at the tick boundary with its look-back, the order stands, and `work_pause.json`
  records the pause. With Work back on the next work day resumes the same order, tells him so, and marks the pause
  resumed. `ChatSession.yield_check` may return a string naming the cause (True still means a visitor). Work RUNS
  (summons, automations) pause the same way through `Runtime.pause_run` / `resume_run`, at the run's next step.
