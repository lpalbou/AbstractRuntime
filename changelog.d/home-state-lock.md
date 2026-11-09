### Fixed

- **One lock and one generation for a home's phase state: a late writer can no longer undo Stop**
  (`identity/life.py`; Codex review 2 #1, agreed with the gateway). `home_state_lock(home, timeout_s=30.0)` is the one
  cross-process lock (fcntl.flock on `<home>/.phase_state.lock`, re-entrant per thread, `timeout_s=None` waits
  without a limit, `HomeStateLockTimeout` otherwise); the gateway takes it INSIDE its own per-home lock and no runtime
  path calls out while holding it. Every phases.yaml write (`write_personal_grant`, `write_resume_to`, the
  migration) re-reads and merges against the current file inside it, and so does `write_entity_state`. The lifecycle
  generation (`lifecycle_generation` in phases.yaml; `read_lifecycle_generation`, `bump_lifecycle_generation`
  returning the new value) fences them: those writers take `expected_generation=` and raise
  `StaleLifecycleGeneration` (writing nothing) when it moved. `hard_stop_loop` revokes personal, clears resume_to and
  bumps the generation in ONE critical section before any signal (result `lifecycle_generation`). The own-time loop
  fences its state and resume_to writes on the generation it started at and ends (`stopped_by=kill_switch`) when it
  moves.
