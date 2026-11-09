### Changed

- **The Stop fence: every runtime state write carries a `LifecycleFence`, required by signature** (new
  `identity/lifecycle_fence.py`; STOP-LANE-DESIGN 2026-11-09 §2.1-2.2; Codex review 3 #5). Work is admitted at the
  home's lifecycle generation and carries the fence; `write_entity_state`, `write_personal_grant`, `write_resume_to`,
  `migrate_phase_sidecar`, `write_loop_status`, `touch_loop_heartbeat`, `record_loop_substrate` and
  `append_phase_changed` take `fence=` keyword-only with **no default** (the optional `expected_generation` is gone),
  check it inside the home state lock against the generation read there, and refuse a write decided before a Stop
  (`StaleLifecycleGeneration`, now a `StoppedByOperator`, itself a `TurnCancelled`: a clean "stopped by the operator"
  record, never a failure). Only `KILL_SWITCH` (paused, own time revoked, the return cleared, loop status stopped) and
  `RESUME` (awake, nothing restored) pass unchecked, and only for those writes. **Breaking for hosts**: every gateway
  call of these writers must pass a fence (an unfenced call raises `TypeError` and writes nothing).
- **The sidecar migration is one locked, fenced step**: the bucket's absence, the sidecar read and the write happen
  inside the home state lock, so a migration racing a Stop can no longer bring back an old personal return.

### Fixed

- **The own-time loop fails closed and writes nothing after a Stop** (design §1 F/H, I7): an unreadable phases.yaml
  used to leave the loop running "not fenced", and any read error counted as no Stop; now the loop does not start (no
  status, no state, no wake), and a generation that moves or becomes unreadable ends it at its next boundary. After a
  Stop mid-day the loop no longer runs the day's look-back (a model call), and never writes its status, state, spend,
  personal-time usage, phase markers or task status over the Stop's records; it says so once, never as `#FALLBACK`.
- **The check is an int compare** (design §7): a process-wide table plus `<home>/.lifecycle_generation`, replaced
  atomically by `bump_lifecycle_generation` under the home state lock after phases.yaml. The bumping process sees its
  own Stop with zero I/O (measured 0.1 us per check); any other process pays one stat per check (about 2 us), reading
  the int only when it changed - instead of parsing phases.yaml at every cancellation check (about 300 us).
- **`write_loop_status` no longer runs `ps` to merge the prior status** (it reads the raw file, and the start-time
  token is read before the lock).
