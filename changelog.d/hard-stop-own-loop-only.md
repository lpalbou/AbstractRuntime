### Fixed

- **The hard stop signals only a live own-time loop, never its caller, and says what it did**
  (`identity/life.py` `hard_stop_loop`, `write_loop_status`; gateway slice finding 2026-11-09). After a freeze the
  status file was written by the CALLING process, so it named the caller's pid and start time; a second hard stop
  then read the caller as the loop and SIGTERMed it (it killed the gateway's test process). Now the status names
  the loop it is about (`write_loop_status(..., owner=(pid, token))`, or no pid), `stopped_by=kill_switch`, and
  before any signal the pid must not be the caller's own, must be alive, must match its start-time token, and its
  command line must run the loop module (`-m abstractruntime.identity.life`). A process that has exited (reaped, or
  a zombie its parent has not reaped) counts as gone. The result is explicit: `{killed, pid, signal:
  "SIGTERM"|"SIGKILL"|None, reason, requested_reason}` (`killed` only once the process is confirmed gone), plus the
  legacy keys (`frozen`, `was_running`, `escalated_to_sigkill`, `requested_by`, `status`).
