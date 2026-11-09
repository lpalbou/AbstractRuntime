### Changed

- **The kill switch cancels the remembered return; nothing is restored after it** (`identity/life.py`; CONTRACT v3
  D3, operator 2026-11-09: "personal time is never given back during a resume"). `hard_stop_loop` writes
  `resume_to = null` (`written_by=kill_switch`) whatever the loop's fate - a home still on the gateway sidecar is
  shadowed too - and reports it (`resume_to_cleared`, plus `resume_to_note` when a corrupt phases.yaml refused the
  write). A loop started after a kill-switch stop never returns to the own time the stopped loop was living, whatever
  `return_to` its host passes: it works the queued jobs and ends idle (`jobs_done`). Only the operator's own act
  (`first_day_unconditional`: a grant or a start) opens own time again.
