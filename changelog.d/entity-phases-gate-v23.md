### Changed

- **The phase graph is spec v23 and the day gate follows it** (`identity/life.py`, `identity/phase_spec.py`,
  `identity/spec/entity_phases.vendored.json`; contract-phases v2 §9, §10.3-§10.5; TARGET 2026-11-08; spec v23
  OWN-TIME-IS-A-PRIVILEGE). The vendored graph is re-vendored byte for byte from abstractentity (v23: four phases,
  the statuses idle and stopped, the self-loops, no sleep->sleep). `read_day_gate` decides, in order: a visit
  posture; HIS OWN SLEEP DECISION (a ```rest block he wrote, passed as `sleep_elected`) unless the last activity
  phase was already sleep (`sleep_refusal`: no sleep->sleep) or it came from own time the operator has since
  revoked; a standing work order with Work on; personal while the operator's grant stands (a day when drives pull;
  the first day after the operator's start when the loop is started with `first_day_unconditional=True`; otherwise
  he RESTS between days inside personal, awake, zero model calls, waking early for a visit, a work order or a
  revoked grant); else `idle`: own time ends awake (`stopped_by=personal_off`, `work_off` or `edge_removed`). The
  gate never chooses sleep by itself. Every loop-written boundary consults the effective graph by its own edge
  (`personal->personal#personal_cycle`, `work->work#task_complete`, `work->personal#no_task|#operator`,
  `personal->work#operator`, `sleep->personal|work#maintenance_done`, `personal|work->sleep#self_elected`); a
  removed edge skips the leg and the floor is idle, never sleep. A self-loop writes no phase marker. 24/7 mode:
  a finished work order no longer ends the loop - the gate chooses another order (work->work), own time
  (work->personal) or idle. The day's session opens in the gate's phase (`build_session_factory`'s factory takes
  `phase`; the work contract rides only a work day).
- **No clock drives a sleep** (contract §10.4; spec v22 retirements). Removed from the loop: the v11 timed personal
  cycle, the 6 h unattended need-check, the 1 h sleep bound over an operator sleep, the settled-desk sleep and the
  2 h grant-use floor. Their dials stay readable and are `wired: False` (`TUNABLE_RETIRED`; an edit warns that it
  governs nothing). An operator sleep lasts until the operator's wake or its maintenance's end (the gateway writes
  awake); a visit until the door's close.
- **The maintenance signals feed his decision**: when `abstractmemory.maintenance_due` says a pass is due, the day
  cue carries one line of facts (new records since the last sleep, records sharing a title, unconnected records)
  and says a ```rest block is his to write if and when he chooses. No text is read; nothing sleeps him.
