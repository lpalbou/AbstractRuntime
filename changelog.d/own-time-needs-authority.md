### Changed

- **Own time needs the operator's authority, never drives alone** (`identity/life.py`; spec v23
  OWN-TIME-IS-A-PRIVILEGE; Codex review 2026-11-09 #2). With the grant armed, `read_day_gate` admits personal only
  for the operator's start (`first_day`), a continuation of personal (`last_phase="personal"`) or a remembered return
  to it (`return_to="personal"`, or the home's `resume_to` bucket / the gateway sidecar). Otherwise own time ends
  awake: `{"phase": "idle", "cause": "no_return"}` and the loop exits with `stopped_by=no_return`. `LifeLoop(return_to=)`
  (default "personal": a direct run is the operator's act) remembers the phase its run was started to (re)enter, and a
  personal day keeps it through a sleep, a visit or a work day. `spawn_loop_process(return_to=)` / CLI `--return-to
  personal|work|none` carry it to the child; a host that passes nothing gets `none` (the home's resume_to decides).
