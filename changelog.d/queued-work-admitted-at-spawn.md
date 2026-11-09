### Fixed

- **Queued work starts with own time off** (`identity/life.py`; Codex review 2 #5, CONTRACT v3 D1). The spawn door
  (`spawn_loop_process`) and the CLI start door refused every start without the personal grant, so a job queued
  while own time was off never ran. `loop_start_refusal(home)` admits a start when a job is queued or the grant is
  armed, and refuses otherwise (`no personal time: <why> - and no job is queued`); the ownership checks (one start
  at a time, a running loop) are unchanged.
