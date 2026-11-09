### Fixed

- **One loop per home through a slow start: the spawn reservation** (Codex review 3 #6; STOP-LANE-DESIGN §2.4).
  `spawn_loop_process` used to release its lock after a fixed 0.6 s whatever the child had done, so a second start
  could spawn a second loop over a home whose first child had not registered yet. Now, under the home state lock and
  before `Popen`, it writes the reservation `loop_status = {phase: starting, generation, spawner_pid,
  spawner_started_at}` (the child pid joins it after `Popen`, never over a child that already registered);
  `read_loop_status` reads it running while the recorded child - or, before `Popen` returned, the spawner - is alive
  (one `waitpid` + `os.kill(pid, 0)`, never `ps`, never a clock). A second start, concurrent or after the first
  returned, is refused ("already starting"); a hand start (`python -m abstractruntime.identity.life`) honors a running
  or reserved loop too. A child that dies at once releases the reservation (`stopped_by start_failed`); one refused
  by a Stop meanwhile surfaces as `StoppedByOperator`, not as a failed start. `LOOP_PHASES` gains `starting`.
- **A loop spawned before a Stop and booting after it writes nothing** (design §1 F): the spawner passes
  `--generation G` (the reservation's); the child checks it before anything else and exits "(stopped before it
  started ...)" with zero writes, and its first write (the boot claim replacing the reservation with its own
  `between`) is fenced at G. A kill switch that preceded the reservation is still known to the booting child
  (`after_stopped_by`), so nothing is restored.
- **The night's completion is fenced** (Codex review 3 #2; design class H): `build_consolidator` takes the loop's
  `fence` (required). A moved generation ends the night at the engine's next stage (`should_continue`), memory
  records no completed maintenance (`sleep_pass(cancelled=fence.moved)`, AbstractMemory f310865: the idle-sleep
  boundary is not reset), the repair step declares, re-arms and authors nothing (`_sleep_authoring(should_stop=)`,
  also handed to the drainer), and neither the night voice nor failure lines follow - one clean line instead.
