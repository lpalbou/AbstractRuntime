### Fixed

- **Turning personal off ends own time awake, never asleep** (`identity/life.py`; contract-phases v2 §10.1, root
  backlog 1178, spec PERSONAL-OFF-NEVER-SLEEPS). In 24/7 mode the day gate's `no_grant` leg wrote `asleep`
  (`written_by=grant-gate`) with a 6 h wake, and the visit door refused him. Now the loop ends own time in both modes
  with `stopped_by=personal_off` (was `personal_disarmed`) and writes no sleep: he stays awake and visitable. An
  ending loop no longer lands him in a `loop-exit` sleep (v22 retires v8 IDLE-IS-SLEEP); it wakes a sleep it wrote
  for itself, and a new loop wakes one an earlier loop left (`LOOP_OWN_SLEEP_WRITERS`: `self`, `personal-cycle`,
  `grant-gate`, `day-gate`, `loop-exit`). An operator sleep, a visit posture and the kill switch are never touched.
  A supervised settled desk ends the run without writing a sleep.
