### Fixed

- **A failed signal write never hides the Stop** (Fable review F9). The bump wrote `.lifecycle_generation` before
  publishing the new generation, so an OSError there (a full disk) left phases.yaml bumped while this process's
  fences and every other process's stat still read the old generation, and `hard_stop_loop` reported "not bumped".
  The generation is published first now; a failed write removes the stale signal (a process that saw it reads
  "vanished", i.e. moved) and raises `LifecycleSignalUnwritten` with the generation the Stop moved to, which
  `hard_stop_loop` reports with its note.
