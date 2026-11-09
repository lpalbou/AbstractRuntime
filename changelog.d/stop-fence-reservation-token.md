### Fixed

- **A spawn reservation whose pid was reused no longer reads "already starting" forever** (Fable review F5). The
  `starting` liveness fold checked the pid only; it now compares the recorded start-time token exactly as `day` and
  `between` do (the spawner's `spawner_started_at` before `Popen` returned, then the child's `pid_started_at`, which
  the spawner now records with the child pid - one `ps` per spawn and per status read, never per effect).
