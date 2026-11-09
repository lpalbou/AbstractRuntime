### Fixed

- **A process that bumped a home once still sees a Stop by another process** (Fable review F1, P1). The bumping
  process used to skip the signal stat for that home forever after its first Stop, so a later Stop by another process
  (a CLI `hard_stop_loop`, a service twin, a second gateway) passed every fence check until the next truth read. Every
  process now pays the one stat per check (about 1-3 us); the bump still updates its own process's table at once.
