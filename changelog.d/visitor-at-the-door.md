### Fixed

- **A visitor reaches an entity within one step of its own time** (operator 2026-11-09: a visit never fails; the
  entity is reachable within a minute or two). The own-time loop used to look for a visitor only between ticks, and a
  tick with web searches and page fetches runs for minutes, so the visit door gave up first. The loop now hands its
  session a `yield_check`: once a visit door has written the visiting posture, a reply's newly elected tools do not
  run (`[tool call not run: <tool> - a visitor arrived]`, with a `#NOTE`), no correction call, format nudge, speak-now
  or liveness call runs, the turn ends on what the entity said, the tick's look-back is skipped (the day's deferred
  look-back covers it), there is no pause before the next boundary, and the loop yields right after. A visitor who
  knocks on the day's last tick, or on a tick that elects rest or a work verdict, also gets the fast close (no inline
  look-back), and a rest elected as the visitor knocks is not slept over the visit.
- **`await_loop_quiescent(timeout_seconds=None)` waits without a limit** (root ADR-0027), and a `day` status that
  `read_loop_status` does not count as running (dead or recycled pid, stale heartbeat) is quiescent.
