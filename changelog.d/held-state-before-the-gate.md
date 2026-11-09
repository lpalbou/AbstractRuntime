### Fixed

- **A sleep handed back by a visit always resumes; the loop never spins over a held state** (`identity/life.py`;
  Codex review 2026-11-09 #1). With nothing pulling, the gate chose the personal rest; the rest returned at once
  because he was asleep (a visit's close had handed his own sleep back) and the loop repeated forever without
  resuming the pass. A held state (asleep, the visiting posture, paused) now goes straight to the state gate, before
  any leg is chosen: his own sleep resumes its maintenance and wakes, an operator sleep, a visit or a pause idles.
