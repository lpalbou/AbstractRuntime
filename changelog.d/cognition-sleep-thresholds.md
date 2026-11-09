### Added

- **Sleep thresholds in the cognition profile** (`identity/cognition_profile.py`; CONTRACT v3 D2). `sleep:
  {after_hours, after_events}` says when an IDLE entity goes to sleep: hours since the last completed maintenance
  pass (1..720) and new memories formed since it (at least 1); defaults 24 and 500. `read_cognition_profile` serves
  both with their sources (defaults when unset), `sleep_thresholds(home)` reads them, and a write outside the schema
  is refused with its field (`sleep.after_hours` / `sleep.after_events`) and a sentence. The gateway's idle sweep
  consumes them.
