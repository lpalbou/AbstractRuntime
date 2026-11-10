### Added

- **`sleep.after_fragmented_share` in the cognition profile** (operator 2026-11-10; float, default 0.15,
  0 < share <= 1). The idle sweep also puts an entity to sleep when more than this share of the home's live memories
  are fragmented: placed in a topic only provisionally, without a vector, or with a title or summary still pending
  (memory's sleep-cadence read counts them). It is validated with a sentence and its field
  (`validate_sleep_after_fragmented_share`, `sleep.after_fragmented_share`), read with its source like the other two
  thresholds, and returned by `sleep_thresholds`. A malformed value in the file reads as its default with a warning.
  The sweep that acts on it is the gateway's.
