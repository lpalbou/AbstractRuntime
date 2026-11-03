### Fixed

- **A run store whose index cannot attribute a session is refused, never started as a plain chat**
  (`core/run_attribution.py`, framework backlog 0938). A third-party store with `list_run_index` but no `session_kind`
  in its rows (an index written before automations v1) used to report "no kinds" and let a `discussion-session:` start
  through as an ordinary chat, skipping the discussion's read-only anchor. It now raises `SessionAttributionError`
  naming the missing column; a store with no index at all names both methods it lacks. Built-in stores are unchanged.
  Upgrading a custom store: add `session_kind` and `role` to its index rows, or implement `session_kinds(session_id)`
  (see `docs/automations.md`, "Storage guarantees").
