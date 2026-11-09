### Fixed

- **A replayed answer is the stored bytes, whitespace included** (`session_history.py`, `history_bundle.py`; Codex
  review 2026-11-09 #10). Session turns stripped the answer before replay, so an entity re-reading its own session
  got "indented answer" for "  indented answer\n\n" (not what it said; prefix caches missed). The answer now replays
  as stored; `.strip()` only decides whether there is one.
