### Changed

- **A placeholder title is marked and never shown as his** (`identity/title_failures.py`, `identity/chat.py`,
  `identity/visit_workflow.py`, `identity/prelude.py`; Codex review 2026-11-09 #9, interim until AbstractMemory
  stores a missing title as None). A record formed with the `untitled <kind> [<key>]` placeholder carries
  `title_missing: true` and `title_failure_reason` in its attributes; the recall shelf, the turn's memory report and
  the prelude show no title for it (`is_placeholder_title`). A visit's untitled election is logged after its
  formation, so the failure line names the record it formed (or why none formed).
