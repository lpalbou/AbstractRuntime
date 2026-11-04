### Added

- **Track record and reputation of an entity or a session** (`identity/track_record.py`, ADR 0052 §5-§6,
  framework backlog 1119). `entity_track_record(home_dir, run_store, ledger_store, *, book_head)` and
  `session_track_record(session_id, run_store, ledger_store)` answer sessions, runs, successes, failures,
  cancelled, in progress and tokens (input, output, total), per task kind (`<phase or role>/<workflow_id>`,
  from ids only) and overall; `compute_reputation(record)` gives trust per task kind,
  `(successes + 1) / (successes + failures + 2)` with its support, the support-weighted overall, and the
  last 20 outcomes as evidence ("no experience yet" when there is none). Everything is computed from
  existing records; nothing is stored. Only work from outside counts: turn roots a person or a trigger
  started and automation occurrences (once each, however many attempts); sub-runs, controllers, the
  session memory run and runs another run or the system started never do. Outcomes come from the run
  status, the occurrence's `automation.completed` decision, an entity task marked `done`, and the work
  order verdicts (`done:` / `blocked:`). Feedback from people and agents (`feedback_user`,
  `feedback_agent`) is typed but has no source yet, so every answer says `basis: "completion-rate only"`.
  Each answer names its subject's `identity_ref` and, for an entity, its `identity_fingerprint`.
  See `docs/track-record.md`.
- **`metrics_many` also returns `tokens_input` and `tokens_output`** (SQLite ledger): `usage.input_tokens`
  (else `prompt_tokens`) and `output_tokens` (else `completion_tokens`) of completed `llm_call` records.
  Existing keys are unchanged.
- **`identity.life.read_work_order_verdicts(home_dir)`**: the verdicts the work lane archived in
  `work_order.done.md`, oldest first.
