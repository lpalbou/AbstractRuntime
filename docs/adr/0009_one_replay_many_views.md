# ADR 0009: One replay of the ledger, many views

Status: Accepted.
Date: 2026-10-27

## Context

The run ledger is the record of a run. Every client that shows a run needs the same few facts
about it: its status, the approval or question it is waiting on, the replies, the tool calls and
the steps. Until this decision, each client worked those facts out for itself from raw ledger
records, run snapshots and its own memory. Panel-chat, the AbstractCode terminal client, the
desktop Assistant, the iPhone and watch apps, the Observer, AbstractFlow and both gateway
consoles each had their own reconstruction. The inventory, with file:line anchors, is in
[backlog 0853](../backlog/planned/0853_replay_view.md). The reconstructions disagreed on the same
run:

- 2026-09-29: a delegated child's tool approval showed as "Approval needed" on one client and as
  "Running a tool" on another. A delegation wait displaced the person-facing wait, and a `$slim`
  dedup pointer was left unresolved.
- 2026-10-04: a `wait_until` timer with no resume record left 182 of 3,278 runs showing "Waiting
  for you" after they had completed.
- 2026-10-11: one Agent node execution showed as two steps, because nothing in the records names
  the execution.

The operator's ruling (framework root backlog 0993,
`abstractframework/docs/backlog/planned/0993_one_ledger_replay_many_views.md`): "there should be
only one unified way to replay the ledger, on top of which we could have different views", and
"logic should not be in the thin client". Each client decides what to show and how to forward
the requests of the users and entities it serves.

Durability had a gap of its own. The SQLite ledger had a per-run `seq` and the JSONL ledger had
none. The gateway's cursor was a list position, and the file-to-SQLite migration renumbered
records (gateway backlog 0082).

## Decision

1. **One reducer, in the runtime.** `abstractruntime.replay` is the only code that turns ledger
   records into run state for display:
   - `replay(records_by_run, runs_by_id)` folds a run tree's records and durable snapshots into a
     `ReplayView`.
   - `ReplayView.apply(records)` folds newer records into it and returns the delta.
   - `replay(all)` is `apply` over every record, so it equals folding the same records in chunks.

   The reducer is pure and deterministic: no I/O and no clock, and its input is records plus
   snapshots. Its JSON is versioned (`replay_view_v1`). A change in meaning is a new version,
   never a silent change.
2. **The reducer owns these rules.** Each rule below was a divergence between clients:
   - The run tree comes from delegation waits (`subworkflow:<id>`), `start_subworkflow` results
     and the snapshots' `parent_run_id`. When the two disagree, the ledger's link wins.
   - A wait is over when its run records a `resume` for its key OR any later step of the same run
     exists. Observability events are not steps.
   - The person-facing interaction is the newest open approval, question or prompted event
     anywhere in the tree. A delegation, a timer or a pause never is one.
   - Status precedence: a terminal snapshot, then the ledger's terminal record, then an open wait,
     then running.
   - `$slim` pointers are resolved against the STARTED record. A pointer that cannot be resolved
     is dropped and named in `warnings`.
   - Executions follow AbstractFlow's `executionIdFor` rules.
   - Each run names the workflow it executes (`bundle_id@version:flow_id`; the root's flow is its
     entrypoint). Workflows are the framework's native unit.
3. **The durable cursor is the per-run ledger `seq`.**
   - Every ledger store numbers records 1, 2, 3, … per run. JSONL persists the number on append,
     allocated under an advisory file lock. A line written before the field existed is numbered
     by its position.
   - Every store answers `list_after(run_id, after, limit)`.
   - `SqliteLedgerStore.import_records` keeps `seq` across a migration.
   - A view's cursor is the seq of each run in its tree. Its token `<run id prefix>:<seq>.…` is
     the view's ETag and the SSE event id.
4. **Hosts serve the view, and clients render it.**
   - AbstractGateway serves `GET /runs/{id}/view` (ETag, 304, `?at=<token>`) and
     `GET /runs/{id}/view/stream` (one `view`, then `delta` events, `Last-Event-ID`). There is
     one cached view per run tree, folded once per appended record and fanned out to every
     subscriber, with no timed re-parse.
   - Clients act through the host's commands. The view's `interaction.command` is the resume to
     send. Clients hold no run state that the server could replay.
   - A client may keep a local view only as a cache of these deltas (`apply_delta`). It never
     builds one from raw records.

## Enforcement

- A client or host change that derives status, waits, tool rows or messages from raw ledger
  records, outside `abstractruntime.replay`, conflicts with this ADR. The fix is a reducer change
  with a fixture, or a view field.
- A reducer change that alters an existing fixture's expected view needs one of two things: the
  fixture's diff in the PR and its reason, or a new `replay_view_vN` when the meaning of a field
  changes.
- A new ledger store implements `list_after`, and its records carry `seq`. A migration between
  stores preserves `seq`.

## Validation

- `tests/test_replay_conformance.py`: the conformance fixtures under `tests/fixtures/replay/`
  (redacted real ledgers and synthetic cases), each with its expected view and hand-written
  semantic checks.
- `tests/test_replay_invariants.py`: replay equals fold over any chunking and interleaving,
  idempotence, delta round trips, and header-safe cursor tokens.
- `tests/test_ledger_cursor_invariants.py` and `tests/test_jsonl_ledger_seq_cross_process.py`:
  the invariants of gateway backlog 0082 at store level (completeness, `list_after` equals a
  replay from the same cursor, cursor stability across reopen and migration, recovery), and dense
  seqs under concurrent processes.
- AbstractGateway `tests/test_gateway_replay_view.py`, `tests/test_migrate_preserves_ledger_seq.py`
  and the principal isolation matrix: the routes through the real app.
- Every client that renders runs runs the conformance fixtures in its CI once it migrates
  (backlog 0853).
