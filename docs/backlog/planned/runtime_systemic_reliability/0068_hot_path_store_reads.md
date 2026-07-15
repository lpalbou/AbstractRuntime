# Planned: Hot-path store reads (SQLite control-probe + column-persisted index fields)

## Metadata
- Created: 2026-07-13
- Status: Planned
- Completed: N/A
- Priority: P1 (per-step waste on the production backend; portable)
- Area: storage/sqlite, core/runtime, storage/steer_sidecar
- Source: 2026-07-13 performance adversary (fable5) findings 4, 5, 2-corollary, 7

## ADR status
- Governing ADRs: None
- ADR impact: None (pure performance; semantics unchanged)

## Context
The cheap companions to 0047/0067 — full-document JSON parses on the
production SQLite backend where a single indexed column would do. Each is
small, portable, and independently shippable; grouped because they share the
"persist the field as a column instead of re-parsing run_json" shape.

## Current code reality (line-verified 2026-07-13; benched at ~2 MB state)
- `_abort_if_externally_controlled` (core/runtime.py:1665) runs at loop top
  AND before every save; `get_state` runs at tick entry. On SQLite each is
  `SELECT run_json` + `json.loads` of the WHOLE document + reconstruction
  (sqlite.py:356-373) — ~2.4 ms/load, ~5–7 ms/step of pure waste. The check
  needs only `status` and `vars._runtime.control.paused`. (On the JSON store
  this is ~free — mtime-validated cache — so it is a SQLite-only tax, and
  SQLite is the recommended production + entity-runtime store.)
- `list_run_index` (sqlite.py:459-497) does `json.loads` of the entire state
  PER ROW to extract `run_lifecycle_index_fields(vars)`. A 100-row
  gateway/UI page over fat resident states ≈ ~240 ms/poll.
- `_append_progress_event` computes `len(self._ledger_store.list(run.run_id))`
  PER progress callback (core/runtime.py:1582) — generated-media runs pay a
  full ledger parse per progress tick.
- `SqliteSteerSidecar.pending` (storage/steer_sidecar.py:134-157) opens a
  fresh connection + CREATE TABLE IF NOT EXISTS + 2 PRAGMAs + SELECT + close
  per call; `_drain_steer_messages` runs per step (~0.39 ms/step). Minor;
  fix only when the file is open anyway.

## Problem
The production backend re-parses multi-MB state documents for one or two
fields, per step and per index row — pure, portable waste that scales with
state size and fleet size.

## What we want to do
- Persist a `paused` column (and reuse the existing `status`) at save time,
  exactly as `wait_reason` already is (sqlite.py:283-295); add a targeted
  `probe_control(run_id) -> (status, paused)` fast path on the store
  Protocol with a full-load fallback for stores that do not offer it.
- Persist the `run_lifecycle_index_fields` as columns at save time so
  `list_run_index` reads columns, not `run_json`.
- Route `_append_progress_event`'s count to `count()` (O(1) on SQLite via
  `ledger_heads`; a no-JSON-parse line count on JSONL) or a uuid suffix.
- (Opportunistic) per-thread cached connection for the steer sidecar when
  that file is touched.

## Why
Removes ~5–7 ms/step of pure waste on the production backend and turns a
~240 ms index page into a column scan — the kind of flat win that compounds
across a 24/7 fleet, with zero semantic change and full portability.

## Requirements
- Column additions ship with an ALTER-if-missing backfill so existing
  SQLite stores upgrade in place (the `ledger_heads` backfill,
  sqlite.py:219-238, is the precedent).
- `probe_control` is an optional Protocol method; the runtime falls back to
  a full load for stores without it (never a hard requirement on custom
  stores).
- Column values stay derived from the SAME source as today
  (`run_lifecycle_index_fields`, `_runtime.control.paused`) — no new truth,
  just a faster read of the existing truth.

## Suggested implementation
S–M total. `paused` column + probe_control (S), index-field columns +
backfill (S–M), progress count() route (S), steer connection cache (S,
opportunistic).

## Scope
SQLite store columns + backfill, the runtime probe fast path, the progress
count route, optionally the steer sidecar connection cache.

## Non-goals
- No change to what `paused` / lifecycle fields MEAN (only where they're read
  from).
- No JSON-store change (already cached).

## Dependencies and related tasks
- Sibling of 0047 (indexed idempotency) and 0067 (durable-write) — same
  "column instead of full-parse" family; can land independently but shares
  the SQLite migration discipline.

## Expected outcomes
Control probe and index page read columns; progress events cost O(1); full
suite green with a migration test for the new columns.

## Validation
- A/B: per-step control-probe time + 100-row index-page time before/after.
- Migration test: pre-column SQLite file → open → columns backfilled →
  queries green.
- Progress: a generated-media run's per-tick cost flat vs ledger size.

## Progress checklist
- [x] `paused` column + probe_control fast path + fallback — shipped
  2026-07-14; pause-flag shape single-sourced in `core.vars.is_paused_vars`
  (runtime + store import it; no second copy). Probe answers None on
  pre-migration NULL rows; runtime full-loads only when CONTROLLED.
- [x] lifecycle-field columns + ALTER-if-missing backfill — shipped;
  duplicate-column race tolerated; backfill applies the same sanitization
  as the write path (Python loop; torn rows match the readers' {} fallback).
  FINDING: `run_json` had to be DROPPED from the index page query — the
  document column FETCH dominated the cost even before json.loads
  (~1.1s/100-row page either way with it riding the SELECT). Pre-migration
  rows fetch their document individually. Bare `idx_runs_updated` added for
  unfiltered ORDER BY pages.
- [x] _append_progress_event count() route — shipped as a uuid suffix
  (strictly cheaper than count() on every backend; the key was a pure
  uniquifier, no consumer reads its shape — grep-verified).
- [x] (opt) steer sidecar per-thread connection — shipped WITH the F3
  purge-healing preserved: stat() per call detects the deleted file and
  reopens; failed writes rollback-or-drop the thread connection.
- [x] Migration + A/B benches (control probe 1.59ms → 0.004ms ≈ 360x at
  ~1.9MB states; 100-row index page 171.7ms → 26.5ms ≈ 6x — remaining cost
  is SQLite row-overflow traversal, the twin columns sit after run_json in
  the physical row; a table rebuild was judged not worth it);
  `tests/test_hot_path_store_reads.py`; full suite green.
- [x] fable5 adversary FOLDED (2026-07-14, four P1 fixes): P1-1 backfill
  UPDATEs guarded `AND paused IS NULL` (mutable-row TOCTOU could stamp a
  stale not-paused over a live pause and the tick then DESTROYED it — the
  one finding that beat the old full-load path); P1-2 sidecar purge
  healing by file identity (st_dev, st_ino), not existence (first-thread
  heal recreated the file and every other thread silently wrote the dead
  inode); P1-3 backfill batched + per-batch commits + O(1) probe via
  `idx_runs_unbackfilled` (fetchall was ~200GB at the design target and
  the single commit made OOM an open-crash-loop); P1-4 explicit
  `probe_control` passthrough on OffloadingRunStore (the gateway's
  production wiring hid the fast path entirely; the test double's
  `__getattr__` forwarding had hidden the seam). P2-1 honest-comment fix:
  the lifecycle column read still walks row overflow (~30ms/100 rows at
  2MB) — the win is the eliminated json.loads. Cleared: WAL freshness,
  controlled-window width, writer census, N+1 bound, uuid keys, enum
  comparisons. Suite 1268 green.

## Guidance for the implementing agent
These are reads of EXISTING truth made cheaper — never introduce a second
source. Keep the probe fast path optional on the store Protocol so custom
stores are unaffected. Backfill must be crash-safe (re-runnable).
