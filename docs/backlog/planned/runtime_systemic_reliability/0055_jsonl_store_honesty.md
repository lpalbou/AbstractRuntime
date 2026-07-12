# Planned: JSONL store honesty — count()/list() convergence + tier documentation

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Priority: P1 (the divergence class already busy-looped a production poller)

## ADR status
- Governing ADRs: None
- ADR impact: None (documentation states the existing tiering; no policy change)

## Context
The gateway's F1 incident (commons c1035): a stream poller busy-looped at
~4,556 full-ledger reads/sec because `count()` persistently exceeded
`len(list())` on a ledger with one corrupt line. The gateway fixed ITS
consumption; the divergence itself is structural in the runtime's JSONL
store.

## Current code reality (line-verified 2026-07-12)
- `JsonlLedgerStore.count()` counts non-empty LINES (json_files.py:502-515).
- `list()` runs concatenated-object RECOVERY per line (json_files.py:474-499)
  — one corrupt line can yield 0 or 2+ records where count() sees 1.
- So count() > or < len(list()) is reachable on any ledger that ever took a
  torn write — every consumer comparing the two inherits the F1 bug.
- `append()` neither locks nor fsyncs (448-453) — 0048's lane.
- The JSON run store's own docstring says archive/prune before ~10k files
  (json_files.py:283-290).

## Problem
Two read paths over one file disagree under corruption; the store's honest
operating tier (dev/single-user vs 24/7) is folklore.

## What we want to do
- `count()` goes through the SAME recovery decode as `list()` (or a cached
  heads/count sidecar invalidated on append) — the two can never diverge by
  construction; recovery events increment 0054's counter.
- Document the tiering plainly in docs + docstrings: JSONL/JSON-file stores
  = dev and single-user tier; SQLite = the 24/7/fleet default (WAL, indexed,
  BEGIN IMMEDIATE precedents).

## Why
Closes the F1 class at the source instead of per-consumer; sets honest
operator expectations before the fleet defaults get chosen.

## Scope / Non-goals
Scope: count() convergence + pins + tier docs. Non-goals: no JSONL format
change; no forced migration (SQLite-by-default decisions are host/factory
choices, coordinate separately).

## Dependencies and related tasks
0048 (shared recovering reader), 0054 (recovery counter), gateway F1 fix
(the consumer-side precedent).

## Expected outcomes
count() == len(list()) on every ledger, corrupt or not; the docs answer
"which store for production" in one line.

## Validation
Pin: a ledger with a torn/concatenated line → count() == len(list()) and
the recovery counter increments; benchmark count() cost before/after (it
loses its cheap line-count shortcut — measure, and cache if material).

## Progress checklist
- [ ] count() through recovery decode (or cached sidecar)
- [ ] Divergence pin
- [ ] Tier documentation
- [ ] Cost measured
- [ ] Fable5 adversary folded
