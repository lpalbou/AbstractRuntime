# Planned: Indexed idempotency lookup (kill the O(ledger) scan per effect step)

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Priority: P0 (the deterministic 24/7 scale cliff)

## ADR status
- Governing ADRs: None
- ADR impact: None (pure performance; semantics unchanged)

## Context
Idempotent effect replay is the runtime's crash-safety currency: before
executing an effectful step, the tick loop looks for a prior COMPLETED
record with the same idempotency key and reuses its result.

## Current code reality (line-verified 2026-07-12)
- `_find_prior_completed_result` (core/runtime.py:2481-2494) calls
  `ledger_store.list(run_id)` — the FULL ledger — on EVERY effectful step
  (call site ~1740). `resume()` performs two more full lists (~2152, ~2179).
- JSONL backend parses every line of the file per call
  (json_files.py:455-500); SQLite selects and JSON-parses every record
  (sqlite.py:685-702).
- A 24/7 resident at 100k ledger records pays a ~full-file parse per step:
  quadratic cumulative cost, charged to the tick thread — stalls that look
  like "the agent froze", and (gateway-hosted) event-loop pressure.
- Precedent for safe migration: the `ledger_heads` backfill
  (sqlite.py:219-238) added a table to existing dbs idempotently.

## Problem
Ledger size linearly taxes every future step of the same run. Long-lived
residents hit a deterministic cliff; nothing else in the design has this
shape.

## What we want to do
- SQLite: an `idempotency_key` column (or generated column) + index on
  (run_id, idempotency_key, status), point query newest-first — with an
  ALTER-if-missing backfill so existing stores upgrade in place.
- JSONL: bounded reverse-tail scan (keys recur only at the same node; a
  window of the last N hundred records suffices) PLUS a per-process per-run
  key→result cache invalidated on append.
- resume()'s two full lists route through the same lookup.

## Why
Replay stays correct at any ledger size; tick cost becomes O(1)-ish in
ledger length; the 24/7 fleet loses its scale cliff.

## Requirements
- Byte-identical replay semantics (newest matching COMPLETED record wins —
  preserve current tie behavior exactly).
- Migration: opening an old SQLite ledger upgrades it idempotently; a
  crash mid-backfill re-runs safely.
- JSONL window size documented + overridable; a miss falls back to the full
  scan (correctness over speed) with a counter increment (0054).

## Suggested implementation
One day incl. backfill; measure before/after with a synthetic 100k-record
run (the measurement lands in the completion report).

## Scope
Both backends + resume call sites + migration + benchmark evidence.

## Non-goals
- No ledger compaction/pruning (append-only posture; archival is 0058).
- No schema change to StepRecord itself.

## Dependencies and related tasks
- 0045 (harness re-runs its shapes over the indexed path), 0053 (growth
  audit cites the same cliff), completed 013 (retries/idempotency design).

## Expected outcomes
Step cost flat vs ledger size on both backends; full suite green with zero
behavior diffs.

## Validation
- A/B benchmark: per-step lookup time at 1k/10k/100k records, both backends.
- Migration test: pre-index SQLite file → open → upgraded → queries green.
- Fallback-miss counter pin (JSONL window miss → full scan → correct reuse).

## Progress checklist
- [ ] SQLite index + backfill
- [ ] JSONL tail window + cache
- [ ] resume() routed through the lookup
- [ ] Benchmarks recorded
- [ ] Fable5 adversary folded

## Guidance for the implementing agent
Do not change what "matching" means — read `_find_prior_completed_result`'s
exact predicate first and pin it before optimizing.
