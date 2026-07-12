# Planned: Per-run driver exclusion + concurrent-driver detection

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Priority: P0 (correctness-grade; both meta-adversaries' top finding)

## ADR status
- Governing ADRs: None
- ADR impact: the run-ownership rule ("a run has one driving thread; all
  cross-thread influence is append-to-sidecar drained at tick boundaries;
  terminal-only transitions are the sanctioned exception because tick
  rechecks them") must land in `docs/architecture.md` alongside 0045's
  invariant.

## Context
The steer sidecar (H4) established "the tick thread is the ONLY writer of
run state" — as a docstring. Nothing enforces it, and concurrent drivers
exist BY DESIGN: the `Scheduler` poll thread, host `emit_event` fan-out
(which calls `self.resume(...)` on listener runs from the EMITTING run's
thread, runtime.py:2851-2871), the gateway runner, `MultiStoreScheduler`.

## Current code reality (line-verified 2026-07-12)
- Zero locks in `core/runtime.py` (the module never imports `threading`).
- `tick()` and `resume()` can be entered concurrently for the SAME run from
  different threads.
- `JsonFileRunStore.load()` returns ALIASED objects (json_files.py:36-52
  documents single-writer as prose) — two drivers mutate the same RunState
  dict: interleaved saves, doubled LLM/tool spend (idempotency only
  collapses after a COMPLETED record lands), potential json.dump on a dict
  mutating mid-serialization.
- Concrete race: an EVENT-with-deadline run passes its deadline while the
  event arrives — scheduler due-scan ticks it while Scheduler.emit_event
  resumes it.
- B8's writers (pause_run 1211-1254, cancel_run 985-1018) are PART of this
  class; 0050 handles the writer inventory — THIS item is the kernel guard.

## Problem
Two drivers on one run is silent corruption: no error, no log, wrong money
spent, torn state. It is the framework's one unbounded silent-corruption
class.

## What we want to do
1. A per-run_id in-process lock (WeakValueDictionary of locks) acquired by
   `tick()` and `resume()`; a second concurrent entrant either blocks
   (bounded) or refuses loudly ("run <id> is being driven by another
   thread") — refusal is the default; hosts opt into blocking.
2. A cross-process detector: `save()` warns + increments a counter (0054)
   when the persisted `updated_at` is newer than the loaded snapshot's
   (last-writer-wins just happened).
3. The ownership rule written in `docs/architecture.md`.

## Why
Closes the two-drivers-one-run class at the kernel instead of per-verb;
makes the H4 docstring true by construction.

## Requirements
- Lock scope covers the whole tick/resume body incl. their saves.
- emit_event's inline resumes acquire the same per-run lock (deadlock-safe:
  emitting run's lock is NOT held while resuming listeners — verify).
- No measurable hot-path cost when uncontended.

## Suggested implementation
~50 lines: module-level `_run_locks: WeakValueDictionary`, a
`_driver_guard(run_id)` context manager, `updated_at` compare in the file
stores' save (JSON + SQLite).

## Scope
In-process exclusion + cross-process detection + doc rule + race tests.

## Non-goals
- No distributed/cross-process locking (one-writer-per-store + the
  directory lease is the ruled shape).
- No deep-copy or frozen store views (hot-path tax; explicitly rejected).

## Dependencies and related tasks
- 0045 (its concurrency mode exercises this guard), 0050 (writer
  inventory), `storage/lease.py` (the cross-process half that exists).

## Expected outcomes
A second driver on a live run is impossible in-process and DETECTED
cross-process; the pause-vs-tick and emit-vs-scheduler races have pins.

## Validation
- New race tests: tick-vs-resume, tick-vs-emit_event-listener-resume,
  scheduler-deadline-vs-event (currently ZERO concurrency stress tests
  exist beyond the steer 3-writer test).
- Full suite green; gateway suite unaffected (their runner is single-loop).

## Progress checklist
- [ ] Per-run guard in tick()/resume()
- [ ] emit_event listener resumes under the listener's lock
- [ ] updated_at stale-save detector + counter
- [ ] Ownership rule in docs/architecture.md
- [ ] Race pins
- [ ] Fable5 adversary folded

## Guidance for the implementing agent
Audit lock-ordering around emit_event fan-out FIRST (emitter lock vs
listener lock); the refusal message must name the run id and the other
driver's entry point if cheaply known.
