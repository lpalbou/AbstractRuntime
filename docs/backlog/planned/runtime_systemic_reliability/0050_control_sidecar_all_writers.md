# Planned: B8 executed and RE-SCOPED — signal-only control for ALL five non-tick-thread writers

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Priority: P1 (before the fleet runs 24/7)
- Supersedes: the B8 filing from the code seat's adversary (commons c971) —
  correct diagnosis, under-scoped inventory.

## ADR status
- Governing ADRs: None
- ADR impact: extends 0046's ownership rule in `docs/architecture.md` with
  the sanctioned-writer inventory.

## Context
H4 proved the pattern: host threads APPEND signals to a sidecar; the tick
thread drains at loop boundaries; the run's driver stays the only writer.
B8 filed pause/cancel as the remaining offenders. The meta-analysis found
the class is wider.

## Current code reality (line-verified 2026-07-12)
FIVE non-tick-thread RunState writers:
1. `pause_run` (core/runtime.py:1211-1254) — load-mutate-save; via the JSON
   store's ALIASED loads it also mutates the tick thread's live object.
2. `cancel_run` (985-1018) — same shape; tick's
   `_abort_if_externally_controlled` narrows but cannot close the
   check-then-save window.
3. Cross-run MEMORY-OWNER writes: `_handle_memory_tag/note/compact/rehydrate`
   save OTHER runs (the session/global memory-owner run) from whatever
   thread ticks the current run (runtime.py:3958, 4300, 4527, 4533, 4704).
   Two runs in one session on different gateway threads race the same owner
   file; nothing rechecks before these saves.
4. `emit_event` inline resume of listener runs from the emitting run's
   thread (2851-2871) — a second full tick driver on someone else's run.
5. Host-side writers: `history_bundle.persist_workflow_snapshot`
   (history_bundle.py:433) and `active_context.rehydrate_into_context`
   (memory/active_context.py:299).

AMENDMENT (2026-07-14, 0067 durability audit — independent line-verified
re-enumeration): the inventory above is CONFIRMED complete for the aliasing
stores; `steer()` is the one compliant model citizen. Two sharpenings:
(a) the tear class only exists on ALIASING stores (JsonFileRunStore cache,
InMemoryRunStore) — `SqliteRunStore.load` returns a fresh object per call,
so on SQLite these writers produce lost-update clobbers, never
mid-serialization tears; (b) loudness delta recorded in
`storage/serialize.py`: the old `asdict` path sometimes raised
(net-growth dict resize mid-copy -> RuntimeError) where the post-0067 C
encoder tears SILENTLY — balanced insert+delete tore silently under both,
so this raises the value of the sidecar migration rather than changing its
shape. Live evidence the class bites: the 0047 issuance-counter tear
(pause mid-effect serialized the advanced counter without the result —
fixed 2026-07-14 by binding the advance to the step's own saves, pinned in
test_effect_issuance_idempotency.py) was exactly writer #1 (pause_run)
racing the tick thread through the alias.

## Problem
The "tick thread is the only writer" rule (steer docstring) is violated by
design in five places; each is a lost-update or data race under concurrency.

## What we want to do
- pause/cancel: signal-only control sidecar (the steer store shape or a
  mailbox name on 0051's primitive); tick applies at loop boundaries;
  direct-write stays ONLY for runs with no live driver (detected via 0046's
  guard), cancel keeps its terminal-recheck exception.
- memory-owner writes: route through a per-run-id lock (0046's) or an
  owner-mutation queue drained by the owner's driver.
- emit_event: listener resume under the LISTENER's 0046 lock now; a
  host-selectable signal-only delivery mode later (inline resume stays for
  single-process hosts — it is load-bearing for tests/demos).
- host-side writers: document + guard with 0046's stale-save detector.

## Scope / Non-goals
Scope: the five writers, tests per writer, doc inventory. Non-goals: no
forced async, no removal of inline emit_event resume (opt-in mode only).

## Dependencies and related tasks
0046 (the guard this composes with), 0051 (shared mailbox store), 0045
(harness pins the drain orderings).

## Expected outcomes
Every RunState write path is either the run's driver, a drained signal, or
a guarded+detected exception — enumerable in one doc table.

## Validation
Race pins per writer (pause-vs-tick, cancel-vs-tick, owner-vs-owner,
emit-vs-scheduler); full suite; gateway suite unaffected.

## Progress checklist
- [ ] Control signals for pause/cancel + drain
- [ ] Memory-owner serialization
- [ ] emit_event under listener locks
- [ ] Host writer guards
- [ ] Doc inventory table
- [ ] Fable5 adversary folded
