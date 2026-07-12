# Planned: Generalized crash-replay harness + ONE written crash-ordering invariant

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Priority: P0 (the keystone — do FIRST; every other track item is a consumer)

## ADR status
- Governing ADRs: None (repo records policy in `docs/architecture.md`)
- ADR impact: the crash-ordering invariant this item writes MUST land as a
  durable section in `docs/architecture.md` (which today contains zero
  occurrences of "thread", "concurrency", or "single-writer").

## Context
Crash-window safety in the runtime is per-feature folklore. The steer-sidecar
drain got a named 5-step ordering with a redelivery test only because its
build adversary demanded it (`core/runtime.py` `_drain_steer_messages`
docstring; `tests/test_steer_sidecar.py::test_redelivery_after_ack_crash...`).
The act-only resume path has a hand-rolled kill test
(`tests/test_act_only_resume_criterion7.py` — single-step tick, then fresh
process). Media-artifact and workspace-policy persistence each re-invent a
restart test. Nothing covers emit_event fan-out, wait resolution, or
subworkflow completion propagation with the same depth.

## Current code reality (line-verified 2026-07-12, commit 5f81ab7)
- The effect path is well-ordered: STARTED record → execute →
  COMPLETED/WAITING/FAILED record (`_execute_effect_with_retry`,
  `core/runtime.py:2517-2561`), then the run save in `tick()`; replay reuses
  results via `_find_prior_completed_result` (runtime.py:1740, 2481).
- BUT four different orderings exist, two silent-on-failure:
  1. COMPLETION inverts: run saved COMPLETED at runtime.py:~1707, terminal
     ledger record appended AFTER (~1713). Crash between → a terminal run
     whose ledger never terminates — exactly the "poller busy-loops on a
     never-converging ledger" incident shape, produced runtime-side.
  2. RESUME inverts the other way: synthetic tool-completion record (~2200)
     and durable resume record (~2282) append BEFORE the resumed state saves
     (~2333) — and both appends sit in `except Exception: pass` (2201,
     2283), so a failed append is silent. Crash between → ledger claims a
     step the durable state denies.
  3. Terminal-wait resume saves first (~2307), appends after (2312-2327),
     inside another try/pass.
  4. emit_event fan-out resumes N listeners one at a time; the emitter's own
     COMPLETED record lands after the loop (runtime.py:2851-2871). Crash
     mid-loop converges only by accident of the wait-state machine.
- Scheduler UNTIL wakes are crash-safe by re-derivation (runtime.py:1633-1656).
- ~130 `except Exception: pass` sites exist; the resume-path ledger appends
  are the load-bearing silent ones.

## Problem
Every new durable feature re-argues crash safety from scratch, reviewers
must re-derive the windows by hand, and two real inversions ship today. A
claim like "replay converges" is not checkable by machine anywhere.

## What we want to do
1. Write ONE crash-ordering invariant into `docs/architecture.md`:
   *state save is the commit point; ledger records for a transition append
   BEFORE the save that makes them true; terminal-path append failures are
   never silent (counter + warning at minimum).*
2. Fix the two inversions (completion append-then-save; resume paths
   conform; surface append failures on terminal paths).
3. Promote criterion-7's pattern into `tests/harness.py`: a parameterized
   kill-and-replay driver — run a workflow on real stores (SQLite + JSON)
   recording every persistence point; for each injection point k, kill/stop,
   restart from stores, assert (a) terminal output equivalence, (b) ledger
   convergence (terminal record present exactly once), (c) no doubled side
   effects (idempotency reuse counted), (d) optional second-driver
   concurrency mode (pairs with 0046).
4. Run the harness over the core workflow shapes: plain effect chain, tool
   approval wait/resume, emit_event fan-out, WAIT_UNTIL wake, subworkflow
   completion, steer drain, and the entity visit workflow.

## Why
This converts every crash-window and concurrency claim in the codebase from
folklore into a checkable property, and it is what keeps items 0046-0055
honest as they land. Both meta-adversaries independently named it the single
highest-leverage investment.

## Requirements
- The invariant text lands in `docs/architecture.md` §Reliability.
- The two inversions are fixed with regression pins.
- The harness is reusable by any test file (no copy-paste per feature) and
  covers at least the seven workflow shapes above.
- Append failures on terminal paths increment a counter (0054's surface) —
  never `pass` silently.

## Suggested implementation
- Injection points via a store-wrapper that raises/kills at the Nth
  persistence call (save/append), parameterized by pytest.
- "Kill" = in-process store teardown + fresh Runtime over the same files
  (true SIGKILL subprocess mode optional, criterion-7 shows the shape).
- Keep the harness synchronous and deterministic; no sleeps.

## Scope
Harness + invariant doc + the two ordering fixes + seven shape suites.

## Non-goals
- No async kernel rewrite. No new ledger format. No property-based fuzzing
  framework (hypothesis) in v1 — deterministic injection points first.

## Dependencies and related tasks
- `0044_meta_analysis_record.md` (evidence), 0046 (concurrency mode
  consumer), 0048 (durable-io — fsync decisions interact with windows).

## Expected outcomes
A new effect/feature gets crash coverage by ADDING A SHAPE, not writing a
harness; the completion/resume orderings match the written invariant.

## Validation
- New: `tests/harness.py` + `tests/test_crash_replay_shapes.py` (all seven
  shapes × injection points, SQLite + JSON stores).
- Full suite green; the two inversion fixes carry named pins.
- `docs/architecture.md` diff includes the invariant.

## Progress checklist
- [ ] Invariant written in docs/architecture.md
- [ ] Completion ordering fixed + pinned
- [ ] Resume orderings conformed + append failures surfaced
- [ ] Harness with injection-point store wrapper
- [ ] Seven shapes covered on both store backends
- [ ] Fable5 adversary on the built harness + fixes, findings folded

## Guidance for the implementing agent
Read the four ordering sites before touching anything; the tick loop's
linearity is the asset — conform orderings, do not restructure the loop.
