# Planned: Sync-store event-loop tripwire (make the contract self-enforcing)

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Priority: P1 (three production incidents of this exact class already)

## ADR status
- Governing ADRs: None
- ADR impact: None (the sync-core ruling stands; this enforces it at the
  boundary instead of by review).

## Context
The runtime's stores are synchronous BY DESIGN (zero `asyncio` in the
package — the sync one-writer tick is why crash windows are analyzable).
Async hosts must `to_thread` them. Three live incidents came from
forgetting: the entity replay/stream starvation (c975, fixed), the one-shot
read routes (H7c, fixed), and the class was found once more in batch
fanout — each diagnosed by sampling a live process.

## Current code reality
Nothing detects the violation. The contract lives in gateway code review
memory ("H7b discipline") and greppable comments.

## Problem
The fourth incident is a matter of time: any new async host (continuum's
console, future servers) can call a sync store on its event loop and
starve every request for the duration of a big read.

## What we want to do
- In the ~5 hot store methods (or one wrapper the factories apply):
  `try: asyncio.get_running_loop()` → warn-ONCE per process + a 0054
  counter ("sync store called on a running event loop: <store>.<method>").
- Optionally ship `AsyncRunStoreFacade` / `AsyncLedgerStoreFacade` (pure
  `asyncio.to_thread` delegation) so hosts have a paved path.

## Why
~20 lines that convert a review rule into a structural tripwire; a lint
cannot see dynamic call paths; a full async contract is a rewrite (ruled
out).

## Scope / Non-goals
Scope: tripwire + counter + optional facades + docs note. Non-goals: no
async core, no behavior change when tripped (warn, never raise — a serving
process must not die because it was diagnosed).

## Dependencies and related tasks
0054 (the counter surface), gateway's H7b/H7c fixes (the incident record).

## Expected outcomes
The next on-loop store call is visible in logs + counters the same day it
ships, not after a starvation incident.

## Validation
Pin: calling a store method inside `asyncio.run(...)` fires the warning
exactly once and increments the counter; calling it off-loop fires nothing.
`get_running_loop` check cost measured (<1µs expected — record it).

## Progress checklist
- [ ] Tripwire in hot paths
- [ ] Warn-once + counter
- [ ] Optional async facades
- [ ] Docs note
- [ ] Fable5 adversary folded
