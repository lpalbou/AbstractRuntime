# Planned: RuntimeHealth — counters, not folklore

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Priority: P1 (before the fleet runs 24/7)
- Absorbs: the observability half of planned 017 (limit warnings item stays
  for its prompt-surface half; cross-reference on completion).

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The runtime's self-knowledge is `logger.warning` at ~25 sites plus
per-feature stats objects nothing aggregates (`SchedulerStats`
scheduler.py:46, `MultiSweepStats` multi_store.py:216). The H7c starvation
was diagnosed by macOS `sample` on a live pid; the healed sidecar purge and
JSONL recoveries are visible only in logs after the fact.

## Current code reality (line-verified 2026-07-12)
Countable-but-uncounted events: tick duration; steer drains + failures
(runtime.py:1129/1209); idempotent replay collapses (`reused_prior_result`
— visible only inside node traces); JSONL corrupt-line recoveries
(json_files.py:494); sidecar heals; stale-save detections (0046);
loop-guard trips (0046); event-loop tripwire hits (0052); vars-size
threshold crossings (0053); terminal-path ledger append failures (0045).

## Problem
"Is the runtime healthy?" is answered by reading logs or sampling
processes. A 24/7 fleet needs a pulse, and the gateway needs something to
serve.

## What we want to do
One `RuntimeHealth` object per Runtime instance: a dict of monotonic
counters + a small last-error ring + coarse tick-duration buckets,
incremented AT THE EXISTING warning sites (no new hot-path work beyond an
int += 1), exposed via one getter (`runtime.health()`); factories wire it
by default. Gateway serves it on its existing health/ops surface (their
lane, coordinate the shape once).

## Why
Turns every degradation this track makes loud into something a dashboard
can watch; the counters-not-log-spam lesson (the temperature-warning
incident) applied to the runtime itself.

## Scope / Non-goals
Scope: the object, ~12 counters, the getter, wiring at existing sites,
one docs section. Non-goals: no metrics framework/prometheus dependency
(a dict; hosts export however they like); no per-call #FALLBACK log spam on
hot paths; no persistence of counters (process-lifetime is honest).

## Dependencies and related tasks
Feeds from 0045/0046/0048/0052/0053; planned 017; gateway ops surface.

## Expected outcomes
`runtime.health()` answers drain failures / recoveries / replays / slow
ticks numerically; the next incident's first diagnostic is a counter read,
not a process sample.

## Validation
Pins: each counter increments from its triggering condition; getter is
cheap and thread-safe (single lock or atomics); gateway serving smoke
(coordinated).

## Progress checklist
- [ ] RuntimeHealth object + getter
- [ ] Counters wired at existing sites
- [ ] Factory wiring
- [ ] Gateway coordination
- [ ] Fable5 adversary folded
