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

## Observability-program tie-in (2026-07-13, observer redesign c1582/c1592)
The runtime seat's answer to observer's "what should be observable" ask
(c1606) makes this item the EMIT half of a shared contract: the runtime
EMITS, observer RENDERS. Two concrete surfaces the program named that this
item should serve:
- **RuntimeHealth counters** (this item): tick throughput/durations, steer
  drain retries, store self-heals (JSONL recoveries, sidecar purges), replay
  collapses — the "is the runtime healthy right now" tile. These are the
  counters listed under Current code reality; the program is the consumer
  that justifies pulling this forward.
- **Stalled-waits index** (small addition, name it here): a pure read over
  non-terminal runs grouped by `wait_reason` + age (the last wait record
  carries the reason). The runtime's #1 silent-stall class is a run parked on
  WAIT_EVENT / WAIT_UNTIL / subworkflow / tool-approval that nobody resumes;
  observer's fleet board renders it, but the runtime should offer the query
  (a `list_stalled_waits()`-style read, or fold into the existing
  `list_due_wait_until` shape). Cheap, portable, high operator value.
The seq-ordered hash-chained ledger is already the "what happened at time T"
source — time-travel needs NO new per-view history; this item just adds the
LIVE counters the ledger cannot express (durations, retries, heals).

## Dependencies and related tasks
Feeds from 0045/0046/0048/0052/0053; planned 017; gateway ops surface;
observer observability redesign (c1582/c1592) is the primary consumer.

## Expected outcomes
`runtime.health()` answers drain failures / recoveries / replays / slow
ticks numerically; the next incident's first diagnostic is a counter read,
not a process sample.

## Validation
Pins: each counter increments from its triggering condition; getter is
cheap and thread-safe (single lock or atomics); gateway serving smoke
(coordinated).

## Progress checklist
- [x] RuntimeHealth object + getter — core/health.py: thread-safe counters,
  coarse tick buckets, gauges (auto `_max`), 16-entry last-errors ring;
  `runtime.health()` returns one JSON-safe isolated snapshot; root export.
- [x] Counters wired at existing sites — effect steps/retries/failures,
  waits entered, resumes, replay reuses, absorbed failures, steer drains +
  delivered + drain errors (ring), inbox drops, vars threshold crossings;
  every tick() timed via a thin wrapper (`_tick_impl` holds the body).
- [x] Factory wiring — always-on in Runtime.__init__ (simpler than opt-in;
  an int += 1 under one lock is free).
- [x] Stalled-waits index — `runtime.list_stalled_waits(older_than_s)`:
  WAITING runs by reason/key/age/paused, oldest first; [] on non-queryable
  stores. (The observer-program surface named in this item.)
- [ ] Gateway coordination (serve `health()` + stalled waits on the ops
  surface — their lane; posted with the ship receipt)
- [x] Fable5 adversary FOLDED (2026-07-14, shared pass with 0053; no P0):
  P1-1 `list_stalled_waits` newest-first candidate window silently hid the
  OLDEST waits once WAITING runs exceeded it ([] on the exact fleet board
  this exists for) — candidates now come from list_run_index(oldest_first=
  True) (new protocol kwarg, all three stores), only aged winners load
  documents (P2-4 cost fix riding the same query), unknown-age rows sort
  FIRST; P3 effect failures + raising ticks now reach the last-errors ring
  (it had one producer); the warn-once caplog pin asserts for real.
  Cleared: tick-wrapper exception propagation, 4-thread counter exactness,
  snapshot isolation. Suite 1285 green.
