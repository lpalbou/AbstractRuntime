# Planned: Bounded run-vars growth for 24/7 residents

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Priority: P1 (before the fleet runs 24/7)

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
A resident agent lives in ONE run for days. Everything that accretes inside
`run.vars` is serialized on EVERY save, and `tick()` saves 2-6× per step
(runtime.py:1727, 1858, 1866, 1879, 1899, 1908). The JSON file store
rewrites the whole state per save (json_files.py:148); SQLite serializes
whole vars per save (sqlite.py:297).

## Current code reality (line-verified 2026-07-12)
Unbounded or weakly-bounded growth surfaces:
- Node traces embed FULL effect payloads incl. LLM message arrays
  (runtime.py:713-723), 100 entries per node (750) — bounded per node,
  unbounded in bytes.
- `evidence_warnings` unbounded (1948-1953).
- `_runtime.inbox` unbounded if the workflow never drains.
- `context.messages` / visit sheets have no stated ceiling (visit history
  is trimmed to 2×history_turns — verified OK; the sheet is not).
- events_inbox is capped at 500 envelopes (gateway-side today) but envelope
  SIZE is unbounded... bridge clamps bodies at 32KB (gateway F7) — runtime
  gets the cap with 0051.
- Vars offloading exists but is deliberately TERMINAL-ONLY
  (offloading.py:182-188); `offload_large_values` is reusable.

## Problem
Tick latency and save cost creep with uptime; the end state is the
1.5GB-RSS-shaped incident inside a single run's vars, or event-loop
starvation host-side as saves slow.

## What we want to do
- Byte-cap node-trace entries: above a threshold, store digest + artifact
  ref (reuse `offload_large_values`) instead of the full payload.
- Cap `_runtime.inbox` and warning lists (drop-oldest + counter, the
  events_inbox precedent).
- A vars-size gauge computed at save (cheap: len of the serialized bytes
  already produced) with a `_limits`-style threshold warning + 0054
  counter.
- MEASURE first: a synthetic long-run benchmark (10k steps) recording tick
  time + save bytes vs step count, both backends — the numbers drive the
  thresholds and land in the completion report.

## Scope / Non-goals
Scope: the three cap classes + gauge + benchmark + docs ("what grows,
what's capped, what offloads"). Non-goals: no auto-compaction of
context.messages (MEMORY_COMPACT is the workflow's explicit tool); no
terminal-offloading change.

## Dependencies and related tasks
0047 (the other per-step cost), 0054 (gauge surface), completed
offloading work, backlog 017 (limit warnings).

## Expected outcomes
A resident's tick cost is flat over days; approaching a ceiling is visible
before it hurts.

## Validation
Benchmark before/after; cap pins (oldest dropped + counted, never silent);
gauge threshold pin.

## Progress checklist
- [ ] Benchmark harness + baseline numbers
- [ ] Node-trace byte cap via artifact offload
- [ ] Inbox/warning caps
- [ ] Vars-size gauge + threshold
- [ ] Fable5 adversary folded
