# Planned: Runtime-owned durable event mailbox (unify the three message lanes)

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Priority: P1 (before the fleet runs 24/7)
- Cross-repo: gateway pre-agreed the move ("carry receiver counts +
  event_id dedup if events_inbox moves runtime-side", commons c1044).

## ADR status
- Governing ADRs: None
- ADR impact: None (the hooks thin-layer ruling stands; this is named
  mailboxes over an existing primitive, not a new bus).

## Context
Three durable message lanes reach a run today, with three crash
disciplines: (1) events_inbox — run-var mailbox whose DURABLE APPEND +
event_id dedup + receiver counts live ONLY in the gateway runner (zero
occurrences of `events_inbox` in this repo); (2) the steer sidecar —
runtime store, hardened ordering, watermark dedup; (3) the command inbox —
`storage/commands.py`, append-only + command_id idempotency + cursor.

## Current code reality (line-verified 2026-07-12)
- `_handle_emit_event` resumes only WAITING listeners (runtime.py:2782-2827);
  busy listeners DROP events on any host that is not the gateway.
- Each emit calls `list_runs(limit=10_000)` (2785) — a full directory scan
  per emit on the JSON store.
- The steer sidecar (storage/steer_sidecar.py) already proves the store
  shape: append / pending / ack, per-run monotonic seq, watermark dedup,
  BEGIN IMMEDIATE cross-process safety, purge healing.

## Problem
Resident agents silently lose wakeups outside the gateway; the same
append+cursor+dedup pattern exists three times; the gateway carries runtime
semantics it explicitly wants to hand back.

## What we want to do
- Generalize the steer sidecar store into a NAMED-MAILBOX primitive:
  mailbox = (run_id, name); steer becomes mailbox "steer" (compat shim).
- `emit_event` with `durable: true` appends the envelope (per-run monotonic
  seq, 500-cap drop-oldest + counter, event_id dedup) to declared-mailbox
  runs INSIDE the runtime handler, and returns receiver counts
  {resumed, appended} — the gateway deletes its copy.
- Replace the per-emit list_runs scan with a declared-mailbox index (or
  QueryableRunStore filter).
- Semantics stay distinct per lane (steer/command/event are different
  CONTRACTS); the STORAGE + crash ordering become one tested thing.

## Scope / Non-goals
Scope: the primitive, emit_event durable path, receiver counts, dedup,
gateway handoff coordination, compat for existing steer data. Non-goals: no
new eventing abstraction; no change to command-inbox semantics (it stays a
separate contract on the shared shape only if it fits naturally — do not
force it).

## Dependencies and related tasks
0045 (ordering pins), 0047 (index precedent), gateway's bridge (consumes
receiver counts), the event-inbox resident flow (abstractflow example).

## Expected outcomes
One durable-messaging module with one crash discipline; resident wakeups
never silently lost on any host; gateway's copy deleted (their suite still
green).

## Validation
Redelivery/dedup/cap pins on the primitive; emit-to-busy-run appends;
emit-to-parked-run resumes; receiver counts pinned; cross-repo: gateway
swaps and their 14 bridge pins stay green.

## Progress checklist
- [ ] Named-mailbox primitive (steer compat)
- [ ] Durable emit path + receiver counts + dedup
- [ ] Mailbox index (kill the 10k scan)
- [ ] Gateway handoff coordinated
- [ ] Fable5 adversary folded
