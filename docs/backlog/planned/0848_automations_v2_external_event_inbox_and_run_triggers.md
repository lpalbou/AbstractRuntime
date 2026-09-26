# 0848-abstractruntime: [FEATURE] Automations v2: durable external-event inbox, generic `event` trigger source, `run.finished`/`run.failed` sources with reconciliation

> Created: 2026-09-26
> Status: Planned (after v1)
> Type: feature
> Priority: P2 (next phase; starts only after [0847](0847_automations_v1_controller_contracts_occurrences_triggers.md) ships)
> Labels: automations, triggers, event-inbox, durability, reconciliation
> Design: untracked/design/automations-PLAN.md (2026-09-26)

## Summary

Let an Automation fire on things that happen outside its own clock: an external
event posted to the runtime (generic `event@1` source) or another run reaching a
terminal state (`run.finished@1`, `run.failed@1`). Admission goes through a
durable inbox stored outside run vars, so an event is admitted exactly once even
across crashes, restarts and a busy controller, and terminal-run sources
reconcile against the store instead of trusting callbacks.

## Why

Operator ruling 3 (2026-09-26, verbatim):

> Schedule + manual first; external triggers in the next phase: **agreed**, but the next phase is a PLANNED backlog item, not proposed.

Design §1, verbatim: "**Minimal scope.** Reuse commands, stores, effects,
bundles and tool ceilings. External admission, automatic summaries and connector
infrastructure are deferred." Design §5 places "durable external inbox, generic
events, terminal chaining/reconciliation" in V2.

v1 events reaching a controller would have nowhere safe to land: today
`_handle_emit_event` resumes only WAITING listeners and a busy listener drops
the event on every host except the gateway, whose `events_inbox` run-var mailbox
lives only in the gateway runner
(see [0051](runtime_systemic_reliability/0051_runtime_owned_durable_event_mailbox.md)).

## Scope and non-goals

### In scope

- **Durable inbox outside run vars**, keyed
  `(automation_id, binding_revision, event_id)`. Appending the same key twice is
  a no-op that returns the original receipt. The inbox never grows the
  controller's vars.
- **Admission design:**
  1. an event is appended to the inbox (durable, deduplicated);
  2. the controller admits the oldest unconsumed entry by *reserving* the
     occurrence id (deterministic, as in 0847 contract B) and recording
     `automation.admitted` with the `event_id`;
  3. the child run is created/loaded under that reserved id;
  4. the inbox cursor for the binding advances **only after** the durable
     parent→child link (the parent wait) is committed.
  A crash at any step re-admits the same entry to the same reserved id; nothing
  is lost and nothing duplicates.
- **Generic `event@1` source** (`capabilities.kind:"event"`, `TriggerWait`
  `{kind:"event",scope:"run",name}`) with a JSON-schema-validated payload
  normalized into a `TriggerEnvelope`.
- **`run.finished@1` / `run.failed@1` sources** watching named workflows or
  runs, with **reconciliation**: on controller start, resume and every wait
  boundary, scan the watched runs' terminal state in the store and admit any
  terminal run not yet represented in the inbox. Callbacks are an accelerator,
  never the source of truth: `core/runtime.py` saves terminal state before it
  appends the terminal status event (`:2690-2691`, likewise `:2640-2641` and
  `:2722-2723`), so a crash between the two leaves a terminal run with no event.
- Misfire policy for event sources: serial occurrences; events arriving while an
  occurrence runs stay in the inbox (no drop, no coalescing unless the binding
  opts in).
- Tests on both persistent stores: crash at each admission step, duplicate
  event ids, restart with a backlog, terminal run with a missing event.

### Out of scope

- Packaged connectors (file, email, build, journal) and richer calendar
  scheduling — later phase.
- Constrained fetching, streaming triggers, automatic Growing summaries.
- A universal broker, delivery router or arbitrary filter language.
- Gateway HTTP ingress for events (owned by the gateway; it calls the runtime
  inbox API defined here).

## Current code reality

Verified 2026-09-26 against `2100d1f`:

- Terminal state save precedes the terminal event (lines above); the method is
  `_append_terminal_status_event` (`core/runtime.py:2068`).
- The only durable per-run inbox for events is the gateway runner's
  `events_inbox` run var (abstractgateway `runner.py:705`, `:2213-2233`); the
  runtime has none.
- `storage/commands.py` provides an append-only command log with `command_id`
  dedup and a cursor store (`JsonlCommandStore.append` at `:276`,
  `JsonFileCommandCursorStore` at `:183`) — the closest existing shape for the
  inbox, but it is keyed per command, not per `(automation_id,
  binding_revision, event_id)`, and dedups only in-process.
- `list_run_index` has no terminal-since filter; reconciliation needs one (or
  `changed_since` from 0847 contract E).

## Acceptance criteria

- [ ] Each `(automation_id, binding_revision, event_id)` produces exactly one
      occurrence across crashes injected at append, admit, child create and
      parent-wait commit.
- [ ] The inbox cursor never advances before the parent→child link is durable.
- [ ] A watched run that became terminal while no callback fired (including a
      crash between terminal save and terminal event) is admitted by
      reconciliation.
- [ ] Events that arrive while an occurrence runs are admitted, in order, after
      it completes.
- [ ] Controller vars stay bounded regardless of inbox size.
- [ ] `event@1`, `run.finished@1`, `run.failed@1` appear through the
      `abstractruntime.trigger_sources` registry.

## Dependencies and ADR status

- Depends on [0847](0847_automations_v1_controller_contracts_occurrences_triggers.md)
  (contracts, deterministic ids, registry, index cursors).
- Should share its store shape with
  [0051](runtime_systemic_reliability/0051_runtime_owned_durable_event_mailbox.md);
  decide at start whether 0051 lands first and this inbox is a named mailbox on
  it.
- ADR impact: none expected beyond 0847's.

## Related

- abstractframework backlog 0928 (Automations umbrella).
