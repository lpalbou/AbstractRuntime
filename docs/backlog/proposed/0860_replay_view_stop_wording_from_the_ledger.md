# 0860 — Replay view: a forced stop reads "Stopped" (the stop's wording is not in the view)

**Status**: proposed · **Priority**: P2 · **Created**: 2026-10-28
**Package**: abstractruntime (`replay/view.py`) · **Slice**: abstractgateway (`discovery/capabilities`)
**Parent**: 0854 (replay view), framework root backlog 0993 · **Found by**: AbstractUIC backlog 0040 (panel-chat 0.6.0 renders the view)

## Problem

A client that renders the replay view cannot tell a Stop the person asked for from one the gateway
forced. The ledger knows: the gateway's kill switch writes an `abstract.status` record with
`killed_by: "kill_switch"` and its text ("Stop forced at 10 s: inference killed", conformance case
`d_cancellation`). The reducer keeps only the text, as `display_status` of the run
(`view.py` `_fold_event`, the `abstract.status` branch), and drops `killed_by`. panel-chat's ledger
fold showed "Stop forced: inference killed" from that record; in view mode it can only say
"Stopped", and a client that clears `display_status` on a terminal status (panel-chat does, as its
ledger fold did) loses the text too.

A second, related gap: AbstractGateway 0.17.0 does not list the view in `GET discovery/capabilities`,
so every client must probe `GET /view` once and treat FastAPI's route-missing 404 as "no view".

## Proposal

- The view carries how the run was stopped: on the root (and each `runs[]` row) a `stop` object
  `{by: "person"|"kill_switch"|"timeout"|…, text, seq}` folded from the attribution record
  (`killed_by`, `cancelled_by` on the cancelled `llm_call`), and a `stop` change in `changes`.
- The gateway advertises `capabilities.replay_view: {supported: true, version: "replay_view_v1",
  endpoints: {view, stream}}`.

## Acceptance

- `d_cancellation`'s expected view names the kill switch and its text; a person's cancel
  (`cancelled_by`) reads as such; a new conformance case for a person's Stop.
- `GET discovery/capabilities` lists `replay_view`; panel-chat passes it as `advertised` and stops
  probing.
