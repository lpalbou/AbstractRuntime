# 0859 — Replay view: ephemeral live-text deltas, ignored by the reducer

**Status**: planned · **Priority**: P1 (high) · **Created**: 2026-10-28
**Package**: abstractruntime (`replay/view.py` `apply`, conformance fixtures, ADR 0010 note) · **Slices**: abstractgateway 0954 (`/view/stream` `text` frames)
**Master item**: framework root backlog 0993; follows 0854 (its "Out" list put live token text on the ledger stream)

## Summary

Finding from the AbstractAssistant desktop migration (draft PR #11, `feat/replay-view-desktop`, its backlog 1063): the `/view/stream` deltas carry committed ledger records only, so a client that renders from the view loses the live token text while an answer is being generated, a regression against the old path (`GET /runs/{id}/ledger/stream`), which interleaves `llm.delta` frames from `LiveDeltaHub` (`live_deltas.py`) and shows tokens as they arrive. Runtime 0854 put live token text out of scope ("`llm.delta` frames stay on the ledger stream"); the migration shows the view stream needs it.

## Chosen route

- The view model stays a fold of committed ledger records only. Live text travels as an ephemeral `text` delta kind
  (`{run_id, node_id, call_id, channel, seq, text}` appended, `text_end`) produced by the gateway from
  `abstractruntime.core.live_deltas` sinks (gateway 0954), never persisted, superseded by the committed message record.
- `ReplayView.apply` / `apply(view, new_records)` ignore ephemeral deltas explicitly (a record of kind `llm.delta`,
  `llm.delta_end`, `text` or `text_end` leaves the view and `last_delta` unchanged), so a client that feeds the whole
  stream into a port of the reducer cannot corrupt its view; ADR 0010 states the rule (the cursor is `seq`; ephemeral
  frames carry none).
- Conformance fixture: a stream with interleaved ephemeral frames folds to the same view as without them; a fixture of
  the client rule "drop a call's live text when its committed message arrives".
- Interim (option B in gateway 0954): clients keep `/ledger/stream` open for live deltas alongside `/view/stream`.

## Acceptance criteria

- [ ] `apply` ignores ephemeral kinds; a test red when the guard is removed.
- [ ] Conformance fixtures shipped as package data; ADR 0010 amended.
