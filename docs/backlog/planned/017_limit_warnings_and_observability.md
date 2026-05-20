## 017_limit_warnings_and_observability (planned)

**Status**: Planned
**Priority**: Medium
**Depends on**: 016_runtime_aware_parameters (completed)

---

## Goal

Expose runtime-owned limit warning state cleanly to hosts without relying on polling alone.

---

## Current code reality

- `_limits`, `get_limit_status(...)`, `check_limits(...)`, and `update_limits(...)` already exist.
- Runtime already updates `_limits["estimated_tokens_used"]` from `LLM_CALL` usage metadata when providers return it.
- Current observability in this repo is centered on durable ledger records and the global event-bus bridge, not an `on_step(...)` callback contract.

So the original item is stale. Token estimation is no longer the missing part, and agent prompt injection is not a runtime-core responsibility in this repo.

---

## Problem

Hosts can inspect limit state, but Runtime does not yet provide a first-class warning signal when a run crosses configured thresholds. That leaves UIs and drivers to poll and re-derive warning transitions themselves.

---

## Planned scope

1. Decide the runtime-owned warning surface:
   - ledger/event-bus emission
   - explicit helper API
   - or another durable runtime signal
2. Emit warning/exceeded transitions only when thresholds are crossed, not on every tick.
3. Cover both iteration and token warnings using the existing `_limits` model.
4. Document the relationship between durable limit state and derived host notifications.

If agent frameworks want warning text injected into prompts or inboxes, that should be tracked in the relevant host/agent repo, not here.

---

## Acceptance criteria

- [ ] Runtime documents which surfaces expose limit warning transitions.
- [ ] Warning and exceeded states can be observed without polling-only UX.
- [ ] Emission is threshold-based and avoids duplicate spam for unchanged states.
- [ ] Existing `_limits` APIs remain the canonical durable state.
- [ ] Token warning behavior is documented as best-effort and usage-metadata-driven.

---

## Validation

1. Threshold-crossing tests for iterations and tokens.
2. Observability tests for whichever warning surface is chosen.
3. Docs review against `docs/limits.md` and existing event-bus/ledger behavior.

---

## Non-goals

- re-implementing agent prompt guidance in this repo
- replacing the existing `_limits` APIs
- guaranteeing tokenizer-accurate accounting across providers

---

## Priority note

This is still useful, but it is not as urgent as the remaining media/workspace boundary gap.
