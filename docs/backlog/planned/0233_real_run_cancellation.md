# 0233 — Real run cancellation: abort in-flight LLM calls (abstractruntime slice)

**Status**: planned · **Priority**: P1 · **Created**: 2026-08-02
**Master item**: `docs/backlog/planned/0233_real_run_cancellation_abort_in_flight_llm_calls.md`
(framework root) — read it first; it carries the full evidence and the cross-package plan.

## Why this is here
`/cancel` today only writes `status=CANCELLED`; the in-flight generation runs to completion
and keeps consuming GPU/paid tokens. CONFIRMED 2026-08-02 by tracing
abstractcode-tui → gateway `_apply_run_control` → `abstractruntime/core/runtime.py:1302`,
plus LM Studio slot-span analysis (5 overlapping generation pairs, largest 211s).

## This package's slice
- `cancel_run` (`core/runtime.py:1302`) must signal the in-flight effect's token
  BEFORE writing `status=CANCELLED`, then let the tick thread unwind.
- `LLM_CALL`/`TOOL_CALLS` handlers thread the token down to abstractcore.
- Ledger records a distinct `cancelled_in_flight` outcome — never a normal completion or a fault.

## Validation (shared)
Cancel a long local generation mid-flight; the LM Studio log must show it stopping within seconds,
no slot left busy, ledger shows `cancelled_in_flight`, tokens stop accruing, and cancelling run A
must not disturb run B.
