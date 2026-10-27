# 0855 — Prefix-stability contract of consecutive LLM calls (only trailing system riders may differ)

**Status**: completed on branch `fix/prefix-cache-reuse` (2026-10-27), UNRELEASED · **Priority**: P1 ·
**Created**: 2026-10-27
**Cross-package**: AbstractCore 0858 (native MLX snapshot before trailing riders: the provider half, must ship in a
Core release); root [1060](../../../../docs/backlog/recurrent/1060_gateway_security_resilience_and_latency_standing_duty.md)
(latency budget) and [1056](../../../../docs/backlog/proposed/1056_programmable_ai_systems_workflows_as_the_central_primitive.md).

## Why
Operator report 2026-10-27: a gateway agent run on a local MLX hybrid model re-prefilled the whole prompt at every
step (10 of 10 calls cold, 24,552 tokens at step 10). Audit of the run's ledger: the runtime's requests ARE
append-only — system prompt and tool list byte-identical across 11 calls, every message of call N present
unchanged in call N+1, the grounding envelope stamped once into the durable user turn — except one thing: the
session attachment index, a `system` rider re-injected at the TAIL on every call (`effect_handlers.py`, backlog
0212). That rider is the intended design; the cold calls came from the provider's snapshot including it (Core
0858). Nothing pinned the runtime's half of the contract, so a later change (a timestamp in the system prompt, the
index moved between turns, a reordered tool list) would silently re-create the same symptom on every backend.

## What shipped
- `tests/test_llm_call_prefix_stability_contract.py`, through the real `Runtime` (ledger grounding pass) and
  `make_llm_call_handler` (attachment injection), with a session attachment and a clock that moves on every read:
  across 4 consecutive tool-loop calls `system_prompt`, `tools` and generation params are identical; call N minus its
  trailing system riders is a prefix of call N+1; no `system` message sits mid-history; a moving timestamp never
  reaches the prefix. Red on regression (verified by mutation): a timestamp in the system prompt → 2 failed; the
  index placed after the first user turn → 1 failed.
- The injection site documents the rider contract and names the Core seam.

## Limits (from the adversarial review)
- The test stops at the LLM-client seam (stub client): what `llm_client` does afterwards (grounding normalization,
  synthetic/volatile key stripping) is not covered, and the transcript is built append-only by the test itself; the
  agent adapter owns that shape. Follow-up: drive the real local client over a stub provider.

## Not changed (deliberately)
The index stays a trailing rider. Moving it into the history (after the turn's user message) would make the tool
loop restorable with today's Core but makes every cross-turn first call cold once Core 0858 ships; stamping it
durably would accumulate copies per turn.
