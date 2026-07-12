# Planned: Prompt-cache message-fingerprint stability (per-turn rebuild fix)

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Priority: P1 (perf-correctness; core's MLX bench proved ~7s/cycle wasted)
- Source: core seat, commons c1127 (maintainer-directed MLX cache
  investigation; core fixed its two layers, this is the runtime layer).

## ADR status
- Governing ADRs: None
- ADR impact: None (correctness/perf; no policy)

## Context
Core's 3-layer MLX cache investigation: core FIXED layer 1 (tool-sort byte
break in PromptCacheModule.normalized) and layer 2 (module-chain fed-token
record), taking generate-side delta from 8,282→238 warm tokens. Layer 3 is
runtime's: `_maybe_prepare_prompt_cache` rebuilds the WHOLE per-session KV
cache EVERY TURN (clear + fork + re-append the full transcript through
prompt_cache_update) because the message-fingerprint prefix-extension check
fails every turn.

## Current code reality (line-verified 2026-07-12, core's spy evidence)
- `_prompt_cache_message_fingerprint` (llm_client.py:271-289) hashes
  ROLE + CONTENT only — tool_calls are EXCLUDED. So two assistant messages
  with identical content-head but different tool_calls fingerprint the same
  (false-MATCH → wrong prefix reuse) AND a message whose tool_calls
  re-serialize differently across turns false-MISMATCHES.
- `_maybe_prepare_prompt_cache` (llm_client.py:~5656) filters envelope-ONLY
  and volatile-flagged messages from `msg_list`, then prefix-checks
  fingerprints; a mismatch → full clear+fork+re-append.
- Core's spy (steer bench, 3 cycles, mlx Qwen3-4B):
  - Turn 2: "prefix_ok=False; first mismatch at msg 0 (the TASK user
    message)" — msg 0 carries an INLINE runtime-grounding envelope with a
    fresh timestamp each turn (NOT envelope-only, so the existing filter
    misses it). THIS IS THE B3 MANIFESTATION (durable/sent byte drift in the
    final user message head, filed as proposed 0063) seen from the
    prompt-cache angle.
  - Turn 3: mismatch at msg 4 (assistant, empty content head) — tool_call /
    reasoning rendering drift across turns; the tool_calls-excluded
    fingerprint is implicated.

## Problem
Every warm turn re-prefills ~8k tokens through the update lane that core's
generate-side delta then doesn't need — ~7s/cycle on the bench, both arms.
Two distinct root causes: (a) inline grounding envelope destabilizes msg 0's
fingerprint (B3-coupled); (b) the fingerprint ignores tool_calls (false
match AND false mismatch, a correctness risk beyond perf).

## What we want to do
TWO coordinated fixes, sequenced by risk:
1. TOOL_CALLS IN THE FINGERPRINT (lower risk, correctness): include a
   stable-serialized tool_calls in `_prompt_cache_message_fingerprint`.
   REQUIRES first proving the tool_calls serialization is STABLE across
   turns for an already-sent message (append-only history says it should be
   — VERIFY; if it drifts, the real fix is stable serialization, and adding
   it to the hash naively would WORSEN the msg-4 mismatch).
2. MSG-0 GROUNDING STABILITY (B3-coupled, higher care): the fingerprint must
   be stable for a message whose only per-turn change is the injected
   grounding envelope — which is ONLY correct if the envelope is likewise
   excluded from what is FED to the cache (else a false-prefix-match caches
   stale KV). This is the same tangle as 0063 (B3): resolve them TOGETHER —
   move the chat-shape envelope to the separate trailing lane the tool-loop
   shape already uses, then both the fingerprint and the fed bytes are
   naturally stable.

## Why
Correctness (false-match can serve wrong KV) + a measured ~7s/cycle on local
control-plane providers; the fix compounds with core's two landed layers.

## Requirements
- No false-match: differing tool_calls never share a fingerprint.
- No new false-mismatch: an unchanged already-sent message keeps its
  fingerprint across turns (the whole point).
- Byte-true: whatever the fingerprint treats as "same" must be what the
  cache was actually fed (no stale KV).

## Suggested implementation
Fix 1 first (isolated, pinnable). Fix 2 rides the 0063/B3 wave (agent
coordinates the adapter tail). Consider core's offered alternative
(SKIP the update lane entirely for providers whose generate deltas
natively — MLX post-core-fix) as a THIRD option: it removes the whole
rebuild class for those providers with zero fingerprint risk, gated on a
provider capability flag. Assess which is cleaner during the build.

## Scope / Non-goals
Scope: the fingerprint + the update-lane rebuild decision. Non-goals: no
grounding-envelope CONTENT change (temporal-only ruling stands); do not
rush fix 2 ahead of the B3 coordination.

## Dependencies and related tasks
- proposed 0063 (B3 — the msg-0 half is the same root; likely ONE wave).
- core's two landed MLX layers (the generate-side delta this unblocks).
- 0049 (llm_client split — the fingerprint lands in normalize.py after).

## Expected outcomes
Warm turns take the incremental-append path; core's bench plain arm drops
toward ~1s/cycle while bust stays ~8s (the honest contrast the maintainer
asked to see); no false-match regressions.

## Validation
- Fix 1 pin: two assistant messages, same content, different tool_calls →
  different fingerprints; an unchanged tool-call message → stable across
  turns.
- Fix 2 pin (with 0063): a 3-turn chat-shape transcript takes the
  incremental-append path (no clear+fork rebuild); fed bytes == fingerprint
  basis.
- Re-run core's /tmp/bench_spy.py: prefix_ok=True on turns 2-3.

## Progress checklist
- [ ] Verify tool_calls serialization stability across turns
- [ ] Fix 1: tool_calls in the fingerprint + pins
- [ ] Fix 2 with 0063/B3: envelope to the trailing lane + pins
- [ ] Assess the skip-update-lane-for-native-delta option
- [ ] Re-run core's bench spy; prefix_ok green
- [ ] Fable5 adversary folded

## Guidance for the implementing agent
Do NOT add tool_calls to the fingerprint before confirming their
serialization is turn-stable — core's msg-4 evidence suggests it may drift,
in which case naive inclusion makes it worse. Fix 2 is B3; treat as one
wave with 0063, not a standalone patch.
