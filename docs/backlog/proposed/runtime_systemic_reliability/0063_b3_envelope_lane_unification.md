# Proposed: B3 — durable bytes == sent bytes for the chat-shape final user message

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A
- Cross-repo: agent seat coordinates (their adapter merges the loop tail;
  sequencing agreed on commons c978: runtime moves the merge to its
  boundary → agent's tail emission becomes unconditional).

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
Filed as B3 by the code seat's prompt-cache adversary (commons c971): the
runtime grounding envelope is injected into the HEAD of the final USER
message for chat-shaped payloads, so the durable copy of that message has
different bytes than what the provider received, and cross-turn server-side
prefix reuse dies at the first user message.

## Current code reality (verified 2026-07-12)
- Chat shape: `_normalize_turn_grounding` rewrites the final user message
  head (llm_client.py:~1029-1033).
- Tool-loop shape ALREADY has the fix lane: the envelope rides a SEPARATE
  trailing envelope-only user message (llm_client.py:~1036-1042), excluded
  from prompt-cache fingerprints, replaced fresh per call.
- Mitigation already in place: tick-side ledger grounding injection exists
  so the ledger shows what was sent (`_mark_grounding_prompt_injected`,
  runtime.py:531-592) — the divergence is real but bounded and documented.
- The volatile-marker machinery (B1 fix) provides the structural flag lane.

## Problem or opportunity
Durable-vs-sent byte divergence on one message class; degraded cross-turn
server prefix caching for chat-shaped conversations.

## Proposed direction
Make chat shape use the SAME trailing-envelope lane as tool-loop shape;
perform any alternation-strict user,user merge at the PROVIDER boundary
only (where volatile markers are already stripped), never in the durable
payload. Agent then flips their adapter tail to unconditional-separate
(their one-line change, pre-agreed).

## Why it might matter
One envelope lane instead of two; byte-true durable payloads; server-side
prefix reuse across turns for chat shapes.

## Promotion criteria
Coordinated wave with the agent seat (both halves same day), or the next
prompt-cache measurement showing chat-shape reuse matters in production.

## Validation ideas
Pin: durable messages byte-equal the provider-bound payload minus stripped
markers; alternation-strict template smoke (the user,user merge happens
post-fingerprint, pre-provider); cache A/B on a chat-shaped session.

## Non-goals
No envelope content changes (temporal-only ruling stands); no head
rewrites anywhere (append-only rendering is the law for entity lanes
already).

## Guidance for future agents
Read the B1 volatile-marker code first — the strip/exclude seams it added
are exactly where the merge must happen.
