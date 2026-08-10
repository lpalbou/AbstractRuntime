"""abstractruntime.integrations.abstractcore.constants

Single source of truth for AbstractRuntime orchestration defaults when it
executes AbstractCore-backed effects (LLM calls and tool calls).
"""

# Default *effect* timeouts (seconds).
#
# IMPORTANT: These are NOT a "workflow/run TTL". Workflows can be long-lived
# (hours/days) or even continuous. These limits only apply to a *single*
# runtime-managed operation (e.g. one LLM HTTP request, one tool call).
#
# Rationale:
# - Local inference can be slow for large contexts.
# - In an orchestrator, timeouts are policy and should be explicit + consistent.
# #[WARNING:TIMEOUT] — ADR-0014 (runtime is the authority for per-effect
# budgets) + ADR-0027 §2/§4. 7200s is the deliberate high safeguard for local
# inference; nothing downstream may cap below it silently.
DEFAULT_LLM_TIMEOUT_S: float = 7200.0
# #[WARNING:TIMEOUT] — STREAMING-ONLY no-progress bound (0152 face 2, core
# c5051 + runtime c5041): httpx's READ timeout separated from the absolute
# total above — a STREAM that delivers nothing for this long dies at the socket
# instead of pinning a tick worker for the 2h backstop. None disables; entity
# lanes thread tighter numbers.
#
# CRITICAL (2026-08-02 LM Studio incident, CONFIRMED): this value is NOT a
# total budget and MUST NOT be applied to non-streaming requests. On a
# `stream: false` call the response body only begins arriving after generation
# has fully finished, so httpx's `read` becomes time-to-first-byte — i.e. a cap
# on the WHOLE generation. Applied client-wide it silently aborted every local
# tool call longer than 5 minutes: LM Studio logged `Client disconnected.
# Stopping generation...` at exactly +300.0s and dropped the half-emitted tool
# call (`Failed to parse tool call: Unexpected end of content`), so the agent
# received `tool_calls: []` and looped. abstractcore now gates the bound behind
# `streaming=True` (abstractcore/providers/_http.py), applying it per streaming
# request only; the 7200s total above governs non-streaming calls, as ADR-0014
# requires. Keep that gate — raising or lowering THIS number does not protect a
# non-streaming call and must never be used to.
# #[WARNING:TIMEOUT] Streaming-only idle-gap bound, 1800s (operator ruling
# 2026-08-02: local models can be legitimately slow between chunks).
# This is the max GAP BETWEEN CHUNKS on a STREAMING response, never a
# generation cap — abstractcore applies it only when streaming=True
# (providers/_http.py). It was 300s AND was being applied to
# non-streaming calls, where the body arrives only after generation
# finishes, so it silently became a hard 300s cap on every long call.
DEFAULT_LLM_READ_IDLE_TIMEOUT_S: float = 1800.0
# #[WARNING:TIMEOUT] — ADR-0014/ADR-0027: high safeguard for one tool call.
DEFAULT_TOOL_TIMEOUT_S: float = 7200.0


