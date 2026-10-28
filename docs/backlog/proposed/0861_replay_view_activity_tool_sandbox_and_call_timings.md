# 0861 — Replay view activities: the tool sandbox and per-call model timings

**Status**: proposed · **Priority**: P2 · **Created**: 2026-10-28
**Package**: abstractruntime (`replay/view.py` `_fill_activity`)
**Parent**: 0854 (replay view), framework root backlog 0993 · **Found by**: AbstractUIC backlog 0040 (panel-chat 0.6.0 renders the view)

## Problem

Two things clients showed from raw records are not in `replay_view_v1.activities[]`:

1. **The command sandbox of a tool call.** A process-spawning tool's result carries
   `output.sandbox` (macOS sandbox-exec, Linux bubblewrap/Landlock, "none — refused", the enforced
   workspaces). panel-chat's `ToolSandboxLine` renders it from the ledger; a view tool call has
   `output` only when the output is the tool's value, and the sandbox block is not kept as a field.
2. **Per-call model timings.** An `llm` activity carries `usage` (tokens) but not the call's
   `gen_time`, the provider's prompt rate (prefill time), cache reuse or the model load. panel-chat's
   turn statistics (`genTimeMs`, `prefillMs`, `decodeMs`, `timedCalls`, the slowest tool batch)
   came from these record fields; in view mode only tokens, call counts, changed files and the
   turn's span remain.

## Proposal

- `tool_calls[].sandbox`: the call's sandbox block as the tool reported it (`label`, `enforced`,
  `workspaces`, `refused`), absent when the tool spawned nothing.
- `activities[kind=llm].timing`: `{gen_time_ms, prompt_tokens_per_s, completion_tokens_per_s,
  cached_tokens, load_ms}` exactly as the record reported them (a field absent when not reported:
  zero is a claim).
- `usage` on the view keeps its meaning (sums); timings are not summed by the reducer.

## Acceptance

- A conformance case with a sandboxed `execute_command` and an MLX call that reports
  `gen_time` / rates; expected views carry both.
- panel-chat's `replayStatistics` and tool rows read them (AbstractUIC follow-up of 0040).
