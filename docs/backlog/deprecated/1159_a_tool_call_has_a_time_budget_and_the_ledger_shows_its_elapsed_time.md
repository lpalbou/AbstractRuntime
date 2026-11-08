# 1159 — A tool call has a wall-clock budget; the ledger and the view show how long it has been running

**Status**: deprecated · **Deprecated**: 2026-11-08 (operator: "i never said that and i don't want that. we don't want silly arbitrary interrupts") · **Priority**: P1 (operator incident 2026-11-08 03:31: "you would think it works and it's just slow? no it isn't… so somewhere those tools crashed") · **Created**: 2026-11-08
**Package**: abstractruntime (`integrations/abstractcore/effect_handlers.py` TOOL_CALLS handler, ledger event schema, `replay/view.py`) · **Consumers**: AbstractUIC panel-chat (the tool row shows the elapsed time and a "still running" cue, uic 1161), AbstractCore `search_files` (its own deadline, root 1160)

## What happened

Code web, conversation run `c00883d1…`, react child `d03dc6d9…` on the operator's gateway (data dir inside the
workspace): the model called `search_files(pattern="theforge", path="/Users/albou/tmp/abstractframework",
output_mode="files_with_matches", max_hits=40)` twice in one `act`. The walk covered the workspace's
`untracked/` folders (review virtualenvs named `venv-<tag>`, 100 GB) and the live gateway data folder
`runtime/` (13,486 ledger files, 5.3 GB of JSONL scanned line by line). The first pair took about eight
minutes (03:28:07 → 03:36), the model then asked for two more searches, still running at 03:40. Throughout,
the gateway burned one core on one thread, the model host was idle, Code showed two spinners reading
"Running" with no elapsed time, and nothing in the ledger said how long the call had been running. The
operator read it as a crash. A workspace `.abstractignore` (root commit, 2026-11-08) fixes this machine's
layout; the runtime gap stays: a tool call can run for hours and the run cannot tell.

## Current code reality (2026-11-08)

- `effect_handlers.py` TOOL_CALLS: tools run synchronously inside the tick (`ToolExecutor` → core's tool),
  no wall-clock budget (`grep -n timeout` finds none on the tool path). A slow tool holds the runner thread.
- Ledger entries for `tool_calls` carry `started`/`completed` but no per-call duration; the replay view's
  tool rows show "0.0s" only after completion (the rendering in the operator's screenshot).
- The core tool has `max_hits` and an ignore set but no deadline (root 1160 asks core for one).

## Scope

1. **A budget per tool call**, declared on the effect payload (`tool_timeout_s`, default 120 s; the
   workflow or the host may raise it per tool: `analyze_media`, long installers). The tool runs on a
   worker thread; `future.result(timeout)`; on expiry the result is an honest tool error the model can act
   on: `search_files exceeded its 120 s budget (scanned N files under <path>); narrow the path, add
   file_patterns, or add a .abstractignore` — the partial figure comes from the tool when it reports one.
   The expired thread cannot be killed in Python: it is daemonized, its late result is dropped, and the
   ledger records `tool_timeout` with the thread id so a leak is visible in health (gateway 1158).
2. **Elapsed time in the record**: each tool result row carries `started_at`/`duration_ms`; a running call
   emits a `tool_progress` note at 10 s, 30 s, 60 s… (bounded) so the replay view can show "running for
   45 s" instead of a bare spinner. No wall-clock assertions in tests by default (ADR 0050 §6; `*_TIMING_STRICT`).
3. **Cancel stops the wait**: a run cancel during a tool call releases the runner at once (the thread may
   finish in the background; the result is dropped).
4. Tests: a fake tool that sleeps past the budget yields the timeout result and the ledger note; a fast
   tool is unchanged; cancel during a slow tool returns within the tick; the view projection exposes
   `elapsed_ms` for a running call.

## Non-goals

- Killing the tool's thread or process (not possible for in-process Python tools; subprocess tools are
  core's to bound).
- Changing which directories a search walks (core 1160 and the workspace `.abstractignore`).

## Validation

- Hermetic gateway: a `search_files` over a tree with a 200k-file decoy folder returns the timeout result
  within the budget + 1 s, the run continues, Code's row shows the elapsed time while it runs.

## Deprecation record

Operator ruling 2026-11-08 11:30: no wall-clock budget on tool calls, no arbitrary interrupts. The elapsed-time
record (started_at/duration_ms on tool results) is kept as a proposed display aid in uic 1161; the eight-minute
search is addressed by the workspace `.abstractignore` (root 8d9477b) and the ignore defaults in root 1160.
