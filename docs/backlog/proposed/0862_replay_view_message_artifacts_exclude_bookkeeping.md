# 0862 — Replay view: a message's artifacts are what the run produced, not its offloaded bookkeeping

**Status**: proposed · **Priority**: P2 · **Created**: 2026-10-28
**Package**: abstractruntime (`replay/view.py` `_final_message` / `_note_artifacts` / `_artifact_refs`; the run-output and node-trace offload)
**Parent**: 0854 (replay view) · **Found by**: operator QA 2026-10-28 (b) on the desktop Assistant (AbstractAssistant backlog 1067)

## Problem

An automation answer in the desktop Assistant showed two chips named by raw ids (`7c8efef…`, `f738932…`); opening
one failed with "artifact not found". Evidence (operator data dir, run `68c30dd3…`, session
`automation:7f7613b4…`, AbstractRuntime 0.15.0, AbstractGateway 0.17.0):

1. **The final message lists bookkeeping as its artifacts.** `_final_message(..., refs=_artifact_refs(output))`
   takes EVERY `$artifact` under the completion output. An agent's output carries `scratchpad`; the runtime
   offloads its large values into the artifact store, so the message's `artifact_ids` were the agent's system
   prompt (`text/plain`, 4,340 B, tag `source=node_trace_offload`, path
   `output.scratchpad.node_traces.reason.steps[0].result.metadata._provider_request.payload.messages[0].content`)
   and its step list (`application/json`, 145,460 B, `source=run_output_offload`, path `output.scratchpad.steps`).
   Neither is something the person asked for.
2. **`view.artifacts` misses a ref the message lists.** `_note_artifacts` walks the completion RESULT, one level
   above the output `_final_message` walks; the system-prompt ref sits at depth 12 from the result, and
   `_artifact_refs` stops at `_MAX_WALK_DEPTH = 12`. The message names an artifact the view does not describe
   (no path, no owning run): a client cannot label it or know which run serves it.
3. **The offload belongs to a child run.** The system-prompt artifact's `run_id` is the agent's child run
   (`d072afe8…`, parent chain under the root), with no `session_id` tag; the gateway's content route serves an
   artifact only through its own run or the session's memory run (`_assert_artifact_visible_to_run`), so a client
   asking through the root run gets 404 "artifact not found" although the blob exists.

## Proposal

- A message's `artifact_ids` exclude refs under the run's bookkeeping (`scratchpad`, node traces, `_`-prefixed
  metadata such as `_provider_request`, `_runtime_observability`), or carry a `role` per artifact
  (`output` / `bookkeeping`) so every client applies one rule. Conformance case: an agent completion whose output
  holds offloaded scratchpad values and one real `image_artifact`.
- `_note_artifacts` and the message walk use the same root and depth bound, so every id a message lists is in
  `view.artifacts` (invariant: `message.artifact_ids ⊆ {a.id for a in view.artifacts}`).
- Offloads made for a run in a session carry the session (tag or descriptor), so the gateway can serve a child
  run's artifact through the root run of the same session; or the view's artifact entry names its owning run (it
  does: `run_id`) and clients are told to use it (AbstractAssistant 0.16.3 does).

## Client status

AbstractAssistant 0.16.3 drops unlisted and bookkeeping refs on its side and opens each artifact through its
owning run; AbstractUIC panel-chat and the iPhone/watch clients should apply the same rule until this lands.
