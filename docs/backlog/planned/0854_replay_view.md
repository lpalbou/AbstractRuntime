# 0854 — Replay view: one reducer of a run tree's ledger, served as a view every client renders

**Status**: planned, in progress (runtime slice released in abstractruntime 0.14.0, 2026-10-27, PR #8 merge `0efe68f`; gateway routes and client migration remain) · **Priority**: P0 · **Created**: 2026-10-27
**Package**: abstractruntime (reducer, durable cursor, conformance fixtures, ADR) · **Slices**: abstractgateway (routes, cache), every client that renders runs (migration)
**Master item**: framework root backlog 0993, `abstractframework/docs/backlog/planned/0993_one_ledger_replay_many_views.md`
**Release wave**: framework root backlog 1055, `abstractframework/docs/backlog/planned/1055_release_wave_2026_10_27_voice_library_replay_memory.md` (abstractruntime 0.14.0 carries this item; abstractgateway 0.16.0 waits for it)
**Cross-package**: AbstractGateway backlog 0082 (`abstractgateway/docs/backlog/proposed/0082_ledger_replay_integrity_and_durable_cursor.md`), 0075 (`abstractgateway/docs/backlog/proposed/0075_event_driven_ledger_streaming.md`), 0943 (`abstractgateway/docs/backlog/planned/0943_replay_view_routes_cache_and_client_migration.md`), 0939 (sessions index, out of scope here); AbstractAssistant backlog 0008 (`abstractassistant/docs/backlog/planned/0008_v2_runtime_replayer_adr_compliance_hardening.md`)
**ADR**: [ADR 0010 — One replay, many views](../../adr/0010_one_replay_many_views.md)

## Summary

Every client that shows a run (status, the approval or question it waits on, tool rows, replies,
activity) rebuilds that state today from raw ledger records, run snapshots and its own memory, each
in its own way. This item puts the one reconstruction in AbstractRuntime: a pure, deterministic
reducer `abstractruntime.replay` that folds a run tree's ledger records and durable snapshots into
a versioned JSON view (`replay_view_v1`), incrementally with `apply`. AbstractGateway serves that
view and its deltas; clients render it and act through the gateway's commands. Each client deletes
its private reconstruction once it passes the conformance suite.

This item also records the inventory of every reconstruction path that exists today (master item
0993, scope step 1), with file:line anchors and the points where each path can disagree with the
reducer.

## Why

Operator, 2026-09-30, after the delegated-approval bug: "there should be only one unified way to
replay the ledger, on top of which we could have different views, but only one way to replay the
ledger." Operator principles this item implements (quoted as given):

- "logic should not be in the thin client" (2026-10-01; full text in 0993: "full execution must
  happen in gateway/runtime and clients just read what's happening + forward execution parameters
  and requests/interactions from the user; all of that should be handled by the replayable
  ledger").
- "each client decides what to show and how to forward requests from users and entities".
- "workflows are the framework's central native primitive (agents, automations, trigger responses
  are workflows)": the view names the workflow (bundle, version, flow, entrypoint) each run
  executes and keeps the run tree (delegation) explicit.
- "special attention to performance (caching, indexing, leveraging the runtime/ledger, the
  file<->SQL coupling)".

Incidents, each a disagreement between two reconstructions of the same ledger:

| Date | What happened | Where it was patched | Rule the reducer now owns |
| --- | --- | --- | --- |
| 2026-09-29 | `basic-agent@0.0.5`: the root parks on `subworkflow:<child>`, the child asks to approve `write_file`. The client that started the turn showed Allow/Deny; a second browser and a phone showed "Running a tool write_file" and a Steer composer. The history fold replaced the child's waiting record with the root's delegation wait, and the tool row read the `$slim` pointer as "no calls". | panel-chat 0.1.20 / AbstractCode web 0.6.2 (client side) | A delegation wait is never the interaction; the newest open approval or question anywhere in the tree is. `$slim` resolved before any client sees the record. |
| 2026-10-04 | Code web's Activity showed "Start · Waiting for you" on a completed run: a `wait_until` timer has no resume record (182 of 3,278 runs on the operator's gateway). | `abstractcode/web/src/lib/activity_rows.ts` (client side) | A wait is over when its run records a resume for its key or any later step. |
| 2026-10-11 | AbstractFlow showed every Agent node twice ("Reviewer 0ms" + "Reviewer 14.5s"): one node execution is several ledger steps with different step ids. | `executionIdFor`, `abstractflow/src/utils/ledgerEvents.ts:64` (client side) | One activity per node execution (`replay/executions.py`, ported from Flow). |
| 2026-10-26 | The iPhone's conversation list shows "1+" where the watch shows a plain number: every client folds `GET /runs?root_only=true` pages by `session_id` over its own window. | not fixed | Out of scope here: gateway backlog 0939 (sessions index with exact counts). |

## Current code reality

Checked on 2026-10-27 against: abstractruntime `feat/replay-view` (82c0c74), abstractgateway main
(227c6890), abstractuic main (c6b7cc6, panel-chat 0.4.1), abstractcode main (85b32ec, web 0.12.0),
abstractassistant `feat/voice-library` (d9fb414), abstractobserver main (e308e6b), abstractflow main
(75c059e), abstractnote main (b8b0f7b), abstractentity main (a2aea5c). Line numbers are those of
these commits.

### What the runtime already provides (this branch)

- `src/abstractruntime/replay/view.py`: the reducer (`replay`, `apply`, `ReplayView.to_json`).
- `src/abstractruntime/replay/executions.py`: execution identity (`ExecutionTracker`), the five
  rules of Flow's `executionIdFor` plus progress ticks.
- `src/abstractruntime/replay/delta.py`: `apply_delta`, `diff_views`, `parse_cursor_token` (the
  client side of a delta and of a reconnect).
- Durable cursor: every ledger record carries its per-run `seq`.
  `storage/json_files.py:2496` `JsonlLedgerStore.append` persists it under an advisory file lock
  that spans the head read and the write (commit 82c0c74, gateway 0082 question 3);
  `storage/json_files.py:2772` `list_after`; `storage/sqlite.py:1523` `list_after`;
  `storage/sqlite.py:1202` `import_records` keeps each record's `seq` (the file to SQLite path);
  `storage/base.py:264` a default `list_after` for every store; `storage/in_memory.py:222`;
  `storage/observable.py:89` passes `seq` to subscribers.

### Server-side reconstruction paths (gateway and runtime)

| Path | Reads, in order | Status | Interaction | Tool / message / activity | Divergence from the reducer |
| --- | --- | --- | --- | --- | --- |
| History bundle: runtime `history_bundle.py:981` `export_run_history_bundle`, served by gateway `routes/gateway.py:10825` `GET /runs/{id}/history_bundle` | root snapshot (`history_bundle.py:1021`); descendants by `run_store.list_children` BFS (`:528`, `:1031`); each run's `ledger_store.list` (`:1093`); tail window of 2,000 records per run by default (`:1125-1150`) | root snapshot only (`:1219-1231`); child snapshots are not in the bundle | root `waiting` only (`:1230`); a child's wait is only in its ledger records, which the tail window can drop | none: raw records; `detail=replay` replaces request-side payloads with `$omitted` (`:947`) | Cursors are record indexes (`cursor_start = start_idx + 1`, `:1127`), not `seq`. `$slim` pointers stay unresolved in `ledgers[*].items`; the warning itself says a pointer whose STARTED record fell before the window "cannot resolve client-side" (`:1135`). Every client folds the bundle itself. |
| `GET /runs/{id}/ledger`: gateway `routes/gateway.py:11502` | run snapshot (existence check), then `ledger_store.list` of the whole run (`:11529`), sliced `[after:after+limit]` (`:11532-11534`) | none | none | raw records | Cursor is a record index; the whole ledger is read for every page. One run per call (`POST /runs/ledger/batch`, `:11538`, for many). |
| `GET /runs/{id}/ledger/stream` + `ledger_tail.py` | snapshot (`routes/gateway.py:11673`), then an incremental tail (`:11708`, `ledger_tail.py:318` `resolve_ledger_tail`: JSONL byte-offset tail `:78`, SQLite `list_after` `:202`, list-slice fallback `:244`); live `llm.delta` frames from `live_deltas` | terminal from the run snapshot, re-read on a settle window; `event: done` after a final drain | none | raw `step` frames, one run per stream; live deltas cover the run and every run below it | Already event-driven: `ObservableLedgerStore.subscribe` is wired as a coalescing dirty flag (`routes/gateway.py:11741`), `Last-Event-ID` is honored (`:11683`), the 0.25 s wait is a timeout on that flag that only `stat()`s the file (`:11884`, needed only in the split-runner deployment where appends happen in another process). **Gateway 0075's "Current code reality" (250 ms full re-parse, `subscribe()` never called, `Last-Event-ID` ignored) is stale.** The stream's `id:` is still "records consumed" (`ledger_tail.py:20-22`), equal to `seq` only while seq is dense. Children need their own streams. |
| `GET /runs/{id}`: gateway `routes/gateway.py:9837` `get_run` -> `service.py:1905` `run_summary` | run store `load` | the snapshot's `status` | `waiting` copied as is: reason, wait_key, until, prompt, choices, allow_free_text, details (`service.py:2057-2066`) | none | Says "waiting" between the runtime's `resume` append and its save without the wait (the ordering hazard of 0993). Never names the child that holds the question. A timer that ended without a resume record still shows its wait until the run saves. |
| Per-user event stream: gateway `events_stream.py:825` `wait_attention_from_record`, called from the runner's append hook (`runner.py:879`); closed by `still_waiting_on` (`events_stream.py:934`) and `wait_outcome` (`:897`) from `runner.py:514-525`; read by `GET /me/events[/stream]` (`routes/me_events.py:75`, `:173`) and `GET /runs/{id}/after` (`run_after.py`, tree by `list_children`, `:180`) | the waiting record at append time; later the run snapshot | `run.finished` pointer | `attention.raised {kind: approval|question}`; `prompt_ref` points the client to `GET /runs/{id}` (`events_stream.py:888`) | none | Its own wait classifier: a tool approval is only `details.mode == "approval_required"` with reason `user` (`events_stream.py:848`); an EVENT wait with a prompt or choices is a "question" (`:849-852`). The reducer's `wait_kind` (`replay/view.py`) also accepts `details.kind == "tool_approval"` and a `tool_approval:` key, and treats an EVENT wait as `event` (not person-facing). The runtime's own `automations/attention.py:168` `wait_kind` agrees with the gateway, not with the reducer. Resolution is decided from the snapshot (`still_waiting_on`), so it inherits the resume-before-save window. |
| `abstractgateway/migrate.py:58-86` (file to SQLite) | each `ledger_<run>.jsonl` line | n/a | n/a | n/a | Renumbers: `seq += 1` per parsed line (`:73`), so a held cursor does not survive and a JSONL `seq` is discarded. The runtime's `SqliteLedgerStore.import_records` (`storage/sqlite.py:1202`) keeps it; the gateway switches to it under 0943. |

### Client-side reconstruction paths

#### panel-chat (`@abstractframework/panel-chat` 0.4.1, `abstractuic/panel-chat/src/workflow_runtime.ts`); used by AbstractCode web and any host that embeds the chat

- **Reads, in order.** `load()` (`:338`) requests `getRun(root)` and `getHistory(root)` in parallel
  (`:348`); AbstractCode web's transport asks for
  `history_bundle?include_subruns=true&include_session=true&ledger_mode=full`
  (`abstractcode/web/src/workspace/session_transport.ts:24-27`). It folds the bundle's session turns
  and then every run's bundle records (`ingestHistory`, `:752-783`), applies the root snapshot
  (`:362-368`), opens the child named by the root's wait (`:371`), then `catchUpAndWatch(root)`
  (`:372`): `GET /ledger?after=` pages (`catchUp`, `:666-681`) then the SSE stream (`followStream`,
  `:692`), and after each stream end another page and another `GET /runs/{id}` (`:719-724`).
  Every record that names a child (`subRunIds`, `:250-259`: `sub_run_id`, `child_run_id`, a
  `subworkflow:` key) opens a catch-up and a stream for that child (`ingestItem`, `:876`).
  `pollRootLifecycle` re-reads `GET /runs/{root}` every 2 s while the run is not terminal
  (`:1273-1290`).
- **Cursor and dedup.** Record identity is `${runId}:${cursor}` where the cursor is the bundle's or
  the page's record index (`:865-869`); a record arriving with a different index (bundle tail vs
  page) is folded twice.
- **Status.** Only from `GET /runs/{id}` (`updateRunState`, `:1217-1235`); the bundle's snapshot is
  ignored for lifecycle (`:775-777`). Ledger terminal records are not read.
- **Interaction.** Any waiting record that is not a delegation becomes the interaction, the latest
  arrival winning (`foldRecord`, `:929-934`; `interactionWith`, `:516-521`); a snapshot wait is
  adopted unless the ledger's `resume` resolved that key (`adoptDurableWait`, `:530-538`,
  `resolvedWaits`, `:288`). The interaction is cleared only by a `resume` record with
  `result.resumed === true` (`:937-944`) or a terminal run state. Event waits are presented as a
  "Send event" interaction (`present_interaction.tsx:111`).
- **Tools.** `foldWorkflowTools` (`workflow_evidence.ts:193`): a waiting record whose payload is a
  `$slim` pointer falls back to `result.wait.details.tool_calls` (`:202-209`); identity is
  `runtime_call_id`/`call_uid`, else `idempotency_key|step_id|cursor` plus the call index.
- **Messages.** `answer_user` records (`:887-893`), `abstract.message` events (`:896-907`), the
  bundle's session turns and input prompt (`:762-774`); the final answer comes only from a terminal
  run snapshot's `output` (`:1225-1233`).
- **Client-held state.** `approvedWaits`, `grantedToolWaits` (`:279-282`), the automatic approver
  (`autoApproveCurrentWait`, `:588`), the stop evidence (`:972`).
- **Divergence.** (1) No "later step closes a wait" rule: a timer or event wait with no resume
  record stays `snapshot.interaction` until a terminal snapshot. (2) "Latest arrival wins" depends on
  which child stream delivers first; the reducer orders by (record time, seq, run id). (3) Event
  waits are person-facing here and not in the reducer's `interaction` (they are in `open_waits`).
  (4) The root's final message waits for the snapshot; the reducer takes the ledger's
  `system:completion:*` record first. (5) Index cursors: a migrated or windowed ledger can repeat
  or skip records. (6) A timed 2 s `GET /runs` poll per open conversation.
- **Evidence of the 0.1.20 fix.** `abstractuic/panel-chat/scripts/check_approval_sync.mjs` with the
  redacted live fixture `scripts/fixtures/run_approval_subrun.json` (header `:1-15`); AbstractCode
  `web/e2e/approval_sync.spec.ts`. Both become conformance cases (see below).

#### AbstractCode web (`abstractcode/web`, 0.12.0)

- The chat is panel-chat's controller (above). The Activity panel is AbstractCode's own fold over
  the controller's records: `activity_rows` (`src/lib/activity_rows.ts:890`), rendered at
  `src/workspace/workspace_panels.tsx:92`.
- **Rows.** One row per `run|effect type|idempotency_key or step_id or cursor|attempt`
  (`group_key`, `:739-743`); `merge_group` (`:751`) keeps the concrete payload over a later `$slim`
  pointer (`:759`). An Agent node is therefore several rows (its `start_subworkflow` and its
  structured-output `llm_call` differ by effect type) and a retry with a new `attempt` is its own
  row; the reducer makes one activity per execution.
- **Wait end.** A wait row ends when its run has a later step (`movedOn`, `:922-923`, the
  2026-10-04 fix) or when any run resumed its wait key (`resumeKeys`, `:906-913`: the set is not
  per run).
- **Dead code.** `src/lib/wait_resolution.ts` (`resolve_blocking_wait`: the newest record of the
  child decides), `src/lib/history_bundle_seed.ts` and `src/lib/subworkflow_follow.ts` are imported
  only by their own tests: three earlier reconstructions to delete, not migrate.
- **Conversation list.** `GET /runs?root_only=true...&limit=` folded by the client
  (`src/workspace/conversation_paging.ts:15`); gateway 0939.

#### AbstractCode terminal client (Rust, `abstractcode/tui/src`)

- **Reads, in order.** Session attach (`runner.rs:1820`): `GET /runs?session_id=..&root_only=true`
  (`gateway/mod.rs:599`); the live run is the newest one whose status is running or waiting
  (`runner.rs:1923-1933`); earlier turns from `history_bundle?detail=replay` (`gateway/mod.rs:932`,
  `runner.rs:1986`). Live attach (`runner.rs:2357`): the bundle (`:2374`), then one
  `ledger/stream?after=<cursor>` per run of the bundle (`runner.rs:2425-2436`,
  `gateway/mod.rs:976`); on a failed bundle, `input_data` and a stream from 0 (`runner.rs:2444-2485`).
  `GET /runs/{id}` on stream idle and at the end (`runner.rs:3664`, `:3826`); `GET /ledger?after=`
  when the stream fails (`:3731`) and a final drain (`:3785`). Automation activity is a separate
  fold over `GET /ledger` that walks children depth first, not in time order
  (`gateway/rail.rs:220`).
- **Status.** Two phases, Running and Idle. A turn ends on an answer-shaped flow end of the root or
  of the bound agent child (`transcript.rs:1681`), on `done`, or on a terminal `GET /runs/{id}`
  (`runner.rs:3667`). No `system:completion:*` or `system:terminal_status:*` read anywhere in `src`.
  A wrapper root still waiting after its child answered shows Idle (`runner.rs:3348`).
- **Interaction.** Waiting records (`transcript.rs:1561`); approval calls from
  `result.wait.details.tool_calls` (`protocol.rs:115`), so `$slim` does not affect the card; one
  `pending_wait` slot, newest wins (`transcript.rs:2594`, `:2613`); deduped by (wait_key, step_id)
  (`:2527`); children followed through `subworkflow` waits (`:1563-1570`, `runner.rs:3311`).
  **Any** later record of the waiting run clears it (`transcript.rs:1603-1610`), observability
  events included.
- **Tools and text.** Rows keyed `call:{run}:{call_id}` or `pos:{run}:{node}:{i}`
  (`transcript.rs:2320`); `$slim` not resolved, completed records fall back to `result.results`
  (`protocol.rs:360-379`); live text from `llm.delta` frames keyed by step id (`live.rs:1-35`,
  `runner.rs:3397`). Bundle records of several runs are merged by sorting on the
  `ended_at`/`started_at` string (`runner.rs:4174-4247`). Reconnects use `after=`, never
  `Last-Event-ID` (`gateway/mod.rs:986`).
- **Divergence.** (a) an `abstract.status` or progress event closes a wait; (b) one slot, no
  tree-wide search, no check that the ledger mentioned a snapshot's wait; (c) no ledger terminal
  records; (d) `$slim` unresolved; (e) no execution identity.

#### AbstractAssistant (desktop Python/Qt, iPhone, watch)

Desktop paths are under `abstractassistant/abstractassistant/`; the iPhone under
`abstractassistant/phone/iphone/Packages/AssistantKit/Sources/AssistantKit/`; the watch under
`abstractassistant/watch/apple/Sources/App/`; iPhone and watch share the Swift gateway client
`abstractassistant/watch/apple/Packages/AbstractGatewayClient/Sources/AbstractGatewayClient/`
(written `GC/` below).

- **Desktop, live turn** (`GatewayWorker` -> `GatewayRunController`). Reads `GET /ledger?after&limit=2000`
  (`gateway/run_controller.py:130-150`), then `ledger/stream`; on stream idle `GET /runs/{id}`
  (`:257-318`). One run at a time on a stack: a `subworkflow:` wait pushes the child and the
  parent's ledger is not read again until the child completes (`:113-128`, `:385-432`). Terminal is
  the snapshot only, then a drain until the ledger stops growing (`:152-200`, `:306-316`); a follow
  that ends without a terminal status is reported as "completed" (`ui/gateway_worker.py:1259-1261`).
  Interactions come from each record's `result.wait` (`gateway/events.py:139-154`), deduped per
  (wait_key, step_id) and cleared by a resume record (`gateway/adapter.py:249-255`, `:296-337`).
  When another client answers, nothing closes the open approval sheet (`app.py:11086-11120`; only
  the attention banner is re-checked, `:6827-6858`). The `threading.Event`s named in assistant 0008
  are set but no longer awaited (`ui/gateway_worker.py:120-123`, `:152`, `:186`); answers go out on
  a background thread (`:810-833`).
- **Desktop, rows and text.** Activity = one cycle per STARTED `llm_call` with `node_id == "reason"`
  plus STARTED `tool_calls` (`gateway/adapter.py:133-215`, `ui/activity.py:302-326`); Agent nodes,
  `start_subworkflow` and structured-output calls get no row. Tool results are read from the
  COMPLETED record's `effect.payload.tool_calls` (`gateway/history_seed.py:398-402`): a `$slim`
  pointer there is not a list, so the result is dropped and the tool never finishes on screen. The
  final answer is any record with `result.output` of the root or the followed run
  (`gateway/events.py:73`, `gateway/adapter.py:275-285`, `ui/gateway_worker.py:1020-1029`); local
  fallback texts come from `ui/gateway_worker.py:742-805`.
- **Desktop, attach and history.** `history_bundle` (subruns, tail 2,000) first
  (`ui/gateway_worker.py:1150`), then `GET /runs/{id}` (`:1151`), then seed and follow from cursor 0
  (`:1152-1161`, `:1242`). Answered waits: the Nth waiting record of a key is answered by the Nth
  resume of that key (`:385-428`); no later-step rule. Pending wait: the root snapshot's `waiting`
  first, through a synthetic record (`_maybe_emit_pending_wait`, `:510-520`), else the newest wait
  record of each ledger without a resume check (`:457-479`), else `GET /runs/{id}` per bundle run
  (`:481-503`): a snapshot wait is adopted even when the ledger resolved it. The transcript is the
  bundle's session turns plus tool cards from the ROOT ledger only (`gateway/history_seed.py:67-100`,
  `:459-490`), so tools run by the agent child are lost on every reseed;
  `LLMManager.replace_gateway_messages` replaces the transcript wholesale
  (`core/llm_manager.py:258-286`, and `sync_session_from_gateway`, `:647-703`), and the reseed is
  skipped once a final answer was seen on the stream (`ui/gateway_worker.py:1252-1258`), so local
  state stands. Assistant 0008's other finding (local messages sent as workflow input) no longer
  holds: no messages are passed to `build_run_input_data` (`ui/gateway_worker.py:1198`,
  `gateway/run_input.py:94`, `:148`).
- **iPhone** (`AssistantStore.swift`). Every 3 s `followActiveRun` (`:200-216`, `:568-633`) reads
  `GET /runs/{id}`, walks the tree by `waitingOnChild` and `GET /runs?parent_run_id`, hydrates each
  child (`:970-1007`), streams `ledger/stream` for at most 4 runs or pages `/ledger`
  (`:1009-1068`); `openSession` reads `/runs?session_id&root_only` and per turn `/runs/{id}` and
  `/input_data` (`:762-827`), never the bundle. Status: the snapshot (`:594`); a timer shows as
  running (`:614-618`). Interaction: every `personWait` snapshot in the tree, not the newest one
  (`:583`; `GC/Models.swift:249-266`); answers from elsewhere found by a 1 s `GET /runs/{id}` poll
  per pending wait (`:221-243`); locally granted tools auto-approved (`:854-909`). Rows
  (`RunActivity.swift`): a "Waiting" row closes on a resume of the same node or any later
  non-waiting record of the run, observability included (`:64-96`); one row per `step_id`, so an
  Agent node is several rows (`:79-111`); tool rows keyed by call id else `step:index`
  (`:113-118`); `$slim` yields no calls, and `finish()` marks the row completed without a result
  (`:181-187`). The reply is the snapshot's `conversationText` (`AssistantStore.swift:589-592`).
- **Watch.** Waits from `GET /runs?status=waiting` filtered to reason `user`, at most 12, then
  `GET /runs/{id}` (`GC/GatewayClient.swift:353-362`): no ledger check, and an approval in a child
  run of a different reason is not listed. `AskView.follow` polls `GET /runs/{id}` every second and
  walks `waitingOnChild` to depth 4 (`Ask/AskView.swift:783-835`); the Runs screen labels any
  waiting turn "Needs you" (`Runs/RunsView.swift:290`), timers included. Wait identity is
  `runId:waitKey` with no step (`GC/Models.swift:112`), and `decidedWaits` answers a repeat of the
  same key as already decided (`Shared/AppModel.swift:419`, `:756`, `:774-781`): a second approval
  on a reused key is skipped and the run stalls. Activity (`Shared/RunActivity.swift:81-116`) reads
  the leaf run only, one `/ledger?limit=25` page per tick (`GC/GatewayClient.swift:884-888`); it
  keys steps by `idempotency_key` (`GC/Activity.swift:167`) and resolves `$slim` from the STARTED
  record with the same key (`:194-195`), the closest of all clients to the reducer.
- **Conversation lists** (gateway 0939, listed for completeness): desktop pages of 200 root runs
  until 100 sessions (`gateway/client.py:540-561`, `core/gateway_sessions.py:58-59`), worst status
  of any turn wins (`:65-72`, `:159-215`); iPhone and watch `SessionSummary.group`
  (`GC/Models.swift:1527-1546`) over 60 and 80 runs (`AssistantStore.swift:114`, `:261`;
  `Shared/AppModel.swift:58`, `:188-189`, `:608`), `anyTurnNeedsPerson` checks reason `user` only.
- **Divergence.** (a) desktop: the Nth-resume pairing and no later-step rule; phone: observability
  closes a wait. (b) desktop: the root snapshot's wait first and adopted even when resolved; phone:
  every person wait at once; watch: reason `user` only and no step identity. (c) every path: the
  snapshot only, and the desktop turns an interrupted follow into "completed". (d) desktop and
  phone drop `$slim` tool results; the watch resolves them. (e) desktop: no node rows; phone: one
  row per step; watch: by idempotency key.

#### AbstractObserver, AbstractFlow, AbstractNote, AbstractEntity

- **AbstractObserver** (`abstractobserver/src`). Attach (`connect_to_run`, `ui/app.tsx:2719`):
  `GET /runs/{id}` (`:2771`) and `/input_data`; `/ledger?after&limit=200` pages from 0 until empty
  (`replay_ledger`, `:2694-2717`); `history_bundle?include_subruns&ledger_mode=tail&ledger_max_items=1`
  only to discover child runs (`:2817`); then a loop of `/ledger` pages and `ledger/stream?after=`
  (`:2873-2891`). In parallel: `GET /runs/{id}` every 2 s (`:1800-1822`), one SSE stream for the
  followed child (`:4196-4250`), `POST /runs/ledger/batch` every 2 s for the other children
  (`:4253-4296`). The Discuss chat is panel-chat's `WorkflowChat` (`ui/automation_discussion.tsx:35-41`,
  `:162`); everything else is Observer's own fold.
  Status: the snapshot (`ui/app.tsx:4505`, `:3817`); no ledger terminal record read. A wait is open
  when the root's LAST record is waiting (`:991-992`, `:3818-3819`, `lib/runtime_extractors.ts:3-8`),
  so any later record, observability included, ends it. The wait modal reads the root's last record
  only and does not follow a `subworkflow:` wait to the child's approval (`ui/app.tsx:3825-3835`);
  the board's `state_line` recurses into the live child but labels a parent whose child needs the
  person "waiting", not "needs you" (`ui/state_line.ts:464-536`, `:524-535`). Steps
  (`ui/run_steps.ts:244`) merge STARTED and terminal records by `step_id`, tool calls by
  idempotency key (the approved-call fix, `:226-237`), and resolve `$slim` against the STARTED
  payload with a sha256 check (`:76-140`, applied at `:272`). No execution identity: an Agent's
  structured-output call, `system:completion:*` and progress ticks are not joined to their
  execution (`ui/run_step_kinds.ts:69-71`).
- **AbstractFlow** (`abstractflow/src`). Live (`hooks/useWebSocket.ts`): `POST /runs/start`, then
  `ledger/stream?after=0` (`:752-803`) and one stream per `sub_run_id` into the same state
  (`:596-605`, `:620-666`); `GET /runs/{id}` after `done` (`:787-799`, `applyRunSummary` `:672-740`).
  Past runs (`components/Toolbar.tsx:615`): `history_bundle?include_subruns&ledger_mode=full`,
  re-folded every 2 s while active (`:741-757`), all ledgers merged by timestamp string, not cursor
  (`:688`), the snapshot's wait pushed last even when the ledger answered it (`:708-721`). Status:
  the snapshot. Interaction: the newest `flow_waiting` event, child waits included, a delegation
  never overriding an approval (`hooks/useWebSocket.ts:415-490`, `:425-433`); a timer clears only on
  a ROOT `node_start` or progress tick (`:351-361`, `:402-413`, `:588-596`), and a child's next
  `node_start` is dropped by the root guard (`:352`), so a child approval answered elsewhere stays on
  screen. `$slim` is not handled anywhere in `src`: tool badges read `effect.payload.tool_calls`
  (`components/RunFlowModal.tsx:5277-5281`). Rows: `run:exec:<execId>`
  (`components/RunFlowModal.tsx:3128-3129`) from `executionIdFor` (`utils/ledgerEvents.ts:64-112`),
  the origin of `replay/executions.py`.
- **AbstractNote** (`abstractnote/web/src/lib/runs.ts:116-176`, used by Remember and Capture):
  follows a run it started until terminal through `ledger/stream?after=0`, then `GET /runs/{id}`;
  falls back to `/ledger` pages. No wait notion: a run parked on a question, an approval or a timer
  polls with no UI. Progress lines read `effect.payload.tool_calls` (`:46-49`), so `$slim` yields no
  line. Only panel-chat presentation components are used.
- **AbstractEntity** (`abstractentity/src`): renders no run view and reads no run ledger; it follows
  runs with `GET /runs/{id}` until terminal (`stream_source.ts:853-858`, `flow_lane.ts:34-74`), a
  waiting run counting as unfinished until a 600 s deadline (`use_flow_conversation.ts:180`). Its
  "ledger" panel is the entity's own replay stream. Nothing to migrate beyond status; the waiting
  case should read `interaction` from the view.

#### Gateway consoles

- **Web console** (`abstractgateway/src/abstractgateway/console.py`, inline JavaScript): lists
  `GET /runs?...&root_only=` (`console.py:8814-8830`), Inspect reads `GET /runs/{id}` (`:8899`) and
  prints `waiting` cut to 160 characters (`:8905`); cancel and `inject_guidance` through
  `POST /commands` (`:8931`, `:8949`). No ledger, bundle or stream read; the run detail is handed to
  the Observer (`/apps/observer/#run/{run_id}`, `:13646-13650`).
- **Terminal console crate** (`abstractgateway/console-tui/src`): `GET /runs` (`api.rs:1109-1127`),
  the admin per-runtime list (`api.rs:1131-1151`), `GET /runs/{id}` (`api.rs:1159`), commands
  (`worker.rs:3245-3290`); the docs Q&A polls `GET /runs/{id}` every 2 s (`ui/docs.rs:16`,
  `worker.rs:2703`); `waiting` shown as JSON in Inspect (`store.rs:3196`, `ui/runtimes.rs:1647-1663`).
  No ledger read.
- **Divergence.** Both show the snapshot's status as is: a run past a timer without a resume
  record, or between a resume append and its save, shows "waiting"; neither names the child that
  holds a question. They need `status`, `display_status`, `interaction` and `runs[]` from the view,
  nothing else.

### Summary of the decisions each path makes differently

| Decision | Reducer (`replay/view.py`) | Paths that decide differently today |
| --- | --- | --- |
| When a wait ends | a `resume` for its key, or any later non-observability step of the same run | panel-chat: only `resume` with `resumed: true`; Assistant desktop: the Nth resume of the key; Code TUI, Observer, iPhone: any later record, observability included; Flow: a root `node_start` or progress tick; gateway attention stream: the snapshot |
| Which wait is the interaction | newest open `tool_approval` / `ask_user` in the tree by (time, seq, run id); never a delegation or timer; a snapshot wait only if the ledger never named its key | panel-chat: latest arrival, event waits included; Code TUI: one slot, newest; Assistant desktop: the root snapshot's wait first, adopted even when resolved; iPhone: every person wait in the tree; watch: reason `user` only, no step identity; Observer modal: the root's last record, children not followed; Flow: newest event, snapshot wait pushed last on history; consoles: none |
| Tool approval vs question | `details.kind == "tool_approval"`, `details.mode == "approval_required"`, or a `tool_approval:` key | gateway `events_stream.py:848` and runtime `automations/attention.py:180`: `mode` only; panel-chat `isToolApproval` (`workflow_runtime.ts:583-586`): `mode` or `kind` |
| Event waits | `event`, listed in `open_waits` (`person_facing: false`) | panel-chat: a "Send event" interaction; gateway and runtime attention: a "question" when it has a prompt or choices |
| Run status | terminal snapshot, then ledger terminal record, then open wait, then running | every client: the snapshot only (Code TUI: also an answer-shaped flow end; Assistant desktop: an interrupted follow reported as completed) |
| `$slim` | resolved against the STARTED record of the same step; unresolved pointers dropped and named in `warnings` | panel-chat and Code TUI: fall back to `wait.details.tool_calls` or `result.results`; Code web: keeps the earlier concrete payload; Observer steps and the watch: resolve it; Assistant desktop, iPhone, Flow, Note: the tool result is lost; the bundle: left to the client |
| Activity identity | one per node execution (`executions.py`) | Flow: `executionIdFor` (the origin); Code web: one row per effect type and key; Observer, watch: by step id and idempotency key; iPhone: one row per step; Assistant desktop, Code TUI: no node rows |
| Cursor | per-run `seq` | every stream and page: a record index |
| Timed reads | none: the view stream is driven by ledger appends | panel-chat: `GET /runs/{root}` every 2 s; Observer: `GET /runs/{id}` and `POST /runs/ledger/batch` every 2 s; Flow: the history bundle re-folded every 2 s for an active past run; iPhone: the tree every 3 s and `GET /runs/{id}` every 1 s per pending wait; watch: `GET /runs/{id}` every 1 s; gateway terminal console docs Q&A: every 2 s |
| Run tree | ledger links (`start_subworkflow` results, `subworkflow:` keys, wait details) outrank snapshot `parent_run_id` | bundle and `run_after`: `list_children`; panel-chat, Code TUI, Flow: ledger links; Assistant desktop: one followed run at a time; iPhone, watch: `waitingOnChild` snapshots; Observer: the bundle, one streamed child and a 2 s batch poll for the rest |

## Decision

Taken; recorded here, not re-opened.

- **The reducer lives in AbstractRuntime** as `abstractruntime.replay`: pure, deterministic
  `replay(records_by_run, runs_by_id) -> ReplayView`; incremental `apply(view, new_records)`, whose
  delta is `view.last_delta`; versioned JSON `replay_view_v1` (`ReplayView.to_json()`); the client
  side of a delta in `replay/delta.py`. `replay(all records) == apply` record by record, by
  construction (`replay` is `apply` over every record).
- **AbstractGateway serves the projection** (gateway 0943):
  - `GET /api/gateway/runs/{run_id}/view`: the view; `ETag` = the view's `cursor.token`; a
    matching `If-None-Match` answers 304.
  - `GET /api/gateway/runs/{run_id}/view/stream` (SSE): one `view` event, then `delta` events
    produced by the runtime's `apply`; the event `id` and `Last-Event-ID` are the cursor token;
    driven by the ledger append subscription of the runs in the tree, no timed re-parse.
  - A per-run view cache, advanced by `apply` on append (never rebuilt from scratch while it is
    warm), dropped when the run tree changes in a way `apply` cannot fold.
  - Authorization as `GET /runs/{id}`.
- **Clients act through the existing routes**: `POST /api/gateway/commands` (the view's
  `interaction.command` is the resume to send, the client adds the answer under
  `payload.payload`) and `/runs/start`. The view adds no command route.
- **The durable cursor is the per-run ledger `seq`.** JSONL persists it on append (under the
  advisory lock of 82c0c74), SQLite has it in the table, `list_after` exists on every store,
  `import_records` keeps it across a file to SQLite migration (the gateway's `migrate.py` moves to
  it). The view's `cursor.token` is `<run id prefix>:<seq>` per run of the tree, joined by `.`.

Trade-offs against the three options of 0993:

- (a) **Server-side projection** — chosen as the delivery: clients become thin, the Rust and Qt
  clients need no reducer port, the gateway can cache and index (one fold per run tree shared by
  every viewer instead of one fold per client), and storage encodings (`$slim`, windows, index
  cursors) stay server-side. Cost: the gateway holds a view per watched run tree, bounded by the
  cache.
- (b) **One shared client reducer package** (TypeScript plus Python/Rust ports): rejected. Three
  implementations to keep equal is the problem this item removes.
- (c) **Both** (server authoritative, client reducer for latency): rejected for now. The view
  stream's deltas are the live path; `llm.delta` live text keeps flowing on the ledger stream as
  today. Offline replay of an exported ledger is served by the same Python reducer
  (`replay(records_by_run, runs_by_id)`), not by a client port. Revisit only with a measured latency
  gap that the delta stream cannot close.

## The view model (`replay_view_v1`, as implemented in `replay/view.py`)

Top level (`ReplayView.to_json`):

- `version`: `"replay_view_v1"`.
- `root_run_id`.
- `workflow`: what the root run executes, `{id, bundle_id, bundle_version, flow_id, entrypoint}`
  (`workflow_ref`: `bundle_id@version:flow_id`, or a non-bundled id as its own flow; `entrypoint`
  is the flow the root was started at).
- `session_id` (from the root snapshot).
- `status` (`running | waiting | completed | failed | cancelled | unknown`) and `display_status`
  (the latest `abstract.status` text, never a terminal word).
- `cursor`: `{token, records, runs}`:
  - `token` is header-safe: `<run name>:<seq>` per run, joined by `.`. A run's name is the
    shortest run-id prefix (8 characters at least, letters, digits, `_` and `-`) that no other run
    of the tree shares, and `resolve_cursor_token` maps it back. The token serves as an ETag and
    as an SSE id.
  - `records` is the number of folded records.
  - `runs` maps each run id to its last folded `seq`.
- `runs[]`: the tree in path order (`order` is the path from the root; a child follows its
  parent, and siblings appear in the order they were started): `run_id`,
  `parent_run_id`, `workflow_id`, `workflow` (as above, without `entrypoint`), `status`,
  `display_status`, `depth`, `order`, `waiting` (`{kind, wait_key, child_run_id}` or null),
  `holds_interaction`, `records`, `first_seq`, `last_seq`.
- `interaction`: the one person-facing wait, or null. Its fields:
  - `kind` (`tool_approval | ask_user | event`), `run_id`, `wait_key`, `step_id`, `seq`;
  - `prompt`, `choices`, `allow_free_text`;
  - `details` (without `tool_calls`), and `tool_calls` (resolved, never a `$slim` pointer);
  - `activity_id`;
  - `answer_payload` (the resume payload's shape, from `automations.attention.ANSWER_PAYLOADS`);
  - `command` = `{type: "resume", run_id, payload: {wait_key}}`.
- `open_waits[]`: every open wait in the tree. Its fields: `run_id`, `kind`, `wait_key`,
  `step_id`, `seq`, `until`, `child_run_id`, `person_facing`. The kinds are `tool_approval`,
  `ask_user`, `event` (person-facing), and `delegation`, `timer`, `job`, `pause`,
  `automation_wake`, `signal` (not person-facing).
- Each delta (not the view) also carries `changes[]`, the typed changes for push dispatch (root
  1058):
  - every change has `{kind, id, run_id, seq}`;
  - `status` changes add `status` and `previous`;
  - `interaction` changes add `change: opened|closed`, `interaction_kind` and `wait_key`;
  - `run`, `message`, `activity` and `artifact` changes have no extra fields.
- `messages[]`: verbatim text with provenance: `id`, `role`, `kind` (`prompt | answer | question |
  final | event`), `text`, `run_id`, `step_ids`, `seqs`, `artifact_ids`, `source` (`run_input |
  answer_user | wait | resume | emit_event | ledger | run_output`), `order`, and per kind `final`,
  `choices`, `wait_key`, `level`.
- `activities[]`: one per node execution: `id` (`<run>:<execution id>`), `run_id`, `execution_id`,
  `node_id`, `kind` (`llm | tool | subworkflow | timer | event_wait | ask_user | answer | step`),
  `effect_type`, `name`, `status` (`running | waiting | completed | failed | cancelled | denied`),
  `started_at`, `ended_at`, `seqs`, `step_ids`, `order`; per kind `tool_calls[]` (`id`, `name`,
  `arguments`, `status`, `output`, `error`), `provider`, `model`, `usage`, `child_run_id`, `wait`
  (`{kind, wait_key, until, ended_by: resume | later_step | terminal}`), `progress`,
  `denied_reason`, `error`.
- `artifacts[]`: every `{"$artifact": id}` reference: `id`, `run_id`, `step_id`, `seq`, `path`.
- `usage`: `{input_tokens, output_tokens, total_tokens, llm_calls}` over the tree.
- `warnings[]`: what the view could not fold (a windowed history, an unresolvable `$slim`
  pointer), as sentences.

Rules (module docstring and code of `replay/view.py`):

1. **Run tree.** Children come from delegation waits (`subworkflow:<id>`, `sub_run_id` /
   `child_run_id` in wait details), `start_subworkflow` results, and snapshot `parent_run_id`. A
   ledger link (seq > 0) outranks a snapshot link, so a run's place does not depend on read order.
2. **Wait end.** A wait is over when its run records a `resume` for its key or any later step of
   the same run; `abstract.status` and `abstract.progress` events are not steps. A terminal record
   also closes it. `activities[].wait.ended_by` records which.
3. **Interaction.** The newest open person-facing wait (`tool_approval`, `ask_user`, or an `event`
   wait carrying a prompt or choices; `wait_kind` agrees with `automations/attention.py`, pinned
   by a test) anywhere in the tree, ordered by (record time, seq, run id). A delegation, timer,
   job, pause, automation wake or prompt-less event (`signal`) never is one; none once the root is
   terminal. A snapshot's wait is adopted only when the run's ledger never mentioned its key (a
   windowed or empty ledger).
4. **Status.** A terminal snapshot wins; then the ledger's terminal record
   (`system:terminal_status:<status>`, `system:completion:*`, a run-level failure); then an open
   wait; then running. A snapshot that still says "waiting" for a key the ledger resolved reads as
   running (the resume-before-save window).
5. **`$slim`.** Resolved against the STARTED record of the step the pointer names (an approved
   tool call completes under a new step id pointing back at the waiting step), only for the payload fields
   the view reads (`tool_calls`, `prompt`, `message`, `text`, `content`, `provider`, `model`,
   `workflow_id`); an unresolvable pointer is dropped and named in `warnings`.
6. **Executions.** `replay/executions.py`: same step; a resume joins the execution whose wait it
   answers; a repeated idempotency key joins the node's latest execution; an Agent's
   `_temp.agent.<node>.structured` call and `system:completion:*` join the latest execution;
   progress ticks join the step they report on and never start one.
7. **Messages.** The root's input prompt from its snapshot; `answer_user` text; ask-user questions
   and their answers; `abstract.message` events; the final answer from the root's
   `system:completion:*` record, else from a completed snapshot's output, always as
   `<root>:final`. A final answer equal to an earlier answer names it in `repeats`, and clients
   show the text once.
8. **Idempotence and order.** Records carry their `seq`; a record at or below a run's folded seq is
   ignored. Records of different runs may arrive in any order. `apply_delta(old, diff_views(old,
   new)) == new`; list items are never removed from a view, only changed.

Open points the conformance suite must settle before 0.14.0 (found by this inventory):

- Settled (2026-10-27): an `event` wait with a prompt or choices is person-facing. The reducer's
  `wait_kind` agrees with `automations/attention.py` (`pause`, `automation_wake` and `signal` are
  not person-facing), and `tests/test_replay_conformance.py` compares the two on every wait of
  every fixture. Still open: the gateway's attention stream keeps its own copy
  (`events_stream.py:848-852`) and should call the runtime's classifier.
- Settled: the raw ledger stream's `id:` ("records consumed") equals the record's `seq`, because
  every store now numbers records densely from 1 (a pre-seq JSONL line is numbered by its
  position). The view stream uses cursor tokens.

## Scope

### In (abstractruntime)

- The reducer `abstractruntime.replay` (view, executions, delta), exported from the package root,
  documented in `docs/` (architecture page and API reference).
- The durable cursor: `seq` on every record of every store, `list_after` on every store,
  `import_records` keeping `seq`, `seq` passed to append subscribers.
- One wait classifier: `replay.wait_kind` used by `automations/attention.py` (or the reverse), so
  the runtime has one answer to "is this wait a person's".
- The conformance fixtures: redacted real ledgers plus run snapshots with the expected
  `replay_view_v1`, shipped as package data so the gateway and clients can run them; cases below.
- ADR 0010 (one replay, many views; the cursor is `seq`; clients render the view).

### In (abstractgateway, tracked by 0943)

- `GET /runs/{id}/view` (ETag, 304), `GET /runs/{id}/view/stream` (SSE `view` then `delta`,
  `Last-Event-ID`), the per-run view cache invalidated on append, authorization as `GET /runs/{id}`.
- `migrate.py` through `import_records`; 0082 and 0075 closed against this item (0075's text
  corrected: the tail is already event-driven).
- A benchmark: view build and `apply` cost on the largest ledgers of the operator's gateway, cache
  hit rate, one fold per tree whatever the number of viewers.

### In (clients, each in its own repo, in this order)

1. panel-chat (`WorkflowSessionController` renders the view; `ingestHistory`, `foldRecord`,
   `adoptDurableWait`, `resolvedWaits`, `foldWorkflowTools` and the 2 s lifecycle poll go), then
   AbstractCode web (`activity_rows` reads `activities[]`; delete `wait_resolution.ts`,
   `history_bundle_seed.ts`, `subworkflow_follow.ts`).
2. AbstractAssistant desktop (`GatewayRunController` and `GatewayWorker` follow the view stream;
   assistant 0008's replay findings close with it), then its iPhone and watch apps.
3. AbstractObserver and AbstractFlow run views (Flow's `executionIdFor` is replaced by
   `activities[].execution_id`).
4. The AbstractCode terminal client (the fold in `transcript.rs` and the bundle merge in
   `runner.rs`).
5. The gateway web console and terminal console (status, interaction, run tree only).

Each client deletes its private reconstruction once it passes the conformance suite; a client
keeps only what it decides to show and how it forwards a person's or an entity's request.

### Out

- The conversation list and its counts (gateway 0939).
- Live token text (`llm.delta` frames stay on the ledger stream).
- Any change to the ledger's append-only semantics or record format beyond `seq`.
- A client-side reducer (option c), until a measured need.

## Acceptance criteria

- [x] `abstractruntime.replay` exported; `replay(all) == apply` record by record on every
      conformance fixture; `apply_delta(old, diff_views(old, new)) == new`.
- [x] Every store returns records with their per-run `seq`; `list_after(after=seq)` returns exactly
      the records after it on JSONL, SQLite and in-memory; a JSONL ledger migrated to SQLite keeps
      every `seq`, so a cursor held before the migration resumes at the same record.
- [ ] Conformance fixtures with expected views. Fourteen cases ship in
      `abstractruntime.replay.conformance`: a, a2, b-h, i-l. Still missing: a question in a
      child, a snapshot read between resume and save, a reconnect at a cursor (covered by the
      gateway tests, not by a fixture), and records of a child arriving before its parent's link
      (covered by the invariants' interleavings). The cases this item asks for are: the 2026-09-29 delegated approval
      (`run_approval_subrun.json`), a question in a child, a parallel tool batch, an approved
      batch (new step id, same idempotency key), a denied batch, a timer wait without a resume
      record (2026-10-04), an Agent node with its structured output (2026-10-11), a cancellation,
      a windowed history with an unresolvable `$slim`, a snapshot read between resume and save,
      a reconnect at a cursor, records of a child arriving before its parent's link.
- [ ] One wait classifier in the runtime; the gateway attention stream uses it.
- [x] ADR 0010 accepted; `docs/` describes the view and the cursor (`docs/replay.md`, `docs/api.md`).
- [ ] Gateway 0943 done: `/view`, `/view/stream`, cache, benchmark; `migrate.py` keeps `seq`
      (implemented on `feat/replay-view`, merged with the gateway's 0.16.0 line).
- [ ] panel-chat and AbstractCode web render the view and pass the fixtures; their private folds
      are deleted (the other clients follow in their own releases; each is tracked in its repo).
- [ ] Two clients on different machines show the same interaction for the 2026-09-29 scenario
      and either can answer it.

## Validation

Runtime (this repo):

- `tests/test_replay_conformance.py`: every fixture folds to its expected `replay_view_v1`.
- `tests/test_replay_invariants.py`: `replay(all) == apply` in any split; the same records in any
  cross-run order give the same view; folding a record twice changes nothing; `diff_views` /
  `apply_delta` round trip; the incident rules each go red when the rule is removed.
- `tests/test_ledger_cursor_invariants.py`: `seq` and `list_after` on every store; JSONL `seq` across
  processes (`tests/test_jsonl_ledger_seq_cross_process.py`); `import_records` keeps `seq`.

Gateway (0943):

- `tests/test_gateway_replay_view.py`: `/view` equals the runtime's `replay` of the same stores;
  ETag and 304; `/view/stream` sends `view` then `delta` events whose application equals a fresh
  `/view`; reconnect with `Last-Event-ID`; authorization as `GET /runs/{id}`.
- `tests/test_migrate_preserves_ledger_seq.py`: a JSONL ledger migrated to SQLite keeps every
  `seq` and a held cursor.

Clients: the conformance fixtures in each client's CI once migrated; panel-chat's
`scripts/check_approval_sync.mjs` and AbstractCode's `web/e2e/approval_sync.spec.ts` kept green on
the view; a manual check on the operator's gateway from two machines after the upgrade.
