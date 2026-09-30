# API reference

This document summarizes the **public Python API** of AbstractRuntime and points to the **source of truth in code**.

Public exports live in `src/abstractruntime/__init__.py`. If you are unsure what is supported for external use, start there.

Stability guideline:
- Prefer imports from `abstractruntime` (package root) and `abstractruntime.storage`.
- Deep imports from `abstractruntime.core.*` / `abstractruntime.storage.*` are fine for advanced use, but treat them as lower-stability unless they are explicitly documented/re-exported.

## Recommended imports

Core kernel:

```python
from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
```

Storage helpers (common stores):

```python
from abstractruntime.storage import (
    InMemoryLedgerStore,
    InMemoryRunStore,
    JsonFileRunStore,
    JsonlLedgerStore,
)
```

Scheduler convenience wrapper:

```python
from abstractruntime import create_scheduled_runtime
```

AbstractCore integration (included in the base `abstractruntime` install):

```python
from abstractruntime.integrations.abstractcore import (
    ApprovalToolExecutor,
    MappingToolExecutor,
    ToolApprovalPolicy,
    create_local_runtime,
)
```

See also: `getting-started.md` (end-to-end runnable examples).

## Core types (durable workflow semantics)

Implementation: `src/abstractruntime/core/models.py`, `src/abstractruntime/core/spec.py`.

- `WorkflowSpec`: in-memory workflow graph (`workflow_id`, `entry_node`, `nodes`).
- `StepPlan`: node return value (what happens next): `effect`, `next_node`, or `complete_output`.
- `Effect` / `EffectType`: durable side-effect request protocol (the runtime mediates execution).
- `RunState` / `RunStatus`: durable checkpoint for a run, persisted by a `RunStore`.
- `WaitState` / `WaitReason`: durable pause metadata for `WAIT_*` / `ASK_USER` / passthrough tool waits.

Durability invariant: `RunState.vars` must remain JSON-serializable (`src/abstractruntime/core/models.py`). For large payloads use artifacts/offloading (`src/abstractruntime/storage/artifacts.py`, `src/abstractruntime/storage/offloading.py`).

## Runtime (start / tick / resume)

Implementation: `src/abstractruntime/core/runtime.py`.

- `Runtime.start(workflow, vars=..., actor_id=..., session_id=..., parent_run_id=..., run_id=...) -> run_id`
  - creates and persists a new `RunState`
  - `run_id` starts the run under an id you choose (`[A-Za-z0-9_-]+`). The run is created only if that id is free; starting it again with the same identity (workflow, session, parent, `vars._meta.occurrence`, `vars._meta.creation_digest`) returns the existing run untouched, and a different identity raises `RunIdentityConflict` (a `ValueError`, `reason_code = "identity_conflict"`). The store must support `create_if_absent` (otherwise `NotImplementedError`). `START_SUBWORKFLOW` accepts the same explicit id as `payload.run_id`
  - a root run that names a session is checked against the session's attribution: in a discussion session it gets the discussion's workspace setup (its own writable workspace and the read-only mount) and provenance, and when the attribution cannot be read (an invalid discussion root, or a store without a run index) the start raises `SessionAttributionError` (see [automations.md](automations.md#discussions))
- `Runtime.tick(workflow, run_id, max_steps=..., step_gate=None) -> RunState`
  - executes node handlers and effects until the run becomes `WAITING`, `COMPLETED`, `FAILED`, or `CANCELLED`
  - optional `step_gate()` is consulted at every step boundary; when it returns `False` the tick returns the persisted state and the run stays `RUNNING` (a later tick continues where it stopped)
- `Runtime.resume(workflow, run_id, wait_key, payload, max_steps=...) -> RunState`
  - validates the `wait_key`, writes `payload` to `WaitState.result_key` (if set), and continues from `WaitState.resume_to_node`
  - a wait is resumed at most once: a resume of a run that is no longer waiting, or that waits on another key, raises `StaleResumeError` (a `ValueError`), for example when another caller resumed it first
  - the check and the commit run under a per-run lock, and tools approved with `{"approved": true}` execute while that lock is held, so a long tool delays a competing resume of the same run, which is then refused
- `run_mutation_lock(run_id)` (package root): the per-run, per-process, re-entrant lock that `tick` holds for the whole tick and `resume` for its commit; take it around your own read-modify-save of a run so a tick cannot overwrite your change
- `Runtime.get_state(run_id) -> RunState` and `Runtime.get_ledger(run_id) -> list[dict]`
  - host-facing read APIs for checkpoints and the append-only ledger
- `Runtime.cancel_run(run_id, reason=None, cancelled_by="api") -> RunState`
  - persists `CANCELLED`, then signals every effect of that run (and its in-flight descendants) executing in this process; the running attempt is recorded as `cancelled` (`StepStatus.CANCELLED`, never retried) with `cancelled_by` and `reason`, and no further effect starts
- `Runtime.set_default_provider_model(provider=..., model=...)`
  - re-points the default provider/model seeded into new runs; pair it with the pooled client's `set_default_provider_model(...)` so both agree
- `Runtime.set_live_delta_sink(sink)` where `sink(event: dict) -> None`, or `None` to remove it
  - receives live token deltas (`llm.delta`, `llm.delta_end`) for LLM calls of runs started with `_runtime.stream: true` (a boolean; `start` refuses any other value); nothing is written to the ledger. See [Live token streaming](integrations/abstractcore.md#live-token-streaming)

Effect cancellation helpers (`src/abstractruntime/core/effect_cancellation.py`):
- `inflight_effects(run_ids=None)` lists executing effects (run, step, provider, model, elapsed time)
- `request_model_effects_cancel(provider, model, cancelled_by="model_eject")` stops the effects using a model before it is unloaded; local `unload_model_residency` calls it for you
- `kill_inflight_effect(step_id, killed_by=...)` is an in-process hard stop for a call that ignores its cancel event (it cannot interrupt a thread blocked inside one native call)

Tool scope: an explicit `allowed_tools` list in `run.vars["_runtime"]`, in a child run's `_runtime`, or in a `TOOL_CALLS` payload is a ceiling intersected across the run tree (`src/abstractruntime/core/tool_scope.py`). Approval policy can remove a prompt but can never grant a tool outside it; a missing key means unrestricted and an empty list denies every tool.

For the execution model (ledger records, effect outcomes, waits), see `architecture.md`.

## Scheduler convenience API

Implementation: `src/abstractruntime/scheduler/*`.

Use `create_scheduled_runtime()` for a zero-config wrapper that bundles `Runtime` + an in-process polling `Scheduler`:
- `ScheduledRuntime.run(workflow, vars=..., actor_id=..., max_steps=...) -> (run_id, state)` (`src/abstractruntime/scheduler/convenience.py`)
- `ScheduledRuntime.respond(run_id, payload) -> RunState` (resumes a waiting run using its stored `wait_key`)
- `ScheduledRuntime.stop()` (stops the scheduler thread/loop)

For time-based waits, the scheduler polls due runs via `QueryableRunStore.list_due_wait_until(...)` (`src/abstractruntime/storage/base.py`, `src/abstractruntime/scheduler/scheduler.py`).

## Storage layer (durability backends)

Interfaces: `RunStore`, `LedgerStore`, and `QueryableRunStore` are defined in `src/abstractruntime/storage/base.py`.

Included backends:
- In-memory (tests/dev): `InMemoryRunStore`, `InMemoryLedgerStore` (`src/abstractruntime/storage/in_memory.py`)
- Filesystem:
  - checkpoints: `JsonFileRunStore` (`src/abstractruntime/storage/json_files.py`)
  - append-only ledger: `JsonlLedgerStore` (`src/abstractruntime/storage/json_files.py`)
- SQLite:
  - `SqliteRunStore`, `SqliteLedgerStore` (`src/abstractruntime/storage/sqlite.py`)

Notes:
- `abstractruntime.storage` intentionally exports only the most common store types. SQLite types are available via:
  - `from abstractruntime import SqliteRunStore, SqliteLedgerStore`, or
  - `from abstractruntime.storage.sqlite import SqliteRunStore, SqliteLedgerStore`

Run index and creation (all built-in run stores, including the offloading wrapper):
- `create_if_absent(run) -> (run, created)`: creates a run only if its id is free and never overwrites an existing one; `store_supports_create_if_absent(store)` / `require_create_if_absent(store)` check a store first. Process-crash safe; power-loss durability is not claimed
- `list_run_index(status=, workflow_id=, session_id=, root_only=, limit=, oldest_first=, automation_id=, role=, session_kind=)`: lightweight rows carrying `automation_id`, `role`, `occurrence_index`, `session_kind` and `workspace_root` (the run's top-level `vars["workspace_root"]` as stored, stripped; `None` when absent); the three attribution filters take a value, a comma-separated string or a list; `root_only=True` returns turn roots (parent-less runs except automation controllers, plus automation occurrences)
- `session_kinds(session_id) -> frozenset` and `latest_occurrence_row(automation_id)`: indexed lookups used by session attribution and automation listings
- `JsonFileRunStore.warm_session_index()` builds the session and children indexes at host startup. Store objects and processes sharing one JSON run folder see each other's created and deleted runs through the creation journal `.runs_created.log`; v1 supports one writer process per store

Common decorators:
- `ObservableLedgerStore` for subscriptions (`src/abstractruntime/storage/observable.py`)
- `HashChainedLedgerStore` + `verify_ledger_chain(...)` for tamper-evidence (`src/abstractruntime/storage/ledger_chain.py`)
- `OffloadingRunStore` / `OffloadingLedgerStore` to store large values by artifact reference (`src/abstractruntime/storage/offloading.py`)

## Commands (durable control-plane inbox)

AbstractRuntime ships append-only, idempotent **command inbox** primitives designed for gateways/workers that must accept retries safely:
- models + interfaces: `CommandRecord`, `CommandStore`, `CommandCursorStore` (`src/abstractruntime/storage/commands.py`)
- backends: in-memory + JSONL (`src/abstractruntime/storage/commands.py`), SQLite (`src/abstractruntime/storage/sqlite.py`)

These APIs are exported at the package root (see `src/abstractruntime/__init__.py`).

## Artifacts (store by reference)

Implementation: `src/abstractruntime/storage/artifacts.py`.
Deep dive: `artifacts.md`.

Key types:
- `ArtifactStore` (interface), `InMemoryArtifactStore`, `FileArtifactStore`
- helpers: `artifact_ref(...)`, `resolve_artifact(...)`, `is_artifact_ref(...)`

The store keeps payload bytes out of run state and persists structured metadata:
- `ArtifactDescriptor` is the Runtime-owned descriptor used by Gateway and Observer. It separates `semantic_kind` such as `voice`, `music`, `sound`, or `image` from `render_kind` such as `audio`, `markdown`, `html`, or `json`, and can carry workflow/node/turn links, media facts, generation/provenance data, source refs, security, and action links.
- `ArtifactAccessStats` records explicit metadata/content/preview/download/export actions when HTTP or UI layers call `record_access(...)`. Plain `load(...)` and `get_metadata(...)` remain side-effect free.
- `search(...)`, `count(...)`, `facet_counts(...)`, and `stats(...)` provide metadata queries for host control planes. `FileArtifactStore` serves these from a repairable SQLite catalog when possible, including exact `total`, `total_bytes`, and requested facet counts without forcing Gateway/Observer to load every matching artifact.

Artifacts are used by:
- offloading wrappers (`src/abstractruntime/storage/offloading.py`)
- evidence capture (`docs/evidence.md`, `src/abstractruntime/evidence/recorder.py`)
- AbstractCore media integration: input artifact refs can be materialized for LLM calls, and generated image/video/voice/music/audio outputs are stored as artifact refs

## Snapshots / bookmarks

Implementation: `src/abstractruntime/storage/snapshots.py`.

- `SnapshotStore` interface + `InMemorySnapshotStore`, `JsonSnapshotStore`
- `Snapshot` model (a named bookmark of run state)

Docs: `snapshots.md`.

## Effect policies (retries + idempotency)

Implementation: `src/abstractruntime/core/policy.py`.

- `EffectPolicy` protocol and implementations: `DefaultEffectPolicy`, `RetryPolicy`, `NoRetryPolicy`
- `compute_idempotency_key(...)` helper

Docs: `architecture.md` (reliability section).

## WorkflowBundles (`.flow`) and VisualFlow distribution

Implementation:
- bundles: `src/abstractruntime/workflow_bundle/*`
- compiler: `src/abstractruntime/visualflow_compiler/*`

VisualFlow compiler helpers are available from `abstractruntime.visualflow_compiler`:
- `load_visualflow_json(...)` normalizes VisualFlow JSON into the stdlib model.
- `visual_to_flow(...)` lowers VisualFlow into the internal Flow IR.
- `compile_visualflow(...)` and `compile_visualflow_tree(...)` compile VisualFlow JSON into executable `WorkflowSpec` objects.

VisualFlow authoring note (media and document nodes):
- Runtime recognizes first-class VisualFlow media nodes such as `generate_image`, `edit_image`, `image_to_image`, `upscale_image`, `image_upscale`, `generate_video`, `text_to_video`, `image_to_video`, `generate_voice`, `generate_music`, `transcribe_audio`, and `listen_voice`.
- Generated-media and transcription nodes lower to a durable `EffectType.LLM_CALL` with an `output` selector (for example `{"modality":"music","task":"music_generation"}`), while `listen_voice` lowers to `WAIT_EVENT`. Hosts should persist the authoring node type rather than pre-lowering to `llm_call`.
- Runtime also recognizes file/document nodes. `read_file` and `write_file`
  handle UTF-8 text/JSON workspace paths. In Gateway-hosted runs, those paths
  follow the shared canonical contract: `rel/path` for the main workspace root
  and `mount_alias/rel/path` for approved mounts. `read_pdf` extracts text and
  metadata from PDF paths with `pypdf`; `write_pdf` renders text or
  Markdown-style content to real PDF bytes with `reportlab`; `write_docx`
  renders Markdown-style content to real `.docx` bytes with the standard
  library; `list_folder_files` enumerates workspace-scoped folders with
  family/extension filters;
  `import_workspace_file` snapshots a workspace file into a durable artifact;
  `read_artifact` projects saved file content back out as text/JSON/bounded
  binary metadata; and `export_artifact` writes a durable artifact back to a
  workspace path. PDF/DOCX bytes are written to the workspace path and only
  JSON-safe metadata/path values are stored in run state. In local Runtime-only
  runs with no workspace scope, relative file-node paths still fall back to the
  process working directory.

Public bundle APIs are exported from `src/abstractruntime/workflow_bundle/__init__.py` and re-exported in `src/abstractruntime/__init__.py`:
- open: `open_workflow_bundle(...)`
- registry: `WorkflowBundleRegistry`
- pack/unpack: `pack_workflow_bundle(...)`, `unpack_workflow_bundle(...)`

Docs: `workflow-bundles.md`.

## Automations

Implementation: `src/abstractruntime/automations/*`, `src/abstractruntime/triggers/*`, `src/abstractruntime/automation_queries.py`.
Deep dive: [automations.md](automations.md).

```python
from abstractruntime.automations import (
    apply_automation_command,   # pause / resume / run_now / revise / stop_current / archive
    create_automation,          # -> (automation_id, revision)
    drive_automation,           # standalone run loop
    get_automation,
    list_attention,
    list_occurrences,
    pending_waits,
    register_controller_bundle,
    start_discussion,
)
from abstractruntime.automation_queries import latest_occurrence, list_automations
from abstractruntime.triggers import get_trigger_adapter, trigger_sources
```

- `create_automation(runtime, request, *, now=None, actor_id=None)`: creates the controller root run (the automation) with a deterministic id; raises `AutomationError` (`invalid_definition`, `unsupported_feature`, `unknown_trigger_source`, `identity_conflict`)
- `apply_automation_command(runtime, *, automation_id, command_id, type, payload=None, actor=None, expected_revision=None)`: the only writer of automation state; returns `{status: "applied" | "rejected", error?, duplicate}`; idempotent per `command_id`
- `record_automation_command_result(...)`: records a host-side failure of a command as a rejected result
- reads: `get_automation(run_store, id)`, `list_occurrences(runtime, id, cursor=, limit=)`, `list_attention(ledger_store, id, after_seq=, cursor=, limit=)`, `pending_waits(run_store, id, limit=)` with typed waits (`ask_user`, `tool_approval`, `event`) and `ANSWER_PAYLOADS`; `list_automations(run_store, status=, cursor=, limit=)`, `automation_summary(run)`, `latest_occurrence(run_store, id)`
- `start_discussion(runtime, *, automation_id, occurrence_index, request_id, prompt, workspace_root, actor_id=None)`: a separate conversation forked at any occurrence N, seeded with the automation's whole timeline 1..N, working in its own writable `workspace_root` with the automation's workspace mounted read-only
- controller bundle: `register_controller_bundle(registry)`, `controller_workflow_spec()`, `controller_bundle_path()`; `CONTROLLER_WORKFLOW_ID` is `abstractframework.automation-controller@1.0.0:controller`
- trigger sources: `trigger_sources()`, `get_trigger_adapter(id, version)`, built-ins `schedule@1`, `manual@1` and `email.received@1`, third-party sources through the `abstractruntime.trigger_sources` entry-point group
- definition v2: `policy.email_allowed_recipients` (default `["self"]`), `policy.untrusted_input_tools` (default `[]`) and `notify.channels` (default `["console"]`); attention items carry `channels`

## Email

Implementation: `src/abstractruntime/email/*`, `src/abstractruntime/triggers/email_received.py`. Deep dive: [email.md](email.md).

```python
from abstractruntime.email import (
    EmailBinding, bind_email_account, strip_client_email_keys, binding_of,   # run-scoped binding
    JsonFileEventInbox, InMemoryEventInbox,                                  # durable event inbox
    EmailInboxFeeder, PollReport, email_event_id,                            # mailbox -> inbox
    email_trigger_consumers, wake_email_automations, prune_email_inbox,      # watcher helpers
    EventInboxRetention,                                                     # inbox retention (90 days / 10,000 events)
    EMAIL_USE_AGENT_TOOL, EMAIL_USE_ACTION, email_use_for_workflow,          # who is sending
    email_action_target, register_email_action_workflow, validate_email_action, render_email_template,
)
```

- `Runtime.set_email_context_resolver(fn)`: `fn(binding, *, use) -> EmailContext | None` (`use` is `"agent_tool"` or `"action"`; `fn(binding)` also accepted), per runtime, memory only
- `Runtime.set_email_binding(binding | None)` / `Runtime.email_binding`: the account occurrences are bound to
- `Runtime.set_event_inbox(inbox)` / `Runtime.event_inbox`: required by `email.received@1`
- `EmailInboxFeeder(inbox, account_ref=...).poll(ctx, *, now=None, force=False) -> PollReport` and `.status()`
- toolsets: `get_default_toolsets(..., email_enabled=True)` (also `list_default_tool_specs`, `build_default_tool_map`, `list_tool_catalog`); no env flag; `list_tool_catalog(email_enabled=False, email_off_reason="not_connected" | "admin_disabled" | "not_available" | "agent_tools_off")`
- mail library for hosts: `abstractruntime.integrations.abstractcore.email_facade` re-exports `abstractcore.comms.email` (its `__all__` plus the `legacy` module; `GATEWAY_NAMES` lists the names hosts use)
- approval: the `send_email_recipient@v2` refiner ([tool-approval.md](tool-approval.md#per-call-refiners))
- `adopt_legacy_schedule_projection(run)`: read-only summary of a legacy gateway `scheduled:*` root
- read-only mounts: `_runtime.workspace_read_only_paths` (absolute folders; file write tools and VisualFlow writers refused inside, reads and command/code tools allowed); helpers `read_only_paths(vars)`, `path_is_read_only(vars, path)` and `READ_ONLY_PATHS_KEY` in `abstractruntime.utils.workspace_paths`
- read-only workspaces: run vars `workspace_read_only: true` (or `_runtime.workspace_read_only: true`); `abstractruntime.integrations.abstractcore.tool_effects.TOOL_EFFECT_CLASSES` classifies every exposable tool as `read`, `write`, `exec`, `delegate`, `comms` or `memory-write`

## Sessions and history

Implementation: `src/abstractruntime/session_turns.py`, `src/abstractruntime/session_history.py`, `src/abstractruntime/core/run_attribution.py`.

- `select_session_turns(run_store, session_id, *, include_occurrences=True, until_ms=None, automation_id=None, through_occurrence=None, include_drafts=False, limit=50)` (package root): a session's turns, oldest first. Turns are parent-less runs except automation controllers, plus automation occurrences (a retried occurrence counts once, as its newest attempt); child runs, runtime-internal runs, legacy scheduled wrappers and draft-test runs are left out
- `session_chat_messages(run_store=, ledger_store=, artifact_store=, session_id=, max_tokens=HISTORY_REPLAY_MAX_TOKENS, until_ms=None, exclude_run_ids=None, automation_id=None, through_occurrence=None, strict=False)` (package root): the session's completed turns as user/assistant message pairs; in a discussion session the discussion's seed comes first. Returns a `ReplayedHistory` (a list of messages) whose `.report` says what was replayed and dropped. `strict=True` raises `SessionHistoryError` (`reason_code = "history_unavailable"`) instead of returning a partial history. `max_messages`, `max_total_chars` and `max_chars_per_message` (keyword-only, default `None`) are the caps retired in 0.7.0: they are accepted so hosts built against 0.6 (AbstractGateway 0.6.0) still get their history, and ignored. Passing any of them logs one warning and names them in `report["ignored_inputs"]`
- The history window: `HISTORY_REPLAY_MAX_TOKENS = 50_000` (package root). Replay keeps the most recent turns that fit 50,000 estimated tokens, newest first, as whole messages. There is no message-count cap and no character cap, and no message is ever cut. The fold stops at the first older turn that does not fit, so the window has no gaps. A total of exactly 50,000 tokens fits. One exception: when the newest turn alone is larger than the window, it is kept whole (`oversize_turn_kept: true`), because cutting it would lose content and dropping it would replay nothing. The model can use the rest of its context window; if a turn is too large for the model, the provider reports the error. Tokens are counted with `abstractruntime.memory.token_budget.estimate_message_tokens`: for a content-part list it counts the text parts' text plus `MEDIA_PART_TOKEN_ESTIMATE` (512) for each media part (`MEDIA_PART_TYPES`: image, audio, file), never the part's base64; any other part (a tool result, a thinking block) counts by its text (0.7.1). `max_tokens` must be a positive int
- `ReplayedHistory.report`: `{policy: "most_recent_whole_turns", max_tokens, token_estimator, replayed_messages, replayed_tokens, dropped_messages, dropped_tokens, dropped_counts_complete, oversize_turn_kept}`. Replay stops reading once the window is full, so in a long session `dropped_counts_complete` is `false` and the dropped counts cover only the turns it read. When turns were dropped, the oldest replayed message starts with a `[#TRUNCATION: N earlier message(s) ... were dropped from replay by the history window ...]` line (ADR-0026). Hosts record the report in the run as `vars._runtime.session_history`
- `fold_history_window(pairs, *, max_tokens=HISTORY_REPLAY_MAX_TOKENS)` (package root): the window itself, applied to chronological turns (lists of whole messages); returns `(kept, report)`
- `announce_dropped(messages, report, *, stamp_metadata=True)` (package root): writes the window's `[#TRUNCATION: ...]` notice at the start of the oldest kept message, in place, when `report` says turns were dropped. Use it after your own `fold_history_window` call (for example on client-sent history) so the notice matches session replay. `stamp_metadata=True` also adds `metadata.replay_truncated` and `metadata.history_window` to that message; pass `False` for plain role/content messages. When the message's content is a content-part list (`[{"type": "text", ...}, {"type": "image_url", ...}]`), the notice is added as one `{"type": "text"}` part and every other part is left as it is, so images stay images (0.7.1); a stamped head keeps its `<runtime_metadata>` envelope first. Content of any other type is left untouched and a warning is logged. `_announce_dropped` is kept as an alias of the pre-0.7.0 private name
- `window_transcript(messages, *, max_tokens=HISTORY_REPLAY_MAX_TOKENS, current_turn_start=None)` (package root): the same window over a transcript you already hold (a turn is a user message and the messages after it). `current_turn_start` is the index of the message that opened the turn in progress: from there on everything is one turn, the newest, always kept whole (`oversize_turn_kept` when it alone exceeds the window), whatever user-role messages a loop adds inside it; an index out of range raises `ValueError`. Returns a `ReplayedHistory` of copies with the notice written by `announce_dropped(..., stamp_metadata=False)`. The entity chat driver and the entity visit workflow use it for their prompts. The visit records the report in `vars._runtime.session_history`; the chat driver puts it on `TurnReport.history_window`. On the react visit arm, BRIDGE sets `_runtime.history_window_tokens` (50,000) and `_runtime.history_window_turn_start` (the visitor's message) and the AbstractAgent react loop (0.3.17 or newer) sends `window_transcript(context.messages)` on each call and records the report; the stored transcript stays whole. The report carries `window_applied: true`; with an older AbstractAgent that records no window the turn still completes, a warning is logged and the report is `{"window_applied": false, "reason": "agent_too_old", ...}` (the whole transcript was sent). When the oldest kept message carries a `<runtime_metadata>` envelope, the notice is written after it
- `session_attribution(run_store, session_id)` (`abstractruntime.core.run_attribution`): `None` or `{"kind": "chat" | "automation" | "occurrence" | "discussion", ...}`; a discussion adds its validated root, automation, occurrence and workspace. Raises `SessionAttributionError` when the lookup cannot be completed
- `is_draft_lifecycle(run_lifecycle)` (package root): whether a run's lifecycle value marks a draft test run

History bundles (`export_run_history_bundle`) and session replay use the same turn selection, so automation occurrences appear as turns (kind `occurrence`, with `automation_id` and `occurrence_index`) in both.

## Run history bundle export (portable replay artifact)

Implementation: `src/abstractruntime/history_bundle.py`.

- `export_run_history_bundle(...)`
- `persist_workflow_snapshot(...)`

This produces a portable record of a run’s state + ledger + artifacts suitable for debugging/review.
When available, it also includes `resolved_actions`: bounded capability/action summaries derived
from Core route resolution and persisted from `LLM_CALL` results.

## Runtime-owned integrations

### AbstractCore (LLM + tools)

Requires: `pip install abstractruntime` (AbstractCore 2.20.0 or newer is part of the base install).

Implementation: `src/abstractruntime/integrations/abstractcore/*`.

Entry points:
- `create_local_runtime(...)`, `create_remote_runtime(...)`, `create_hybrid_runtime(...)` (`src/abstractruntime/integrations/abstractcore/factory.py`)
- public discovery facade: `AbstractCoreDiscoveryFacade`, `get_abstractcore_discovery_facade(...)` (`src/abstractruntime/integrations/abstractcore/discovery_facade.py`)
- public host facade: `AbstractCoreHostFacade`, `get_abstractcore_host_facade(...)` (`src/abstractruntime/integrations/abstractcore/host_facade.py`)
- email facade: `email_facade` (AbstractCore's mail library for hosts; see [Email](#email))
- public Telegram host wrappers: `TelegramTdlibNotAvailable`, `bootstrap_telegram_auth_from_env(...)`, `get_global_telegram_client(...)`, `stop_global_telegram_client()`, `send_telegram_message(...)` (`src/abstractruntime/integrations/abstractcore/telegram_facade.py`)
- public durable run facade: `AbstractCoreRunFacade`, `get_abstractcore_run_facade(...)` (`src/abstractruntime/integrations/abstractcore/run_facade.py`)
- effect handler wiring: `build_effect_handlers(...)` (`src/abstractruntime/integrations/abstractcore/effect_handlers.py`)
- tool executors: `MappingToolExecutor`, `AbstractCoreToolExecutor`, `PassthroughToolExecutor`, `ApprovalToolExecutor`, `ToolApprovalPolicy` (`src/abstractruntime/integrations/abstractcore/tool_executor.py`)
- discovery-facade delegation is implemented by the configured AbstractCore LLM clients in `src/abstractruntime/integrations/abstractcore/llm_client.py` (`list_providers`, `list_provider_models`, `get_voice_catalog`, `list_tts_models`, `list_stt_models`, `list_music_providers`, `list_music_models`, `list_vision_provider_models`, `list_cached_vision_models`, `list_vision_adapters`)
- the host's OpenAI credential for AbstractVoice's `openai` engines, `voice_openai_api_key`, travels two ways. For discovery, pass it to `get_voice_catalog`, `list_tts_models` or `list_stt_models`; the local clients hand it to the voice plugin's config, and a remote client never sends it (a remote AbstractCore server uses its own voice configuration). For execution, put it in `create_local_runtime(llm_kwargs={"voice_openai_api_key": ...})`. Constructor kwargs become the provider's `config`, which is also the capability-plugin config. Providers read only the keys they name, so a `voice_*` key changes no text request (tests `tests/test_voice_openai_api_key_path.py`)
- host-facade client delegation is implemented by the configured AbstractCore LLM clients in `src/abstractruntime/integrations/abstractcore/llm_client.py` (`get_prompt_cache_capabilities`, `get_prompt_cache_stats`, `prompt_cache_set`, `prompt_cache_update`, `prompt_cache_fork`, `prompt_cache_clear`, `prompt_cache_prepare_modules`, `upsert_text_bloc`, `get_bloc_record`, `list_blocs`, `get_bloc_kv_manifest`, `ensure_bloc_kv_artifact`, `load_bloc_kv_artifact`, `list_bloc_kv_artifacts`, `delete_bloc_kv_artifact`, `prune_bloc_kv_artifacts`, `delete_bloc`, `get_model_residency_capabilities`, `list_model_residency`, `load_model_residency`, `unload_model_residency`, `get_memory_snapshot`, `list_session_prompt_caches`, `clear_session_prompt_caches`, `lock_model_residency`, `unlock_model_residency`, `get_context_estimate`); the last six are optional in the client contract, and the facade returns `{"ok": false, "supported": false, ...}` when the configured client does not implement one
- switching the default model (`set_default_provider_model` on the pooled client) unloads the previous in-process model unless an owner in the process still uses it; `list_model_residency` diagnostics report `pending_ejects` and `last_switch_ejects`, and `unload_model_residency(runtime_id=...)` accepts any listed `local:<task>:<provider>:<model>` id, including `local:embedding:<provider>:<model>` rows (see `integrations/abstractcore.md#unloading-and-switching-models`)
- `lock_model_residency`, `unlock_model_residency`, and `get_context_estimate` accept an optional payload mapping and/or keyword arguments (keyword arguments win on conflicts); a lock requires provider-verified residency — locking a non-resident pair refuses with a structured `model_not_resident` payload (load with `lock: true` instead; remotely the Core server's HTTP 409 refusal is converted to the same payload), while unlock never requires residency so a locked-but-evicted pair is always releasable; a locked model refuses `unload_model_residency` with a structured `model_locked` payload — locally as a client-side per-`(provider, model)` lock, remotely by converting the Core server's HTTP 409 refusal — and unloads only with `force=true`, releasing the lock only after the unload succeeds
- the `MODEL_RESIDENCY` effect supports `list_loaded`, `load`, `unload`, `lock`, and `unlock` operations with the same soft-fail semantics; on `unload`, `force` is forwarded only when authored in the effect payload
- local `list_model_residency` merges AbstractCore's host-wide loaded-model sweep into text-generation listings (sweep-only rows carry `source: "provider_server"` and no `task` label), and residency rows normalize provider size extras to `size_bytes` / `size_vram_bytes` while keeping the originals; local text rows also carry registry-declared `modalities` (omitted on a registry miss), host identity (`host_id` / `host_name`), and runtime-owned lock truth (`locked` / `lockable`, with `pinned` a truthful alias of `locked` — never the default-identity flag, which `default` alone carries), while remote listings keep the Core server's row identity
- host-local prompt-cache export/import admin also lives on the host facade and client delegation layer (`list_prompt_cache_exports`, `prompt_cache_export`, `prompt_cache_import`) and is intentionally local-only
- host-facade email helpers (`list_email_accounts`, `list_emails`, `read_email`, `send_email`) use the process's own AbstractCore account (single-user installs)
- run-facade helpers create and resume durable child runs for existing runs (`execute_llm_call`, `execute_tool_calls`, `resume_tool_calls`, `generate_image`, `edit_image`, `upscale_image`, `generate_video`, `image_to_video`, `generate_voice`, `generate_music`, `transcribe_audio`, `send_email`, `send_telegram_message`)
- task-specific image/video helpers preserve batch and adapter controls such as
  `count`/`n`, `seeds`, ordered `lora_adapters`, and video `flow_shift`; local
  subprocess isolation stays within the same public contract.

Execution controls and cancellation:
- `params.thinking` (bool or reasoning level) and `params.speculation` (`False`, `True`, or a Core speculation object such as `{"mode": "native_mtp", "num_draft_tokens": 2}`) are forwarded locally and remotely; `run.vars["_runtime"]["speculation"]` sets a run-wide preference inherited by subworkflows, Agent loops and delegated calls, and `False` stays Off across every boundary (see `integrations/abstractcore.md#execution-controls-and-local-concurrency`)
- `get_abstractcore_discovery_facade(...).get_execution_capabilities(model_name=None, provider=None)` asks the actual execution host (local or remote) what it supports, without loading a model
- the `LLM_CALL` handler hands the effect's cancel event to AbstractCore (`generate(..., cancel_event=...)`); remote clients close the request to the AbstractCore server on Stop, which the server treats as a cancel
- every `LLM_CALL` is offered a progress callback; providers that report text phases (prefill / generate / complete) produce `abstract.progress` ledger events with `kind: "llm"`
- `abstractruntime.turn_grounding.stamp_user_turn_grounding(messages, grounding=...)` writes the grounding envelope once into the stored user turn, so the prompt sent to the model stays a byte prefix of the next turn's prompt

`LLM_CALL` payloads are JSON-safe effect payloads. Common fields:
- `prompt`, `messages`, `system_prompt`, and convenience `text`
- `media`: a media path, artifact ref (`{"$artifact": "..."}` or `{"artifact_id": "..."}`), media dict, or list of those
- `output`: AbstractCore output selector; top-level `outputs` is accepted as a runtime alias
- `params`: provider/model routing, generation controls, prompt-cache keys or `prompt_cache_binding`, structured-output schema options, and tracing metadata

Multimodal support:
- common remote-light AbstractCore media, vision, voice, audio, and music dependencies are part of the base Runtime install
- local clients call AbstractCore's unified `generate(..., media=..., output=...)`
- remote and hybrid clients support AbstractCore Server chat media content arrays plus image generation, image edits, image upscaling, text-to-video, image-to-video, speech, music generation, and transcription endpoints; pass an output-specific `model` for remote media provider routing, otherwise the server endpoint can use its configured capability default
- remote transcription requires one audio media item that resolves to a local file path or artifact-backed temporary file
- generated image/video/voice/music/audio bytes require a runtime `ArtifactStore`; the result contains `artifact_id` / `artifact_ref` instead of inline bytes
- media-only normalized results expose `runtime_provider` / `runtime_model` separately from `media_provider` / `media_model`
- local image/video generation runs in a one-shot subprocess (`execution_mode="local_one_shot_subprocess"`) unless every requested media model is resident (explicitly loaded with `load_model_residency`), in which case it runs in-process on the loaded pipeline (`execution_mode="resident_in_process"`, `resident_load_ids`)
- optional local media residency failures complete with `status_hint="warning"` and `degraded=true`; local media warmup that no capability plugin can serve (`image_generation`, `image_upscale`, `video_generation`, `text_to_video`, `image_to_video`, `tts`, `stt`, `music_generation`) reports `requires_long_lived_server=true`, and image/video tasks also report `execution_mode="local_one_shot_subprocess"`
- Gateway/hosts remain responsible for explicit Core server URLs, Core server auth headers, provider/model defaults, selected local-inference profiles, and translation of Gateway-owned env/config into explicit Runtime inputs; Runtime persists only JSON-safe routing metadata and artifact refs

Prompt cache / cached sessions:
- LLM clients expose cache control methods listed above for host-side preparation and inspection
- `LLM_CALL.params.prompt_cache_key` selects a cache key for a call; runtime can also derive a session-scoped key from `run.vars["_runtime"]["prompt_cache"]` or the Runtime-owned `ABSTRACTRUNTIME_PROMPT_CACHE` process default
- `LLM_CALL.params.prompt_cache_binding` is the durable exact-reuse input for bloc-backed prompt caching; if a binding includes `key`, Runtime adopts it as the effective prompt-cache key and refuses mismatches before provider execution
- Runtime only auto-derives session prompt-cache keys for text/chat calls; non-text output selectors such as image, voice, music, and transcription keep explicit `prompt_cache_binding` support but do not receive an inferred cache key
- when Runtime injects the derived session key, the client stamps session attribution (`session_id`, `run_id`, `workflow_id`, `node_id`, `namespace`) into the cache entry's metadata after each generate; caller-supplied `prompt_cache_key`s and binding keys are never stamped, so session-scoped clearing cannot destroy caches shared across sessions
- `get_abstractcore_host_facade(...)` exposes `get_memory_snapshot()`, `list_session_prompt_caches(session_id=None)`, and `clear_session_prompt_caches(session_id)`; session caches persist until a host clears them, the owning model is unloaded, or the provider evicts them — completing a run does not clear them
- `get_abstractcore_host_facade(...)` also exposes durable bloc helpers (`upsert_text_bloc`, `get_bloc_record`, `list_blocs`, `get_bloc_kv_manifest`, `ensure_bloc_kv_artifact`, `load_bloc_kv_artifact`, `list_bloc_kv_artifacts`, `delete_bloc_kv_artifact`, `prune_bloc_kv_artifacts`, `delete_bloc`)
- local Runtime owns the bloc root policy: `~/.abstractruntime/blocs` by default, `<base_dir>/blocs` for `create_local_file_runtime(...)`, and explicit `bloc_root_dir=...` overrides when needed
- provider cache/session handles are not durable runtime state and should not be stored in `RunState.vars`

Workspace scope:
- run vars `workspace_root`, `workspace_access_mode`, `workspace_allowed_paths`, `workspace_ignored_paths`, and the host-only `workspace_builtin_deny_prefixes` / `workspace_builtin_allow` scope file and shell tools; child runs inherit them (see `integrations/abstractcore.md#workspace-scoped-tools`)

Attachment registration limits:
- `TOOL_CALLS.payload.max_attachment_bytes`, `run.vars["_runtime"]["max_attachment_bytes"]`, or `ABSTRACTRUNTIME_MAX_ATTACHMENT_BYTES` bound the bytes Runtime stores when local `read_file` outputs are captured as session attachments

Docs: `integrations/abstractcore.md`.

### AbstractMemory bridge (KG effects)

Implementation: `src/abstractruntime/integrations/abstractmemory/effect_handlers.py`.

This provides handlers for `MEMORY_KG_*` effects (opt-in wiring layer).

## Utilities (host UX)

- Rendering helpers: `abstractruntime.rendering.stringify_json(...)` and `abstractruntime.rendering.render_agent_trace_markdown(...)` (`src/abstractruntime/rendering/*`)
- Active-context helpers (what is sent to the LLM): `ActiveContextPolicy`, `TimeRange` (`src/abstractruntime/memory/active_context.py`, exports in `src/abstractruntime/memory/__init__.py`)

## See also

- `../README.md` — install + quick start
- `getting-started.md` — first durable workflow
- `architecture.md` — component map + durability invariants
- `faq.md` — common questions and gotchas
- `integrations/abstractcore.md` — `LLM_CALL` / `TOOL_CALLS` wiring
