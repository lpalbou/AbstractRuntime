# Changelog

All notable changes to AbstractRuntime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-09-30

### Added

- **Email for each user** (see `docs/email.md`). A run acts for one account: its non-secret binding
  `_runtime.email_account = {account_ref, address}` is resolved at the moment a tool needs it through the host's
  per-runtime resolver (`Runtime.set_email_context_resolver(fn)`), so credentials never enter run vars, the ledger,
  tool arguments or results. `Runtime.set_email_binding(...)` binds automation occurrences; hosts set root runs with
  `bind_email_account(...)` after `strip_client_email_keys(...)`. Child runs inherit the binding and the
  pre-authorised recipients (the parent's value wins). Once a runtime has a resolver, AbstractCore's email tools in
  that process never fall back to the local AbstractCore settings.
- **`send_email_recipient@v2` approval refiner.** A `send_email` runs without an approval wait only when every
  recipient is the registered address (`_runtime.operator_email`) or is listed in `_runtime.email_allowed_recipients`;
  `reply_email`, unknown argument keys and anything unproven ask. The account's recipient policy and send limits
  (AbstractCore `guarded_send`) still apply to every send.
- **Durable event inbox and mail feeder** (`abstractruntime.email`): `JsonFileEventInbox` / `InMemoryEventInbox`
  (append-only, unique event ids, per-stream dedupe keys and state, crash-repaired index), attached with
  `Runtime.set_event_inbox(...)`; `EmailInboxFeeder.poll(ctx)` appends each new message whole, advances its
  UIDVALIDITY + UID cursor only after a durable append, resynchronises after a folder rebuild without loss or
  duplicates, passes a message that fails three polls, and reports connection failures as typed
  `{code, cause, fix, retryable}` with a capped backoff (60 s to 15 min). `email_trigger_consumers(...)` and
  `wake_email_automations(...)` serve the host's watcher.
- **`email.received@1` trigger source.** Typed filters (`from_in`, `from_domain_in`, `to_in`, `subject_contains`,
  `has_attachment`), batches no more often than `every` (default `1h` for automations that run a model, `60s`
  otherwise; at least `60s`), up to `max_batch` messages, each message at most once per automation across restarts
  and folder rebuilds. Occurrences receive the messages as `input_data.trigger` marked `content_trust: "untrusted"`
  and, for a string `prompt`, inside a fixed untrusted frame; the recorded envelope carries metadata only.
- **Send-email action** (`abstractframework.email-actions@1.0.0:send_email`, `register_email_action_workflow`,
  `email_action_target`): automations without a model send templated mail (fixed placeholders, `each` or `digest`
  mode) through an ordinary `send_email` tool call, under the same approval rule, policy and limits.
- **Automation definition schema v2**: `policy.email_allowed_recipients` (`"self"` and exact addresses, default
  `["self"]`, frozen into each occurrence) and `notify.channels` (`["console"]` or `["console", "email"]`), which
  attention items carry as `channels`. Stored v1 definitions read with the defaults.
- **Email tools follow the host's decision**: `email_enabled=` on `get_default_toolsets`, `get_default_tools`,
  `list_default_tool_specs`, `build_default_tool_map` and `list_tool_catalog`; the email kind adds `reply_email`,
  `search_emails`, `get_email_attachment` and the read-only `list_email_folders`.
- **Oversized mail never blocks the feeder**: a message over AbstractCore's reading limit is appended with its headers,
  attachment list and typed `body_skipped` record (bodies `null`, never cut), listed in `PollReport.body_skipped`,
  passed to occurrences in `input_data.trigger.emails[].body_skipped` and shown as `Body (not fetched)` in the frame.
- **The resolver learns who is sending** (`use="agent_tool"` or `use="action"`), so a host applies its users'
  "agent email tools" choice to agent tools only; the send-email action is identified by its node function.
- **`email_facade`** (`abstractruntime.integrations.abstractcore.email_facade`): AbstractCore's mail library for hosts.
- **Event-inbox retention**: `EventInboxRetention` (default 90 days and 10,000 events), `inbox.prune(...)` and
  `prune_email_inbox(runtime, retention=...)`, which never removes an event an active email automation has not read.
- **`policy.untrusted_input_tools`**: tools named one by one that an email-triggered occurrence may run without asking.
- `list_tool_catalog(email_off_reason=...)` names why email is off (`not_connected`, `admin_disabled`,
  `not_available` — the administrator has not made agent email tools available to this user — and `agent_tools_off`).

### Changed

- **Requires AbstractCore 2.20.0 or newer** (`abstractcore.comms.email`, the email tools on a per-run resolver).
- An automation triggered by `email.received@1` also withholds tools whose row declares a model-chosen destination
  (`fetch_url`, `browser_probe`) from its unattended grant; `reply_email` is withheld like `send_email`.
- `send_email` / `reply_email` `attachments` and `get_email_attachment`'s `output_dir` follow the run's workspace
  scope; `get_email_attachment` is a `write` tool.
- A run policy that only withholds tools (`tool_policy.withheld_tools`) is applied like any other run policy, so the
  per-call refiners run for it.
- The per-call refiners (`send_email_recipient@v2`) also run when a run carries no `_runtime.tool_policy`, on top of
  the executor's static policy (its `require_approval_tools` still wins), and decide per call: one `send_email` that
  needs a person makes the batch wait.
- No environment variable enables the email tools: `ABSTRACT_ENABLE_EMAIL_TOOLS` is gone and
  `ABSTRACT_ENABLE_COMMS_TOOLS` enables WhatsApp and Telegram only; pass `email_enabled=True`.
- Under "allow all tools", an email-triggered occurrence still withholds `fetch_url` and `browser_probe` unless the
  user named them in `policy.untrusted_input_tools`; the untrusted frame tells the agent not to follow links or
  instructions contained in the emails and to act only on the automation's mission.
- After a folder rebuild with nothing to resynchronise, the feeder stores the new baseline AbstractCore returns
  (`reset` and `baseline`) instead of fetching one itself.
- Large structured tool outputs (for example a `read_email` with a big HTML body) are offloaded to session
  attachments under the same inline limit as text outputs; the result carries artifact references.
- The process-local email helpers (`list_email_accounts`, `list_emails`, `read_email`, `send_email`) are no longer
  exported from `abstractruntime.integrations.abstractcore`; they remain on the host facade for single-user installs.

## [0.7.3] - 2026-09-29

### Security

- **An automation's tool grant no longer pre-approves sending messages** (framework backlog 0992
  WP0). With the default `policy.tool_approval: "auto"`, the grant listed every tool the runtime
  classifies, `send_email` included, and an explicit name in the grant overrides the
  `model_controlled_destination` belt: an occurrence that read an inbound email or a web page could
  mail data to any address that text named. The grant now withholds every tool whose inventory row
  carries `comms_send` (`send_email`, `send_whatsapp_message`, `send_telegram_message`,
  `send_telegram_artifact`), even when the target's `allowed_tools` names it, and lists them in
  `tool_policy.withheld_tools`. Their calls take the normal approval point: a `send_email` whose
  every recipient is the registered user's address (`_runtime.operator_email`) still runs
  unattended; any other recipient parks the occurrence on a `tool_approval` wait until a person
  approves or refuses it (the behaviour of `"ask"`). Recipients named in an automation's definition
  are not auto-approved yet; that allowlist is part of backlog 0992.

### Fixed

- **A resident image/video model serves generation.** After `load_model_residency(task="image_generation")`
  (the Gateway's `POST /models/load`, the console Load button, a flow's model residency node), a local
  image or video request whose every media output names a resident model runs in-process on the loaded
  pipeline (`execution_mode="resident_in_process"`, `resident_load_ids`) instead of loading the model
  again in a one-shot subprocess. Measured through AbstractGateway with FLUX.2 [klein] 4B on a 16 GB
  NVIDIA card: 17-19 s per image on the resident pipeline against 54-59 s with a reload per request, and
  no second copy of the weights on the GPU. Pooled clients of `create_local_runtime(...)` use their pool's residency core. Models that
  were not explicitly loaded keep the isolated subprocess (crash isolation for native Metal/CUDA
  failures), so the explicit load is the opt-in.

### Changed

- **Requires AbstractCore 2.19.2 or newer** (`abstractcore`, `abstractcore[apple]`, `abstractcore[gpu]`).
  It brings AbstractVision 0.3.33, whose loads report `loaded_new` (a fresh image-model load reads as
  `loaded`, not `already_loaded`) and whose unload frees the model's memory, which the resident path
  above relies on.

## [0.7.2] - 2026-09-29

### Changed

- **Dependencies follow AbstractCore's three install settings.** The base install requires
  `abstractcore>=2.19.0` (its light install: remote providers, built-in tools and the MCP worker's
  web tools, media inputs, and the voice/vision/music plugins), `abstractruntime[apple]` requires
  `abstractcore[apple]>=2.19.0` and `abstractruntime[gpu]` requires `abstractcore[gpu]>=2.19.0`.
  What gets installed is unchanged; the deprecated AbstractCore extras (`remote`, `tools`, `voice`,
  `all-apple`, `all-gpu`, ...) are no longer named.
- **`openai` may be 2.x (`openai<3.0.0,>=1.109.1`).** The `<2.0.0` cap held vLLM at an old release
  in the `gpu` setting (newer vLLM needs `openai>=2`), and that release does not start with
  Transformers 5.
- **The MCP worker's startup tip names the right install.** When `abstractruntime-mcp-worker` fails
  to start, its tip says that the web tools come with AbstractCore's light install and suggests
  `pip install -U abstractcore` to repair it, instead of pointing to the deprecated
  `abstractcore[tools]` extra.

### Fixed

- **The local STT model catalog lists each engine's own models.** When AbstractVoice's catalog
  carried no per-provider STT map (for example with no OpenAI key),
  `GET /api/gateway/audio/transcriptions/models` listed OpenAI's `gpt-4o-transcribe` as a
  faster-whisper model and reported it as `active_model`. `local_list_stt_models()` now asks each
  STT provider for its own list (without loading an engine) and takes `active_model` from the active
  provider's list.

## [0.7.1] - 2026-09-28

### Fixed

- **Images in replayed history stay images.** When the history window drops older turns, it writes a
  `[#TRUNCATION: ...]` notice at the start of the oldest kept message. If that message held a list of content parts
  (text plus an image, `[{"type": "text", ...}, {"type": "image_url", ...}]`), the notice turned the whole list into
  one string, so the model received the image's base64 as text. The notice is now added as one text part and every
  other part is left exactly as it was. A message that starts with its `<runtime_metadata>` envelope keeps it first.
  This affects `announce_dropped` and `window_transcript`, and so the gateway's `/runs/start` client-context window and
  any replay that keeps images. Content of any other type is left untouched and a warning is logged.
- **An image no longer counts as tens of thousands of tokens in the window.** `estimate_message_tokens` turned a
  content-part list into a string, so a 150 KB inline image counted about 50,000 tokens and pushed every older turn out
  of the history window. Text parts now count their text, and each media part (image, audio, file) counts a flat 512
  tokens (`MEDIA_PART_TOKEN_ESTIMATE`, AbstractCore's per-image fallback). The input-budget trim
  (`trim_messages_to_max_input_tokens`) uses the same estimate.

## [0.7.0] - 2026-09-28

Replayed session history is one window: the most recent 50,000 tokens of whole turns, recorded in the run.
Library callers of `session_chat_messages` and `automation_timeline_messages` should read the **Breaking** note.

### Changed

- The base install and the `apple` / `gpu` profiles require AbstractCore 2.18.0 or newer. Its voice extra brings
  AbstractVoice 0.13.0, the first release that honours the `voice_openai_api_key` setting described below.

- **Session history replay is one window: the most recent 50,000 tokens.** `session_chat_messages` keeps the newest
  turns that fit `HISTORY_REPLAY_MAX_TOKENS` (50,000) estimated tokens, as whole messages, newest first. The window has
  no gaps, and a total of exactly 50,000 tokens fits. The 40-message cap, the 24,000-character total budget and the
  8,000-character cut per message are removed, so no replayed message is ever cut. When the newest turn alone is larger
  than the window, it is kept whole and the report says so (`oversize_turn_kept`). Tokens are counted with the existing
  estimator, `memory.token_budget.estimate_message_tokens`. This window applies to automation growing mode (it
  replaces the 40-message / 24,000-character limit of contract D), to discussion seeds (`automation_timeline_messages`)
  and to host session seeding. The window is the only limit on replayed history; the model can use the rest of its
  context window (operator ruling 2026-09-28, ADR-0026 §3: budgets are met by choosing whole turns, never by cutting
  content).
- **Library API:** `session_chat_messages` still accepts `max_messages`, `max_chars_per_message` and
  `max_total_chars` (keyword-only, default `None`) but IGNORES them: they no longer limit anything. Hosts built against
  0.6 (AbstractGateway 0.6.0 passes `max_messages=40, max_total_chars=24000` and treats any error as "no history") keep
  replaying their sessions under the window. Passing any of them logs one warning and names them in
  `report["ignored_inputs"]`. Pass `max_tokens` (a positive int) to use a different window.
- **Breaking (library API):** `automation_timeline_messages` takes `max_tokens` instead of `max_messages` /
  `max_chars_per_message` / `max_total_chars`. `GROWING_MAX_MESSAGES` and `GROWING_MAX_TOTAL_CHARS` are removed from
  `abstractruntime.automations.models`.
- **The entity chat driver and the entity visit workflow use the same window.** They used to keep only the last 10
  turns in the prompt, without saying so. Now every turn is kept, and each prompt carries the most recent 50,000 tokens
  of whole turns, with the `[#TRUNCATION: ...]` line when older turns were dropped. The visit records the window's
  report in `vars._runtime.session_history`; `ChatSession` puts it on `TurnReport.history_window` (and
  `ChatSession.history_window`).
- **The react visit arm (the one AbstractGateway uses) sends the window too.** Its transcript is the durable
  `context.messages`, sent by the AbstractAgent react loop on every call; before, nothing bounded it (after 12 turns of
  ~10k tokens a request was ~117k tokens, and a poisoned tool result rode every later turn). BRIDGE now sets
  `_runtime.history_window_tokens` (50,000); the adapter sends `window_transcript(context.messages)` on every call —
  a tool result stays with its turn, so a 495k-character tool result drops out of the request once a newer turn
  exists — and records the report in `vars._runtime.session_history`. The stored transcript stays whole. BRIDGE also
  records `_runtime.history_window_turn_start` (the visitor's message), and the window keeps the turn in progress whole
  from there (`window_transcript(..., current_turn_start=)`): an `ask_user` answer or operator guidance inside the turn
  no longer splits off the turn's own question and tool results; a turn larger than the window is kept whole and
  reported as `oversize_turn_kept`. HARVEST
  keeps a turn whose adapter recorded no window (AbstractAgent older than 0.3.17 sends the whole transcript) but
  logs a warning ("abstractagent < 0.3.17 does not apply the history window; upgrade abstractagent") and records
  `{"window_applied": false, "reason": "agent_too_old", ...}` in `vars._runtime.session_history` (counting the messages
  the last request carried, not the reply stored after it); a windowed turn
  records `window_applied: true`. **The window on the react arm needs AbstractAgent 0.3.17 or newer.**
- The `[#TRUNCATION: ...]` line is written after a stamped message's own `<runtime_metadata>` envelope, which stays at
  the head (written in front, it made the payload boundary stack a second envelope).
- **Breaking (library API):** the `history_turns` parameter of `build_visit_workflow` and `ChatSession` is removed,
  with `DEFAULT_HISTORY_TURNS` (no caller in the framework passed it). The entity visit no longer sets the
  per-message character limits `_limits.max_tool_message_chars` (32,000) and `_limits.max_message_chars` (80,000) on
  re-sent history, and `VISIT_HISTORY_TOOL_RESULT_CAP_CHARS` / `VISIT_HISTORY_MESSAGE_CAP_CHARS` are removed. A
  single oversized tool result is still limited by AbstractAgent's own guard for oversized messages (200,000 characters).
- `trim_messages_to_max_input_tokens` (used only when a caller sets a positive `max_input_tokens`) keeps each tool
  result together with the assistant tool call before it. It used to drop the call and keep the result, which
  OpenAI-style APIs reject.

- **The context mode decides whether an automation's target reads history.** At admission the runtime sets
  `use_context` (and `include_context` when present) to `true` for growing occurrences and `false` for independent
  ones, and to `true` for discussion forks. A `use_context: false` frozen in an older definition no longer makes a
  growing automation or a discussion replay nothing (basic-agent's `use_context` pin defaults to `false`). The run
  records `_runtime.automation_context = {mode, use_context, target_use_context}`. Occurrences that were already
  admitted before the upgrade keep their frozen inputs.

### Added

- Voice discovery (`local_get_voice_catalog`, `local_list_tts_models`, `local_list_stt_models`) accepts a
  host-supplied `voice_openai_api_key` and passes it to AbstractVoice as the plugin setting of the same name (the
  OpenAI credential for its `openai` engines; AbstractVoice reads no environment variable for it). The discovery
  facade and the local clients accept and forward it (`AbstractCoreDiscoveryFacade.get_voice_catalog` /
  `list_tts_models` / `list_stt_models(voice_openai_api_key=...)`); the remote client accepts it and never sends it.
  In `llm_kwargs` it reaches the voice plugin through the provider's `config` and changes no text request; tests
  prove this on real OpenAI, LM Studio, Ollama and MLX provider classes, with the network refused.
- **The window is recorded.** `session_chat_messages` and `automation_timeline_messages` return a `ReplayedHistory`
  (a list of messages) with a `.report`: `policy`, `max_tokens`, `token_estimator`, `replayed_messages`,
  `replayed_tokens`, `dropped_messages`, `dropped_tokens`, `dropped_counts_complete`, `oversize_turn_kept`. A growing
  occurrence records it in `vars._runtime.session_history`, so the `automation.admitted` record's frozen inputs carry
  it too. A discussion root records its seed window the same way. When older turns are dropped, the oldest replayed
  message starts with a `[#TRUNCATION: ...]` line that gives the number of dropped messages and tokens and the window.
  New exports at the package root: `HISTORY_REPLAY_MAX_TOKENS`, `ReplayedHistory`, `fold_history_window`,
  `announce_dropped` (writes that `[#TRUNCATION: ...]` line after a host's own `fold_history_window` call, for example
  on client-sent history; `session_history._announce_dropped` stays as an alias) and `window_transcript` (the window
  over a transcript the caller already holds).
- `local_list_tts_models` reports why a provider cannot be listed: when the listing is unavailable, `error` carries
  AbstractVoice's `unavailable_reason` (for example "no OpenAI API key is configured ..." or an unknown provider id).
  It used to return `available: false, error: null`.
- Replay reads turns newest-first in batches that double in size, and stops once the window is full. A long session
  therefore costs about as much as its window holds.

### Tests and CI

- CI installs the extras that exist (`.[test]`) and runs the history-window, entity window, automation timeline,
  input-trim and voice-key tests. The five tests that need the sibling `abstractflow/` checkout or the
  `abstractagent` package skip when those are absent (a standalone checkout or an sdist).

## [0.6.0] - 2026-09-27

Automations v1: run a workflow on a schedule or on request as a durable, crash-safe controller run. Hosts that
fold root runs into sessions should read the **Changed** notes (occurrences appear as session turns).

### Added

- **Automations** (`abstractruntime.automations`, see `docs/automations.md`): run a workflow on a trigger. An
  automation is a durable controller run (the packaged bundle `abstractframework.automation-controller@1.0.0`), and
  each firing (an occurrence) is a child run with a deterministic id, so a crash never loses an occurrence or starts
  one twice. Every transition is an `automation.*` record in the controller's ledger.
  - `create_automation(runtime, request, *, actor_id=None)` returns `(automation_id, revision)`; the same request
    returns the same automation and a reused `request_id` with a different request raises `identity_conflict`.
    `get_automation`, `list_occurrences` and `drive_automation` (a run loop for hosts without one) complete the
    service API.
  - `apply_automation_command` pauses, resumes, runs now, revises, stops the current occurrence or archives. Commands
    are idempotent per `command_id` (a reused id for a different command is refused with `identity_conflict`), are
    checked against `expected_revision`, and never interleave with the controller. Pause stops scheduled runs only;
    resume never fires missed runs; revising `policy` changes only the fields sent.
  - Context: independent occurrences (the default) start fresh in their own session; in growing mode each occurrence
    joins the automation's session and receives its previous turns (at most 40 messages / 24,000 characters).
  - Failed occurrences are retried: 3 attempts by default, 30 s then 60 s apart, capped at 10 minutes. An occurrence's
    inputs are frozen when it is admitted, so a later revision never changes it.
  - Automations are quiet unless the output carries `notify: true` or `notify: {title, body}`, or an occurrence still
    fails after its last retry. `list_attention` pages these items oldest first.
  - `policy.tool_approval`: `"auto"` (default; creating the automation is the consent, and each occurrence receives a
    frozen `_runtime.tool_policy.auto_approve_tools` grant) or `"ask"`. Unclassified tools still ask.
  - `pending_waits` lists occurrence runs waiting on a person as typed waits: `ask_user`, `tool_approval` (with the
    calls to approve in `details`) or `event` (with `{scope, name}` when known); `ANSWER_PAYLOADS` gives the resume
    payload for each kind.
  - `start_discussion(..., workspace_root=...)` forks a separate conversation at occurrence N: a new root run in its
    own session, seeded once with the automation's whole conversation through N (every occurrence 1..N, in either
    context mode, oldest dropped first under the history budget), working in its own writable workspace with the
    automation's workspace mounted read-only alongside, and without the automation's tool grant. Discussion sessions
    are scoped to their automation.
  - `get_automation` and `list_automations` summaries carry `next_fire_at` for every active scheduled automation,
    also while an occurrence runs (the tick the controller will admit next, coalescing included), and
    `current_occurrence` (`{index, run_id, attempt, status}` of the occurrence in flight), so clients never
    compute schedules themselves.
  - `adopt_legacy_schedule_projection(run)` summarizes a legacy gateway `scheduled:*` root read-only.
  - Limits: `every` is at most `366d`, `count` at most 1,000,000.
- **Trigger sources** (`abstractruntime.triggers`): `schedule@1` (fixed UTC intervals such as `5m` or `24h` on a grid
  that does not drift; `start_at`, exclusive `until` and `count`; one-shot when `every` is absent; missed ticks run
  once, marked `coalesced`) and `manual@1`. Other packages add sources through the `abstractruntime.trigger_sources`
  entry-point group; a broken third-party source is listed as unavailable, and a missing built-in source is an error.
- VisualFlow node type `automation` (the controller's nodes, adapters `automation.<node_id>`).
- `abstractruntime.automation_queries`: `list_automations(store, status=, cursor=, limit=)` (automation summaries,
  newest first, with a cursor that survives restarts; archived automations stay listed) and
  `latest_occurrence(store, automation_id)`. `changed_since` is refused with `unsupported_feature`.
- Explicit run ids: `Runtime.start(..., run_id=...)` and `START_SUBWORKFLOW` `payload.run_id` create a run only if the
  id is free. Starting it again with the same identity (workflow, session, parent, `vars._meta.occurrence`,
  `vars._meta.creation_digest`) returns the existing run untouched; a different identity raises
  `RunIdentityConflict` (a `ValueError`, `reason_code = "identity_conflict"`). A parent that crashes after starting
  such a child, but before saving its wait, finds the same child on replay; if the child already finished, the parent
  receives its result directly.
- `RunStore.create_if_absent(run) -> (run, created)` on the SQLite, JSON-file, in-memory and offloading stores (the JSON
  store publishes a fully written temp file with an atomic hard link and never replaces an existing run file).
  `store_supports_create_if_absent` / `require_create_if_absent` check a store. Recovery after a process crash is
  covered; power-loss durability is not claimed.
- `run_mutation_lock(run_id)`: the per-run, per-process lock that `Runtime.tick` holds for the whole tick and
  `Runtime.resume` for its commit. Hosts take it around their own read-modify-save of a run. v1 supports one writer
  process per store.
- Run index attribution: rows carry `automation_id`, `role` (`controller`, `occurrence`, `descendant`, `discussion`,
  `legacy_schedule` or empty), `occurrence_index` and `session_kind` (`chat`, `automation`, `occurrence`,
  `discussion`), and `list_run_index` filters on them (`session_kind="chat,discussion"` works). Existing SQLite stores
  fill the new columns once when opened; the JSON store re-reads each run file once. Every store has
  `session_kinds(session_id)` and `latest_occurrence_row(automation_id)`.
- Run index rows carry `workspace_root`: the folder the run executes in (its top-level `vars["workspace_root"]` as
  stored, stripped, never resolved; `None` when absent), so apps can open a session's folder without loading runs.
  Existing SQLite stores fill it once when opened; the JSON store re-reads each run file once.
- `session_attribution(store, session_id)` (`abstractruntime.core.run_attribution`) reports what kind of session an id
  is (`chat`, `automation`, `occurrence` or `discussion`, with the discussion's validated root, automation and
  workspace). Every root run started in a discussion session gets the discussion's own workspace, access settings and
  read-only mounts (the whole-workspace read-only flag only when the discussion carries it) and its provenance,
  whatever the caller passed; when the session cannot be attributed (an invalid
  discussion root, or a store without a run index) `Runtime.start` raises `SessionAttributionError`.
- `select_session_turns(store, session_id, ...)`: the one definition of a session's turns (parent-less runs except
  automation controllers, plus automation occurrences; never child runs, internal runs, legacy scheduled wrappers or,
  unless asked, draft-test runs), with `include_occurrences`, `automation_id`, `through_occurrence`, `until_ms`,
  `include_drafts` and `limit`. `is_draft_lifecycle` is exported too.
- `session_chat_messages(..., automation_id=None, through_occurrence=None, strict=False)`: bound the replayed turns to
  one automation or to the history as it stood at occurrence N, however old. In a discussion session the seed is
  replayed first and dropped first under the budget. `strict=True` raises `SessionHistoryError`
  (`reason_code = "history_unavailable"`) instead of returning a partial history.
- Read-only workspaces: a run started with `workspace_read_only: true` (or trusted runtime policy
  `_runtime.workspace_read_only: true`) can read its workspace but not change it. Tools that write files or run
  commands or code, and any tool the runtime has not classified, are refused, as are the VisualFlow `write_file`,
  `write_pdf`, `write_docx`, `write_chart` and `export_artifact` nodes. A read-only workspace folder is never created,
  a read-only run without one is refused, and child runs and nodes inherit the setting.
  `tool_effects.TOOL_EFFECT_CLASSES` classifies every exposable tool (`read`, `write`, `exec`, `delegate`, `comms`,
  `memory-write`).
- Read-only mounts: `_runtime.workspace_read_only_paths` (absolute folders, symlinks resolved) makes those folders
  readable but not writable by the file tools (`write_file`, `edit_file`) and the VisualFlow file/export writers,
  while the run's own workspace stays writable. Commands and code (`execute_command`, `execute_python`, ...) are not
  restricted by a mount: the shell cannot be sandboxed, so mounts protect the file tools only. Child runs and
  VisualFlow nodes inherit the mounts and can only add more. Helpers `read_only_paths(vars)` and
  `path_is_read_only(vars, path)` in `abstractruntime.utils.workspace_paths`.
- `select_session_turns(..., automation_id=A)` gathers A's occurrences from every session (an independent-mode
  automation runs each occurrence in its own session), interleaved with the session's own turns.
- `JsonFileRunStore.warm_session_index()` builds the session and children indexes at host startup.

### Changed

- Session history bundles and session replay include automation occurrences as turns (kind `occurrence`, with
  `automation_id` and `occurrence_index`); a retried occurrence counts once, as its last attempt. Draft-test runs are
  left out of a session's history.
- `list_run_index(root_only=True)` returns turn roots: parent-less runs except automation controllers, plus automation
  occurrences. Apps that fold root runs into sessions show an automation's session as a chat whose turns are its
  occurrences.
- Children of an automation occurrence carry `vars._meta.occurrence` with `role = "descendant"`, and children of a
  discussion carry its `vars._meta.discussion` (without the seed); a child cannot clear or change either.

### Fixed

- Session lookups on the JSON run store use an index of each session's runs instead of scanning the store (about
  0.1 ms per new chat turn at 20,000 runs). Several store objects or processes on one run folder see each other's
  created and deleted runs through a creation journal (`.runs_created.log`).
- History "through occurrence N" returns the history up to that occurrence in sessions of any length, and raises
  (`strict`) or returns nothing when the session has no occurrence N.
- A discussion session's seed is read only from a validated root: every run of the session must name the same root,
  and that root must belong to the session and carry the seed.
- The JSON run store removes run temp files left behind by a crash (older than 10 minutes) when it opens.
- Automation identity metadata (`vars._meta.automation`, `.occurrence`, `.discussion`, `.creation_digest`) always
  stays inline when a finished run is offloaded.

## [0.5.1] - 2026-09-26

### Added

- `StaleResumeError` (a `ValueError`, exported from `abstractruntime`): what `Runtime.resume` raises
  when the run is no longer waiting or waits on another key, so a host that resumes the same wait
  from two places can tell a lost race from a real failure. The messages are unchanged.

### Fixed

- A wait is resumed at most once. Two callers resuming the same wait at the same moment could both
  succeed, and the run then executed the resumed node twice (seen as an Agent node starting a
  second full agent loop after its child finished). The second caller is now refused with
  "Run is not waiting", as it already was when the calls did not overlap.

## [0.5.0] - 2026-09-26

### Fixed

- When an LLM call is re-run after a stray kill, its interrupted first run ends with
  `delta_end` `cancelled` with `detail: "reinvoked"` (the run continues) and the re-run streams
  under its own call id, so a live view never shows a partial answer followed by the full one.
- The last batch of a streamed reply is sent before the call's final record is written, never just
  after it; text arriving once the call has returned is dropped.
- A streamed reply that only turns out to lack token usage at the end now ends with
  `reason: "completed"` in the live view (the text did stream); `usage_unavailable` is recorded on
  the `LLM_CALL` record only.
- Streamed answers without usage are no longer invisible to the aborted-generation check: it judges
  them from `finish_reason` and the text, and the record says so with `metadata.usage_estimated`.
- Servers that reject `stream_options` but still send token usage (LM Studio) stream again.
  Streaming is refused for missing usage only when the provider reports the rejection AND a
  streamed answer came back without usage; one call in ten streams again to re-check, and a
  streamed answer with usage lifts the refusal.
- A child run or a VisualFlow node can no longer switch off the host's built-in workspace
  protection: it may add `workspace_builtin_deny_prefixes`, never remove the host's, and cannot
  widen `workspace_builtin_allow` beyond the parent's (its own folder counts only when it is
  inside an allowed folder).
- Harmony-format models (gpt-oss) stream their answers live: the `final` channel is sent as
  content, `analysis` as reasoning, and tool calls are held back. A call whose whole answer is held
  back ends with `reason: "unavailable"`, `detail: "tool_envelope_holdback"` instead of a silent
  empty live view.
- One streamed answer without token usage no longer turns streaming off for that model until
  restart. That call is reported (`usage_unavailable`) and the next call streams again; streaming
  stays off only when the provider reports that its server rejects usage in streams.
- Host-protected folders no longer grow the system prompt. A host can pass its own protection as
  `workspace_builtin_deny_prefixes` (for example the gateway's data folder and credential folders)
  and `workspace_builtin_allow` (the run's own folder inside it). File tools, listings and the
  shell's starting folder are refused under a denied prefix, except under an allow entry. These
  entries are enforced but never written into the model's system prompt, which keeps it identical
  from turn to turn (and the prompt cache warm); only the operator's own `workspace_ignored_paths`
  are shown to the model. Child runs inherit both lists.
- MLX calls stream again when they use a prompt-cache key: AbstractCore now puts the prompt-cache
  record on the stream's last chunk, so the streamed record matches the non-streamed one.
- Streamed LLM calls now record the same result as non-streamed ones: `raw_response` is present
  (the provider's terminal chunk), and the last reasoning fragment no longer leaks into
  `metadata.reasoning_delta`.
- Changing the default text model no longer leaves the previous model in memory. The previous
  in-process model (MLX or HuggingFace) is unloaded from the whole process before the new default
  is loaded, unless the pool still uses it or it is locked. A call still generating on it is not
  cancelled; the model is unloaded when that call ends. Measured on a 27B MLX model: 17 GB stayed
  resident after a console switch; now it is freed.
- Changing the default model in one service never unloads a model another service, user or
  entity in the same process still uses or has locked. The check covers every owner in the
  process and runs under one lock with the unload, so a model being loaded again at that moment
  is not unloaded either.
- `list_model_residency` diagnostics carry `pending_ejects` (a previous model waiting for its
  running call to end before it is unloaded) and `last_switch_ejects` (unloaded, kept because
  something still uses it, or failed, with the reason).
- A load with `ttl_s` or `keep_alive` on an in-process model, or on a model that was already
  loaded, now reports them under `unsupported_options` with a warning; provider warnings appear in
  the response's top-level `warnings`.
- A model load that fails part-way unloads what it had loaded.
- Chat compaction summarizes a run with the model the run uses, instead of loading the default
  model just for the summary. A run with no model of its own is summarized with the current
  default model, resolved on each call; the summarizer no longer holds the model that was the
  default at startup.
- Unloading a model by `runtime_id` alone now works for every task: TTS, STT, image, music and
  embedding rows, not only text. A backend's own id is looked up in the listing.
- Listings no longer report a task error on every call for `image_upscale`, `video_generation`,
  `text_to_video`, `image_to_video` and `text_to_audio`; they are served by the image and music
  capabilities.
- Changing a capability default unloads the models the previous capability routes had loaded
  instead of leaving them in memory.
- Unloading a local MLX model through the runtime (`model_residency` unload, on
  a single client or the multi-model pool) now frees it from every holder in
  the process, not only from the runtime's own instance. Before, the weights
  could stay in memory when another part of the process still held the model
  (for example a client created with a per-request override, a chat summarizer
  built at startup, or a runtime left over after a bundle reload), and the
  unload still reported success. The unload result now includes a
  `process_eject` report and names what, if anything, is still held.
- Model residency listings now show MLX models the process still holds, even
  when the runtime's own pool no longer references them
  (`provider_state: "resident_via_other_holders"`). Such a model can be
  unloaded from the listing like any other.

### Added

- **Live token streaming.** `Runtime.set_live_delta_sink(sink)` lets a host receive the text of an
  answer while it is generated. For runs started with `_runtime.stream: true`, each LLM call streams
  and the sink gets `llm.delta` events (`run_id`, `parent_run_id`, `node_id`, `call_id` = the LLM
  call's step id, `seq`, `text`, `channel` = `content` or `reasoning`) and one `llm.delta_end`
  (`completed`, `failed`, `cancelled`, or `unavailable` with a `detail`) after the call's final
  record is written. Deltas are never written to the ledger. Thinking is sent on its own `reasoning`
  channel, including inline `<think>` markup, and tool-call markup never reaches the live text.
  A call streams only when its recorded result keeps its usage, `raw_response` and prompt-cache
  telemetry; otherwise it runs non-streamed and the reason is sent to the client and recorded as
  `_runtime_observability.stream_unavailable`. Child runs inherit `stream`. `Runtime.start`
  refuses a non-boolean `_runtime.stream`. Remote mode (AbstractCore server) stays non-streaming.
  See `docs/integrations/abstractcore.md#live-token-streaming`.
- Embedding models loaded in the process appear in `list_model_residency` as `task: "embedding"`
  rows (`local:embedding:huggingface:<model>`, with holders and bytes) and can be unloaded like any
  other model.
- HuggingFace models held in the process outside the runtime's pool are listed and unloaded like
  MLX ones.

### Changed

- Compatibility: AbstractRuntime 0.5.0 requires AbstractCore 2.16.0 or newer (was 2.15.1 in
  0.4.35) in the base install and in the `apple` and `gpu` extras, and `models_engines_support()`
  reports 2.16.0 as `required`. The live streaming, process-wide unload and residency features above
  use modules that first ship in AbstractCore 2.16.0. Upgrading AbstractRuntime upgrades AbstractCore
  accordingly; if you pin AbstractCore yourself, raise the pin to 2.16.0.

## [0.4.35] - 2026-09-25

### Fixed

- `models_engines_support()` and the `AbstractCoreTooOld` error now name
  AbstractCore 2.15.1 as the required version (they said 2.14.0), matching the
  base-install floor. Who is affected: hosts such as AbstractGateway that read
  `required` to decide whether the models and engines features are usable. They
  could accept AbstractCore 2.14.x, where `host_job_cancel(by=..., user=...)`
  fails. No action is needed if you install AbstractRuntime normally: the
  AbstractCore floor is unchanged at 2.15.1.

### Documentation

- `ROADMAP.md` no longer shows a stale "v0.4.2" status or a priority list; it
  points to `CHANGELOG.md` and the AbstractFramework backlog.

## [0.4.34] - 2026-09-24

### Changed

- `config_facade.host_job_cancel(job_id, *, by="api", user=None)` records who
  cancelled a host job (a model download, delete or engine install): `by`
  (`api`, or `console` for a click in an embedded console) and the signed-in
  `user` are stored on the job as `cancelled_by`, `cancelled_by_user` and
  `ended_reason`. Existing calls without these arguments keep working and are
  recorded as `by="api"`. Hosts such as AbstractGateway can pass the requesting
  account to show who cancelled a job.
- Compatibility: AbstractRuntime 0.4.34 requires AbstractCore 2.15.1 or newer
  (was 2.14.0) in the base install and in the `apple` and `gpu` extras.
  Upgrading AbstractRuntime upgrades AbstractCore accordingly; if you pin
  AbstractCore yourself, raise the pin to 2.15.1 as well.

## [0.4.33] - 2026-09-23

### Added
- **Models and engines for hosts.** `config_facade` passes AbstractCore's
  models and engines features through to hosts, so the Gateway can offer the
  same model browser and engine installer as `abstractcore models` and
  `abstractcore engines` without importing AbstractCore:
  `host_profile`, `engine_inventory`, `engine_status`,
  `engine_install_plan`, `engine_download_url`, `engine_install`,
  `model_catalog`, `list_installed_models`, `model_delete_blockers`,
  `delete_model_artifact`, `start_model_download_job`, `host_jobs_list`,
  `host_job`, `host_job_cancel` and `console_fragment`. Payloads are
  AbstractCore's own (`host_profile_v1`, `engines_status_v1`,
  `model_catalog_v1`, `models_installed_v1`, `host_job_v1`).
- `models_engines_support()` reports whether the installed AbstractCore has
  these modules. It never raises.
- `AbstractCoreTooOld` (a `NotImplementedError`) is raised when the installed
  AbstractCore predates these modules. The message names the installed
  version and the upgrade command.
- `HostActionRefused` carries an HTTP status and a structured body for
  refusals: engine installs not allowed, unknown engine, an install already
  running, a delete blocked by a loaded or shared model. Hosts no longer need
  AbstractCore's exception types to tell refusals apart.

### Changed
- The AbstractCore floor is now 2.14.0 in the base install and in the `apple`
  and `gpu` extras.

## [0.4.32] - 2026-09-23

This release also contains everything listed under 0.4.31, which was never
published on its own.

### Added
- **Stop reaches the running effect.** `Runtime.cancel_run(...)` now signals the
  effect that is executing for the run (and for its in-flight descendants), not
  only the stored status. The `LLM_CALL` handler passes the signal to
  AbstractCore as `cancel_event=`, so a local provider stops within one token,
  and the remote client closes its request to the AbstractCore server, which
  treats the disconnect as a cancel. The stopped attempt is recorded as
  `cancelled` (new `StepStatus.CANCELLED`, `EffectOutcome.cancelled`) with
  `cancelled_by`, `reason` and timing fields, is never retried, and nothing
  after it runs. Remaining calls of a tool batch are reported as not started.
  `cancel_run(..., cancelled_by=...)` is a new keyword (default `"api"`).
- `core/effect_cancellation.py`: `inflight_effects()` lists executing effects;
  `request_model_effects_cancel(provider, model)` stops the effects using a
  model; `kill_inflight_effect(step_id, killed_by=...)` is an in-process hard
  stop for a call that ignores its cancel event.
- **Model eject stops its calls first.** Local `unload_model_residency`
  cancels the effects using the unloaded model before unloading it, and the
  ledger records `cancelled_by: "model_eject"` with the model named.
- **Speculation (MTP) controls.** `LLM_CALL.params.speculation` (`False`,
  `True`, or a Core speculation object) and boolean or string `thinking` are
  forwarded by local and remote clients. `_runtime.speculation` sets a
  run-wide preference inherited by subworkflows, Agent loops, delegated
  children and structured-output follow-up calls; an explicit `False` stays
  Off across every boundary. VisualFlow LLM Call and Agent nodes accept a
  `speculation` input. Scoped AbstractCore defaults reach provider
  construction, and `config_facade.normalize_speculation_control()` validates
  host values with Core's vocabulary.
- **Native MLX execution controls.** Local clients admit concurrent calls only
  when the loaded MLX instance advertises safe scheduling; other instances
  stay serialized through streamed completion. Remote results expose Core's
  `execution`, `speculation`, `performance` and `prompt_cache` metadata.
  New `get_execution_capabilities(model_name=None, provider=None)` on the
  clients and the discovery facade asks the actual execution host without
  loading a model.
- **Text phase progress.** Every `LLM_CALL` is offered the durable progress
  channel. Providers that report prefill/generation phases produce
  `abstract.progress` ledger events with `kind: "llm"` (`phase`,
  `prompt_tokens`, `cached_tokens`, `fed_tokens`, `generated_tokens`,
  `ttft_s`, `tokens_per_second`). The callback travels beside the effect
  (`core/progress_channel.py`), so `effect.payload` stays JSON-serializable.
- **Run-tree tool ceiling.** An explicit `allowed_tools` list in `_runtime`, in
  a child run, or in a tool payload is intersected across the run tree
  (`core/tool_scope.py`). Approval policy can remove a prompt but never grant
  a tool outside the ceiling; malformed lists and broken ancestry fail closed.
- `Runtime.tick(..., step_gate=callable)` lets a host pause a run at the next
  step boundary; the run stays `RUNNING` and a later tick continues.
- `Runtime.set_default_provider_model(...)` and the pooled client's
  `set_default_provider_model(...)` / `set_capability_defaults(...)` re-point
  the default provider/model without a restart.
- `abstractruntime.turn_grounding`: `stamp_user_turn_grounding()` writes the
  grounding envelope once into the stored user turn, so each turn's prompt is
  a byte prefix of the next one and provider prompt caches survive across
  turns. Session replay returns the stored bytes.
- `JsonFileRunStore.list_event_waiters(...)` / `list_event_waiters_by_prefix(...)`
  (optional `EventWaiterQueryableRunStore` protocol, also forwarded by
  `OffloadingRunStore`). `emit_event` uses this index instead of scanning
  every run file.
- The model receives a description of its workspace scope (default directory,
  access mode, extra roots and exclusions), and out-of-scope path errors list
  the authorized roots.
- VisualFlow: inline pin expressions (`node.data.pinExpressions`, sandboxed
  with RestrictedPython), `continueOnError` on effect nodes, a `write_chart`
  node, a `write_docx` node, image embedding and branded exports in PDF/DOCX
  renderers, and `shq` / `text_of` sandbox helpers.
- `abstractruntime.__version__`.
- The configured reasoning effort on AbstractCore's text capability route is
  applied when a call names no `thinking`.
- `config_facade.read_email_settings()` and `read_maintenance_settings()`.
- LLM results carry `route` (the provider, model and `base_url` that actually
  served the call, with a `mismatch` flag).
- In-process `on_token` streaming callbacks (`set_on_token`) on local and
  pooled clients; `read_idle_timeout_s` for LLM calls.
- `WAIT_EVENT` accepts a deadline; `_runtime.wait_until_streak` counts
  consecutive `WAIT_UNTIL` parks.
- `RuntimeHealth` counters, bounded run-vars growth for long-running runs,
  an indexed idempotency lookup, fair scheduling across several run stores
  (`scheduler/multi_store.py`), a steer sidecar store, and durable session
  conversation replay.
- `history_bundle`: a `detail="replay"` profile and in-band `warnings[]` when a
  bundle cannot be complete.
- Entity runtime (`abstractruntime.identity`): the per-entity home runtime,
  chat driver, visit workflow, life loop, diary and memory effects
  (`MEMORY_CONSOLIDATE`, `MEMORY_PROBE`, `MEMORY_TEND`, `LIFE_QUERY`,
  `ENTITY_TOOLS_QUERY`, `ENTITY_TOOLS_EXECUTE`), phase graph and entity tools.
  See `docs/entity-runtime.md`.
- Tool surfaces: `browser_probe` in the `web` toolset (asks for approval by
  default), a camera toolset registered when `abstractcamera` is installed,
  a `git_read_only@v1` approval refiner, and agora hub tools.

### Changed
- Dependency floors: `abstractcore[remote,tools,vision,voice,audio,music]>=2.13.41`,
  `abstractcore[all-apple]>=2.13.41` (`apple` extra),
  `abstractcore[all-gpu]>=2.13.41` (`gpu` extra), `AbstractMemory>=0.3.0`,
  `abstractsemantics>=0.0.5`. `RestrictedPython>=7.0` and `pyyaml>=6.0` are
  declared dependencies.
- The `gpu` extra's setuptools floor is `>=77.0.3` (was `>=80.10.2`), which
  vLLM's `setuptools<80` requirement can satisfy.
- Migration: with RestrictedPython installed, VisualFlow Code nodes always run
  under its policy. Augmented assignment on subscripts (`d["k"] += 1`) is
  refused; rewrite it as read, modify, write.
- The default iteration budget (`RuntimeConfig.max_iterations` and the
  Agent-node fallback) is 20. Workflow-declared values still win.
- LLM calls default to `read_idle_timeout_s=300`: a stream that delivers
  nothing for 5 minutes is aborted. Pass `read_idle_timeout_s: None` in
  `llm_kwargs` to disable it.
- Tool approval waits use a unique, replay-stable key per approval
  (`tool_approval:{run_id}:{node_id}:{effect_identity}`). Runs already
  waiting on an older key can still be approved.
- A connected VisualFlow node of an unknown type fails compilation with
  `UnknownNodeTypeError` instead of running as a no-op.
- Terminal ledger records are slimmer, the offloading ledger store is used by
  the durable factories, and hot-path store reads avoid full-document parses.
- Deterministic LLM client errors and prompt-cache binding failures are not
  retried.

### Fixed
- A per-call provider pin reaches the provider it names: pooled clients no
  longer hand the default endpoint's `base_url` / `api_key` to other providers.
- Catalog discovery works when the default text client cannot be built.
- A fresh install with no provider configured constructs its runtime; calls
  without a provider fail with a message naming what to configure.
- The session prompt-cache prefix is prepared with the `thinking` value the
  call generates with, and prompt-only calls under a runtime-derived key no
  longer append to their own cache.
- The remote client forwards `thinking`; streamed reasoning keeps the complete
  final text.
- Effect-only paths into a VisualFlow End node no longer copy runtime
  bookkeeping into the result.
- Visual `llm_call` nodes forward provider and model independently.
- Native tool calls are kept by the chat driver.
- `JsonlCommandStore.append` fsyncs before returning.
- Run-output offload reduces the largest children first, so a small answer
  stays inline next to a large scratchpad.
- The JSON run store cache is LRU-bounded; hash-chained ledgers no longer fork
  under concurrent handles.
- Entity-lane `execute_command` kills its whole process tree on timeout.
- PDF export renders scientific and typographic glyphs.

## [0.4.31] - 2026-08-27

Never published separately; these changes ship in 0.4.32.

### Added
- Host facade methods `get_memory_snapshot()`, `list_session_prompt_caches(session_id=None)`, and
  `clear_session_prompt_caches(session_id)`. Hosts can read the Core-owned host memory snapshot
  (RAM, process RSS, device allocation) and enumerate or clear live session prompt caches per
  session across local, multi-local, and remote runtimes. The three methods are optional in the
  LLM-client contract: a configured client that does not implement one still binds, and the facade
  answers `{"ok": false, "supported": false, ...}` for that call.
- Session attribution on derived prompt-cache keys. When Runtime injects the session-scoped
  prompt-cache key for a text/chat `LLM_CALL`, the client stamps `session_id`, `run_id`,
  `workflow_id`, `node_id`, and `namespace` into the cache entry's metadata after each generate —
  locally through the provider's key-meta contract, remotely via `POST /acore/prompt_cache/key_meta`
  (skipped for servers without the route). Caller-supplied `prompt_cache_key`s and binding keys are
  never stamped, so clearing a session cannot destroy caches shared across sessions. Session caches
  are not cleared automatically when a run ends; clearing stays an explicit host operation.
- Local `list_model_residency` merges AbstractCore's host-wide loaded-model sweep into
  text-generation listings: models resident on host-local provider servers (for example Ollama or
  LM Studio) now appear with `source: "provider_server"` even when they were not loaded through
  this runtime. Rows loaded by this runtime win deduplication and absorb the sweep's size fields.
- Model-residency locks. The host facade and all execution modes expose
  `lock_model_residency(...)`, `unlock_model_residency(...)`, and `get_context_estimate(...)`,
  each accepting an optional payload mapping and/or keyword arguments (keyword arguments win on
  conflicts). A locked model refuses `unload_model_residency` with a structured
  `{"ok": false, "error": "model_locked", ...}` payload instead of an exception; `force=true`
  unloads it, and the lock is released only after the unload succeeds. Lock requires
  provider-verified residency: a warm client or configured default alone is configuration, not
  memory, and locking a non-resident pair refuses with
  `{"ok": false, "error": "model_not_resident", ...}` (load with `lock: true` instead); unlock
  never requires residency, so a locked-but-since-evicted pair can always be released, and the
  Ollama keep-alive restore is skipped for a non-resident model so unlock never loads it back as
  a side effect. Local clients enforce the
  lock per `(provider, model)` pair, reinforce it best-effort with Ollama's `keep_alive` knob
  (reported under `provider_side`), refuse foreign runtime ids with a not-found payload, and
  exempt locked pairs from the multi-local pool eviction when the default provider/model is
  re-pointed (a locked pair that becomes the new default identity is rebuilt and its lock cleared,
  with a logged warning). Remote clients relay `POST /acore/models/lock|unlock` and convert the
  server's HTTP 409 refusals into the same structured payloads (`model_locked` on unload,
  `model_not_resident` on lock). The three methods are
  optional in the LLM-client contract, degrading to `{"ok": false, "supported": false, ...}` at
  the facade; locks apply to text-generation runtimes only.
- `MODEL_RESIDENCY` effect operations `lock` and `unlock`, with the same soft-fail semantics as
  the existing operations; on `unload`, `force` is forwarded only when authored in the effect
  payload.
- `get_context_estimate(...)` relays AbstractCore's analytical context-fit estimator: in-process
  for local clients (defaulting to the client's provider/model identity), via
  `GET /acore/models/context_estimate` for remote clients.
- Local text-residency rows carry AbstractCore's registry-declared `modalities` (omitted on a
  registry miss; `input.image` is stripped with `modalities_note: "vision_unusable"` when the
  provider reports its vision lane unusable), the serving host's identity (`host_id` /
  `host_name`), and runtime-owned lock truth (`locked` / `lockable`); provider claims cannot
  supply lock state. `pinned` on local rows is a truthful alias of `locked` (same value, Core
  parity), never the default-identity flag — `default` alone marks the client's default pair, so
  a configured capability default is no longer presented as pinned. Remote listings keep the
  Core server's row identity and are never re-stamped
  with the client's.

### Changed
- Residency rows normalize provider-reported `size` / `size_vram` extras to `size_bytes` /
  `size_vram_bytes` across local and remote listings; the original fields are kept.
- `unload_model_residency` also drops the runtime's client-side prompt-cache mirrors for the
  unloaded model, alongside AbstractCore's clearing of the in-provider cache stores.
- Raised the AbstractCore dependency floor to `abstractcore>=2.13.40`, the release that provides
  the shared memory/residency utility surface and the prompt-cache `key_meta` endpoint.

## [0.4.29] - 2026-06-14

### Changed
- Raised the AbstractCore dependency floor to `abstractcore>=2.13.38`, so Runtime's base and hardware install profiles depend on the released Core utility surface and synchronized Voice-backed capability floor.

## [0.4.28] - 2026-06-06

### Added
- VisualFlow `read_pdf` and `write_pdf` document nodes. `read_pdf` extracts PDF text/metadata with `pypdf`; `write_pdf` renders text or Markdown-style content to real PDF bytes with `reportlab` while keeping run state JSON-safe.
- Runtime discovery now exposes installed compatible vision adapters through `list_vision_adapters(...)`.
- Runtime's local/remote image and video media helpers now preserve task-specific batch generation controls (`count` / `n`, `seeds`) and ordered `lora_adapters`, and VisualFlow media nodes now lower those fields for image generation, image edit, text-to-video, and image-to-video.

### Changed
- Runtime's base dependency path no longer selects AbstractCore's media extra or direct PyMuPDF/PyMuPDF4LLM/PyMuPDF-layout packages for VisualFlow PDF support.
- Raised the AbstractCore dependency floor to `abstractcore>=2.13.37`, matching the released Core/Vision adapter, batch-generation, and media-parameter contract used by Runtime's base and hardware profiles.
- Forwarded newer Core/Vision controls such as `guidance_2`, `flow_shift`, and image-upscaler parameters through generated-media execution.
- The per-turn `<runtime_metadata>` prompt envelope is now temporal-only by default (`local_datetime` plus a country-free `display`). Full grounding remains in result metadata (`runtime_grounding`); operators can opt fields back into the prompt with `ABSTRACTRUNTIME_GROUNDING_PROMPT_FIELDS` (comma-separated subset of `local_datetime,timezone,country,user,display`).
- When the runtime injects a `<runtime_metadata>` envelope into the user turn, it now also appends a stable "RUNTIME GROUNDING" contract to the system prompt explaining that the envelope is machine context, not a user language/locale preference.

### Fixed
- Markdown-to-PDF rendering now handles ATX heading levels 1 through 6, preventing deeper headings such as `####` from appearing as literal paragraph text in generated PDFs.
- VisualFlow LLM Call and Agent structured outputs now expose the parsed object on the `data` output while preserving the existing textual `response` output.
- VisualFlow LLM Call nodes now preserve inline `resp_schema` / `response_schema` constraints when provider and model are left on Auto, so Gateway/Core default routing still receives a structured-output `response_model`.
- VisualFlow `answer_user` lowering now always emits a string `message` payload, preventing connected-but-null message inputs from creating invalid `ANSWER_USER` effects.
- Structured-output field descriptions from JSON Schema now survive Runtime's Pydantic response-model conversion, so providers receive the same guidance authored in Flow.
- Local subprocess media execution now supports task-compatible image edit and image upscaling inputs under the same durable image/video contract as in-process Runtime media calls.
- Runtime now ships its own workspace-path and file-filter helper modules instead of importing unreleased AbstractCore internals, so published installs and release CI use the same supported dependency surface.

## [0.4.27] - 2026-06-03

### Changed
- Raised the AbstractCore dependency floor to `abstractcore>=2.13.32` so Runtime hosts inherit provider endpoint profiles, route-specific multimodal defaults, and updated audio-understanding model metadata.
- Updated AbstractCore discovery integration to expose Gateway/Core capability defaults, provider endpoint profiles, and route-specific media catalogs to thin clients.

### Fixed
- LLM and generated-media execution now resolves Gateway/Core default provider and model selections when VisualFlow nodes leave provider/model on Auto.
- Media artifact resolution and VisualFlow generated-media calls now preserve uploaded artifacts and progress callbacks across Runtime/AbstractCore boundaries.

## [0.4.26] - 2026-05-31

### Changed
- Moved AbstractCore remote/tool/media capability integration and the MCP worker dependency set into the base `pip install abstractruntime` profile. Runtime now exposes only the base, `abstractruntime[apple]`, and `abstractruntime[gpu]` user install profiles for functionality vs. local-inferencer selection; the `abstractcore`, `multimodal`, `mcp-worker`, `all-apple`, and `all-gpu` extras are no longer part of the supported install surface.
- Raised the AbstractCore dependency floor to `abstractcore>=2.13.31` so Runtime installs inherit the latest remote-light media and Wan A14B vision contracts.

### Fixed
- Local text-to-video and image-to-video media-only calls now run in an isolated subprocess, preserving progress callbacks while preventing native MLX/Metal video failures from killing the Gateway/Runtime parent process.

## [0.4.25] - 2026-05-29

### Added
- File-backed artifact stores now expose `content_path(...)` for hosts that need a stable local path while in-memory stores continue to return `None`.

### Changed
- Minimum optional AbstractCore dependency floor is now `abstractcore>=2.13.30` (and matching `multimodal`, `mcp-worker`, and hardware-profile cascade extras), aligning Runtime with the latest Core media/plugin floors and image-to-image residency truth.

### Fixed
- Artifact-backed media resolution now preserves image roles such as `source` and `mask`, keeping image-edit and image-to-video requests wired correctly through AbstractCore.
- Model-residency discovery now treats `image_to_image` as its own vision task and deduplicates shared loaded-model records when task is omitted.

## [0.4.24] - 2026-05-26

### Added
- Runtime now surfaces AbstractCore/AbstractVision video generation through the existing generated-media boundary:
  - `LLM_CALL` output selectors for `{"modality":"video","task":"text_to_video"}` and `{"modality":"video","task":"image_to_video"}`
  - remote Core Server routing for `/v1/videos/generations` and `/v1/videos/edits`
  - durable run-facade helpers `generate_video(...)` and `image_to_video(...)`
  - VisualFlow node lowering for `generate_video` / `text_to_video` and `image_to_video`
- Provider progress callbacks are converted into JSON-safe `abstract.progress` ledger events during `LLM_CALL` execution without persisting Python callback objects.

### Changed
- Minimum optional AbstractCore dependency floor is now `abstractcore>=2.13.29` (and matching `multimodal`, `mcp-worker`, and hardware-profile cascade extras), aligning Runtime with Core video endpoints, video residency tasks, and AbstractVision 0.3.16 progress-capable generation.
- Runtime docs and AI-readable `llms.txt` / `llms-full.txt` now document text-to-video, image-to-video, and generated-media progress events.

## [0.4.23] - 2026-05-26

### Added
- Run lifecycle helpers and execution metric surfaces for VisualFlow execution and Gateway run retention workflows.
- Storage deletion primitives for durable run cleanup across the Runtime storage backends.

### Changed
- Minimum optional AbstractCore dependency floor is now `abstractcore>=2.13.28` (and matching `multimodal`, `mcp-worker`, and hardware-profile cascade extras), aligning Runtime with the latest Core capability defaults, MLX-Gen catalog, and OmniVoice discovery contracts.

### Fixed
- Effect invocation tracing now records generated-media and code-node execution details consistently across local and Gateway-hosted runs.

## [0.4.22] - 2026-05-23

### Changed
- Minimum optional AbstractCore dependency floor is now `abstractcore>=2.13.27` (and matching `multimodal`, `mcp-worker`, and hardware-profile cascade extras), aligning Runtime with the latest Core capability plugin floors and server contracts.

### Fixed
- Remote and VisualFlow music generation now fail closed on legacy `backend` / `music_backend` selectors and require `provider` / `music_provider` as the backend selector, matching AbstractCore Server `/v1/audio/music` validation.
- VisualFlow `generate_music` lowering now preserves boolean `structure_prompt` values (including explicit `False`) in the pending output selector, keeping the Flow/Gateway/Core contract consistent.

## [0.4.21] - 2026-05-22

### Added
- Public model-residency capability discovery on the AbstractCore host facade so hosts can branch on task support before showing warmup controls.
- Durable run-facade support for image edits through `edit_image(...)`.
- First-class VisualFlow lowering for `edit_image` / `image_to_image` and `generate_music` media nodes.
- A focused troubleshooting guide and repository code of conduct in the core documentation set.

### Changed
- Minimum optional AbstractCore dependency floor is now `abstractcore>=2.13.25`, matching the released Core validation for task-aware text/image/TTS/STT residency.
- Remote Runtime media execution now routes image edits through AbstractCore Server `/v1/images/edits` or provider-scoped `/{provider}/v1/images/edits`.
- Runtime no longer auto-derives session prompt-cache keys for non-text generated-media or transcription output selectors; explicit `prompt_cache_binding` remains supported.
- Local and remote model-residency responses now fail closed unless Core-owned residency truth verifies the loaded state.
- Runtime docs, backlog, ADR links, and AI-readable `llms.txt` / `llms-full.txt` now reflect the Core-owned residency boundary and current media node support.

### Fixed
- Artifact-backed media resolution now preserves image/audio role metadata without failing when content type metadata is absent.

## [0.4.20] - 2026-05-21

### Added
- Runtime now exposes a public host-local prompt-cache export/import admin surface on `get_abstractcore_host_facade(...)`:
  - `list_prompt_cache_exports(...)`
  - `prompt_cache_export(...)`
  - `prompt_cache_import(...)`

### Changed
- Local Runtime now owns the prompt-cache export root/catalog policy:
  - `~/.abstractruntime/prompt_cache_exports` by default
  - `<base_dir>/prompt_cache_exports` for `create_local_file_runtime(...)`
  - exact provider/model partitioning with Runtime-managed metadata sidecars
- Remote and hybrid runtimes now fail honestly for prompt-cache export/import admin with a structured local-only response instead of implying server-side support.
- Runtime docs and AI-readable `llms.txt` / `llms-full.txt` now document the secondary host-local export/import contract distinctly from the primary durable bloc/binding prompt-cache path.

## [0.4.19] - 2026-05-21

### Added
- Runtime now ships the missed standalone email comms wrapper/export layer for host-local operator surfaces:
  - `abstractruntime.integrations.abstractcore.comms_facade`
  - package-level email helper exports from `abstractruntime.integrations.abstractcore`

### Changed
- Host-facade email helpers now delegate through Runtime's own comms facade instead of importing `abstractcore.tools.comms_tools` directly in the facade method body.
- Runtime docs and AI-readable `llms.txt` / `llms-full.txt` now describe the standalone email comms facade/export layer alongside the existing host facade, Telegram wrappers, and durable run-owned comms sends.

## [0.4.18] - 2026-05-21

### Added
- Runtime now exposes the remaining Gateway-facing comms/Telegram package boundary through public Runtime wrappers:
  - host-local email helpers on `get_abstractcore_host_facade(...)`
  - host-local Telegram TDLib/bootstrap/global-client wrappers in `abstractruntime.integrations.abstractcore.telegram_facade`
- Outbound email and Telegram sends can now execute as durable Runtime-authored child runs through `get_abstractcore_run_facade(...)`:
  - `send_email(...)`
  - `send_telegram_message(...)`
  - `resume_tool_calls(...)` for approval-gated or passthrough tool waits

### Changed
- Runtime docs and AI-readable `llms.txt` / `llms-full.txt` now distinguish clearly between:
  - host-local operator comms helpers
  - durable run-owned outbound comms execution and replay semantics
- Outbound comms replay now follows the Runtime-owned truth model: recorded send requests and outcomes are replayed as data, not re-executed as external sends.

## [0.4.17] - 2026-05-21

### Added
- Runtime now surfaces AbstractCore-backed music through the same durable boundary as other generated artifacts:
  - host discovery snapshot methods for music providers/models
  - durable run-scoped `generate_music(...)`
  - artifact-backed normalized music outputs for local and remote Runtime paths

### Changed
- Minimum optional AbstractCore dependency floor is now `abstractcore>=2.13.24`.
- The `multimodal` extra now installs `abstractcore[remote,vision,voice,audio,music]>=2.13.24`, which includes AbstractMusic's lightweight remote ACE backend path via `abstractmusic>=0.1.4`.
- Runtime docs and AI-readable `llms.txt` / `llms-full.txt` now describe the shipped music boundary and the current `0030` Gateway cleanup scope more accurately.

## [0.4.16] - 2026-05-21

### Added
- Public durable AbstractCore bloc lifecycle operations on `get_abstractcore_host_facade(...)` across local, remote, and hybrid runtimes:
  - `list_blocs(...)`
  - `list_bloc_kv_artifacts(...)`
  - `delete_bloc_kv_artifact(...)`
  - `prune_bloc_kv_artifacts(...)`
  - `delete_bloc(...)`

### Changed
- Minimum optional AbstractCore dependency floor is now `abstractcore>=2.13.23`.
- Runtime's public docs and AI-readable `llms.txt` / `llms-full.txt` now describe the shipped durable bloc prompt-cache lifecycle boundary, including per-model KV artifacts and explicit cleanup controls.

## [0.4.15] - 2026-05-20

### Added
- Public AbstractCore Runtime facades for:
  - discovery/catalog snapshot queries (`get_abstractcore_discovery_facade(...)`)
  - prompt-cache and model-residency host control operations (`get_abstractcore_host_facade(...)`)
  - durable run-scoped child-run execution for image, TTS, STT, and direct LLM calls (`get_abstractcore_run_facade(...)`)
- `EffectType.MODEL_RESIDENCY` with AbstractCore-backed load/list/unload handling and VisualFlow lowering support for runtime-owned model residency control.

### Changed
- Minimum optional AbstractCore dependency floor is now `abstractcore>=2.13.20`.
- Local cached-vision discovery now depends on AbstractCore's public `get_local_vision_cache_catalog()` helper instead of private server internals.
- Local media residency no-op/unsupported responses now complete truthfully with warning/degraded metadata, and media-only results distinguish runtime orchestration identity from the actual media backend identity.
- Documentation was refreshed for the current Runtime/Core boundary, including root docs, ADR/backlog references, and AI-readable `llms.txt` / `llms-full.txt`.

## [0.4.14] - 2026-05-19

### Fixed
- Runtime extras that pull AbstractCore provider/tool dependencies now declare current compatible OpenAI/httpx/anyio bounds directly, preventing Python 3.10 pip installs from backtracking through the full OpenAI 1.x history.

## [0.4.13] - 2026-05-19

### Fixed
- The `multimodal` extra now uses AbstractCore's current remote/vision/voice/audio abstraction extras and declares the media document dependencies directly, avoiding Core's older narrow media constraints when combined with `[all-apple]` or `[all-gpu]`.
- Apple/GPU Runtime profile extras now bound setuptools to a modern version compatible with Torch's `<82` constraint, preventing resolver backtracking into broken legacy setuptools releases.

## [0.4.12] - 2026-05-19

### Fixed
- Remote AbstractCore transcription now uses provider-scoped audio routes such as `/{provider}/v1/audio/transcriptions` when an STT provider is selected.
- VisualFlow generated media nodes now keep LLM `provider`/`model` routing separate from image, TTS, and STT provider/model pins.

### Changed
- Minimum AbstractCore optional dependency floor is now `abstractcore>=2.13.15`.

## [0.4.11] - 2026-05-13

### Fixed
- AbstractCore effect-handler media materialization now preserves artifact `content_type`, media `type`, artifact ids, and safe filename extensions instead of dropping artifact-backed media to bare paths. This keeps generated WAV artifacts valid for downstream transcription nodes.
- Added explicit MIME extension aliases for common generated media types such as `audio/wav`, `image/png`, and `video/mp4` so platform MIME table differences do not create extensionless temp files.

### Changed
- Minimum AbstractCore optional dependency floor is now `abstractcore>=2.13.14`.


## [0.4.10] - 2026-05-12

### Fixed
- Generated media VisualFlow nodes now keep media model selection in the output spec and reserve LLM `provider`/`model` routing for explicit `runtime_provider`/`runtime_model` overrides.
- Legacy `provider`/`model` pins on image, TTS, and STT media nodes remain accepted as media selector fallbacks for existing flows.

### Changed
- Minimum AbstractCore optional dependency floor is now `abstractcore>=2.13.13` so generated media and audio catalog contracts stay aligned.

## [0.4.9] - 2026-05-09

### Changed
- Added `AbstractMemory>=0.2.6` as a base dependency so Runtime's
  `MEMORY_KG_*` effect contract always has the AbstractMemory TripleStore
  models available.

### Notes
- Runtime still does not depend on `AbstractMemory[lancedb]`. Hosts such as
  AbstractGateway choose the durable/vector memory backend, path, embeddings,
  and readiness policy.

## [0.4.8] - 2026-05-08

### Changed
- Minimum `abstractcore` optional dependency increased to `>=2.13.12`, and the
  semantics floor increased to `abstractsemantics>=0.0.3`.
- Added explicit hardware-profile cascade extras:
  `abstractruntime[apple]`, `abstractruntime[gpu]`,
  `abstractruntime[all-apple]`, and `abstractruntime[all-gpu]`.

### Notes
- Runtime still owns durable execution, not local model engines. These extras
  delegate to the matching AbstractCore profile so Gateway and root aggregate
  installs can compose a single profile vocabulary.

## [0.4.7] - 2026-05-08

### Changed
- Minimum `abstractcore` optional dependency increased to `>=2.13.11` for the `abstractcore`, `multimodal`, and `mcp-worker` extras so Runtime aligns with the current Core server-auth, provider-key, generated-media, and capability-catalog contracts.
- AbstractCore integration imports now fail fast when a stale local AbstractCore install is older than the 2.13.11 Gateway/Core deployment baseline.
- Documentation now makes the Gateway handoff explicit: hosts choose Runtime plus the Core/capability/memory profile, pass Core server URLs/auth headers deliberately, and keep provider clients, auth objects, model handles, and sessions out of durable runtime state.
- Runtime no longer reads Gateway-owned environment variables directly. Prompt-cache defaults use explicit Runtime state or `ABSTRACTRUNTIME_PROMPT_CACHE`, read-file attachment registration limits use explicit Runtime state/payload values or `ABSTRACTRUNTIME_MAX_ATTACHMENT_BYTES`, and workflow bundle registries use shared/framework or explicit directories.

### Testing
- Added packaging boundary coverage proving Runtime exposes no fake hardware profile extras (`apple`, `gpu`, `all-apple`, `all-gpu`) and keeps the Core floors aligned.
- Added import-boundary coverage proving the runtime kernel and package root do not import optional Core/Vision/Voice/Memory/Music stacks.
- Added a remote client regression test proving Gateway auth/provider-key environment variables are not inherited as AbstractCore server auth or provider-key headers.
- Added regression tests proving Gateway env vars alone do not enable prompt-cache keys, shrink attachment registration limits, or select workflow bundle registry directories.

## [0.4.6] - 2026-05-07

### Changed
- Minimum `abstractcore` optional dependency increased to `>=2.13.10` so Runtime picks up AbstractCore's async/sync text-generation output-selector parity in addition to the public output-selector contract.

### Fixed
- AbstractCore output-selector imports now fail fast when an older local AbstractCore install exposes the helper module but does not include the 2.13.10 async parity fix.

## [0.4.5] - 2026-05-07

### Changed
- Minimum `abstractcore` optional dependency increased to `>=2.13.9` so Runtime can use AbstractCore's public output-selector contract instead of mirroring provider-private multimodal selector logic.
- Runtime's AbstractCore output-spec adapter now delegates selector detection, normalization, generated-media detection, non-chat dispatch detection, and runtime metadata stripping to `abstractcore.core.output_specs`.

### Fixed
- Explicit `voice_clone` output specs no longer require a Runtime `ArtifactStore` before dispatch because AbstractCore exposes them as generated resources rather than binary media outputs.

## [0.4.4] - 2026-05-07

### Added
- **AbstractCore multimodal generation integration**:
  - `LLM_CALL` now forwards AbstractCore's unified `generate(..., output=...)` selector for image generation, TTS/voice output, and audio transcription
  - generated binary outputs are normalized into JSON-safe runtime results with ArtifactStore-backed refs instead of inline bytes
  - local runtimes can use AbstractCore capability plugins such as AbstractVision and AbstractVoice through the same runtime effect shape
  - remote runtimes support AbstractCore Server image generation, speech, and transcription endpoints, plus OpenAI-compatible chat media content arrays
- **Multimodal packaging extra**:
  - new `abstractruntime[multimodal]` extra installs `abstractcore[media,openai,vision,voice,audio]>=2.13.8`
- **VisualFlow LLM media selectors**:
  - LLM nodes lowered from VisualFlow can request generated media through `output` / `outputs` from node config or input data

### Changed
- Minimum `abstractcore` optional dependency increased to `>=2.13.8` for the unified multimodal response types.
- `LLM_CALL` accepts top-level `text`, top-level `output`, and top-level `outputs` as a runtime alias for AbstractCore `output`.
- `LLM_CALL.media` accepts one media item or a list; artifact refs are materialized to provider-ready temporary files before model calls.
- Remote AbstractCore clients now preserve existing OpenAI-style content arrays when adding media attachments.
- Remote AbstractCore clients now resolve ArtifactStore-backed media refs for direct client use, matching the runtime effect-handler path.
- VisualFlow LLM pending-call lowering now carries `output` / `outputs` selectors into runtime LLM effects.
- VisualFlow LLM result syncing now projects generated media artifacts into node outputs such as `outputs`, `resources`, `artifact_ref`, `artifact_id`, and `meta.output_mode`.

### Fixed
- Runtime artifact metadata (`run_id`, tags, artifact ids) is kept out of AbstractCore provider/capability kwargs while still being applied to stored generated media artifacts.
- Generated binary media now fails closed without an ArtifactStore instead of embedding base64 bytes in durable state.
- Remote image/TTS/STT calls no longer reuse the chat model unless an output-specific media model is supplied.
- Remote media inputs now either convert to a provider-ready content item or fail before dispatch; unsupported remote image edits, voice reference inputs, and non-file STT inputs are rejected explicitly.
- Turn-grounding injection now preserves structured multimodal message content arrays instead of stringifying them.
- Session-scoped prompt-cache key derivation now uses the effective AbstractCore client provider/model identity when an `LLM_CALL` payload omits explicit provider/model overrides.

### Documentation
- Documented multimodal `LLM_CALL` payloads, artifact-backed response shape, remote endpoint coverage, cached-session/prompt-cache boundaries, and the `abstractruntime[multimodal]` extra in the AbstractCore integration guide, API reference, architecture guide, FAQ, README, getting-started guide, docs index, and AI-ready `llms*.txt` files.
- Added a planned workspace/media access policy item covering default workspace-only access, explicit user allow/deny paths, and a conscious full-machine access mode for long-running agency deployments.

### Testing
- Added focused coverage for multimodal response normalization, artifact-backed generated media, media-only transcription calls, remote image/TTS/STT endpoints, remote media guardrails, remote chat media content arrays, content-array prompt extraction, direct remote artifact-ref media resolution, text-alias routing, provider-request redaction, and runtime metadata/tag boundaries.
- Added coverage for effective prompt-cache key identity and VisualFlow LLM media selector/result projection.

## [0.4.3] - 2026-05-06

### Added
- **AbstractCore prompt-cache control plane**:
  - local, multi-local, and remote LLM clients expose `get_prompt_cache_capabilities`, `get_prompt_cache_stats`, `prompt_cache_set`, `prompt_cache_update`, `prompt_cache_fork`, `prompt_cache_clear`, and `prompt_cache_prepare_modules`
  - local clients can maintain compartmentalized `system | tools | history` prompt-cache modules when providers support `local_control_plane`
  - remote clients proxy `/acore/prompt_cache/*` endpoints for gateway/CLI hosts
- **Artifact-backed media for AbstractCore LLM calls**:
  - local and remote AbstractCore clients can resolve runtime artifact refs into provider-ready media inputs
  - AbstractCore runtime factories now pass the runtime artifact store into LLM clients
- **Durable tool approval execution**:
  - `ToolApprovalPolicy` and `ApprovalToolExecutor` support safe auto-approval, durable approval waits, and approved re-execution
  - runtime factories expose the configured tool executor for approval-style `TOOL_CALLS` resumes
- **VisualFlow multi-entry lowering**:
  - authoring graphs with multiple incoming `exec-in` routes can be lowered into internal `join_exec` and `path_mux` nodes
  - per-entry input overrides survive pause/resume and file-store restart scenarios

### Changed
- AbstractCore remote provider-key overrides now use `X-AbstractCore-Provider-API-Key` headers instead of body/query `api_key` fields rejected by current AbstractCore servers.
- AbstractCore LLM clients keep per-turn grounding out of stable system prompts, coalesce leading system messages, strip internal tool-activity system messages, and propagate trace metadata headers.
- AbstractCore runtime factories expose the underlying LLM client for host-side control-plane operations and continue to honor AbstractCore timeout/config defaults.
- Default runtime iteration budget increased from 25 to 50.
- Minimum AbstractCore optional dependency increased to `>=2.13.5` so the documented prompt-cache control plane, hardened server auth, provider-key header routing, Telegram tools, and current model/provider behavior are available by default.
- Documentation: align version references with `pyproject.toml` (0.4.3), document AbstractCore prompt-cache operations, update remote provider-key guidance, and add concrete VisualFlow multi-entry authoring metadata.
- CI/release automation now builds the package and docs on normal CI and exposes a manual-only guarded release path for PyPI, GitHub Releases, and the docs site.

### Fixed
- VisualFlow While nodes again route `condition=true` to Loop and `condition=false` to Done/parent/complete after the execution-handle tracking refactor.
- Tool approval resumes now execute approved calls in-runtime when configured, return structured tool errors when denied or unavailable, and append completion ledger records for ledger-only replay clients.
- JSONL ledger listing now recovers concatenated JSON records defensively.
- `TOOL_CALLS` now emits durable warnings for missing or duplicate tool call ids.
- Optional VisualFlow fixture tests now skip cleanly when assessment fixtures are absent.

### Testing
- Added focused coverage for prompt-cache module preparation/rebuilds, remote prompt-cache proxying, artifact-backed media, tool approval waits/resumes, JSONL ledger recovery, remote provider-key headers, VisualFlow multi-entry prompt overrides, direct effect re-entry, same-predecessor route handles, stale route metadata, join-only fan-in, and While routing regressions.

## [0.4.2] - 2026-02-08

### Changed
- **Dependencies**:
  - bump minimum `abstractcore` / `abstractcore[tools]` to `>=2.11.8` (`pyproject.toml`)
  - bump minimum `abstractsemantics` to `>=0.0.2` (`pyproject.toml`)

## [0.4.1] - 2026-02-04

### Added
- **Durable prompt metadata for EVENT waits**:
  - `WAIT_EVENT` effects may include optional `prompt`, `choices`, and `allow_free_text` fields.
  - The runtime persists these fields onto `WaitState` so hosts (including remote/thin clients) can render a durable ask+wait UX without relying on in-process callbacks.
- **Rendering utilities** (`abstractruntime.rendering`):
  - `stringify_json(...)` + `JsonStringifyMode` to render JSON/JSON-ish values into strings with `none|beautify|minified` modes.
  - `render_agent_trace_markdown(...)` to render runtime-owned `node_traces` scratchpads into a complete, review-friendly Markdown timeline.
- **Documentation refresh**:
  - clearer entrypoints: `README.md` → `docs/getting-started.md`
  - new reference docs: `docs/api.md`, `docs/faq.md`, `docs/architecture.md`
  - maintainer-facing orientation: `llms.txt`, `llms-full.txt`
  - new repo policies: `CONTRIBUTING.md`, `SECURITY.md`, `ACKNOWLEDGMENTS.md`

### Fixed
- Normalize AbstractCore tool specs for skim tools so `paths` is always an array parameter (improves JSON schema consistency for tool callers).

## [0.4.0] - 2025-01-06

### Added

- **Active Memory System** (`abstractruntime.memory.active_memory`): Complete MemAct agent memory module
  - Runtime-owned `ACTIVE_MEMORY_DELTA` effect for structured Active Memory updates (used by agents via `active_memory_delta` tool)
  - JSON-safe durable storage in `run.vars["_runtime"]["active_memory"]`
  - Memory modules: MY PERSONA, RELATIONSHIPS, MEMORY BLUEPRINTS, CURRENT TASKS, CURRENT CONTEXT, CRITICAL INSIGHTS, REFERENCES, HISTORY
  - Active Memory v9 format with natural-language markdown rendering (not YAML) to reduce syntax contamination
  - All components render into system prompt by default (prevents user-role pollution on native-tool providers)

- **MCP Worker** (`abstractruntime-mcp-worker`): Standalone stdio-based MCP server for AbstractRuntime tools
  - Exposes AbstractRuntime's default toolsets as MCP tools via stdio transport
  - Human-friendly logging to stderr with ANSI color support
  - Security: allowlist-based command execution safety (`TOOL_WAIT` effect for dangerous commands)
  - New optional dependency: `abstractruntime[mcp-worker]` (includes `abstractcore[tools]`)
  - Entry point: `abstractruntime-mcp-worker` CLI script

- **Evidence Capture System** (`abstractruntime.evidence.recorder`): Always-on provenance-first evidence recording
  - Automatically records evidence for external-boundary tools: `web_search`, `fetch_url`, `execute_command`
  - Evidence stored as artifact-backed records indexed as `kind="evidence"` in `RunState.vars["_runtime"]["memory_spans"]`
  - Runtime helpers: `Runtime.list_evidence(run_id)` and `Runtime.load_evidence(evidence_id)`
  - Keeps RunState JSON-safe by storing large payloads in ArtifactStore with refs

- **Ledger Subscriptions**: Real-time step append events via `Runtime.subscribe_ledger()`
  - `create_local_runtime`, `create_remote_runtime`, `create_hybrid_runtime` now wrap LedgerStore with `ObservableLedgerStore` by default
  - Hosts can receive real-time notifications when steps are appended to ledger

- **Durable Custom Events (Signals)**:
  - `EMIT_EVENT` effect to dispatch events and resume matching `WAIT_EVENT` runs
  - Extended `WAIT_EVENT` to accept `{scope, name}` payloads (runtime computes stable `wait_key`)
  - `Scheduler.emit_event(...)` host API for external event delivery (session-scoped by default)

- **Orchestrator-Owned Timeouts** (AbstractCore integration):
  - Default **LLM timeout**: 7200s per `LLM_CALL` (not per-workflow), enforced by `create_*_runtime` factories
  - Default **tool execution timeout**: 7200s per tool call (not per-workflow), enforced by ToolExecutor implementations

- **Tool Executor Enhancements** (`MappingToolExecutor`):
  - **Argument canonicalization**: Maps common parameter name variations (e.g., `file_path`/`filepath`/`path`) to canonical names
  - **Filename aliases**: Supports `target_file`, `file_path`, `filepath`, `path` as aliases for file operations
  - **Error output detection**: Detects structured error responses (`{"success": false, ...}`) from tools
  - **Argument sanitization**: Cleans and validates tool call arguments
  - **Timeout support**: Per-tool execution timeouts with configurable limits

- **Memory Query Enhancements** (`MEMORY_QUERY` effect):
  - Tag filters with **AND/OR** modes (`tags_mode=all|any`) and **multi-value** keys (`tags.person=["alice","bob"]`)
  - Metadata filters for **authors** (`created_by`) and **locations** (`location`, `tags.location`)
  - Span records now capture `created_by` for `conversation_span`, `active_memory_span`, `memory_note` when `actor_id` available
  - `MEMORY_NOTE` accepts optional `location` field
  - `MEMORY_NOTE` supports `keep_in_context=true` flag to immediately rehydrate stored note into `context.messages`

- **Package Dependencies**:
  - New optional dependency: `abstractruntime[abstractcore]` (enables `abstractruntime.integrations.abstractcore.*`)
  - New optional dependency: `abstractruntime[mcp-worker]` (includes `abstractcore[tools]>=2.6.8`)

### Changed

- **LLM Client Enhancements**:
  - Tool call parsing refactored for better robustness and error handling
  - Streaming support with timing metrics (TTFT, generation time)
  - Response normalization preserves JSON-safe `raw_response` for debugging
  - Always attaches exact provider request payload under `result.metadata._provider_request` for every `LLM_CALL` step

- **Runtime Core** (902 lines changed):
  - Enhanced resume handling for paused/cancelled runs
  - Improved subworkflow execution with async+wait support
  - Better observable ledger integration

### Fixed

- **Cancellation is Terminal**: `Runtime.tick()` now treats `RunStatus.CANCELLED` as terminal and will not progress cancelled runs
- **Control-Plane Safety**: `Runtime.tick()` stops without overwriting externally persisted pause/cancel state (used by AbstractFlow Web)
- **Atomic Run Checkpoints**: `JsonFileRunStore.save()` writes via temp file + atomic rename to prevent partial/corrupt JSON under concurrent writes
- **START_SUBWORKFLOW async+wait**: Support for `async=true` + `wait=true` to start child run without blocking parent tick, while keeping parent in durable SUBWORKFLOW wait
- **ArtifactStore Run-Scoped Addressing**: Artifact IDs namespaced to run when `run_id` provided (prevents cross-run collisions, preserves purge-by-run semantics)
- **AbstractCore Integration Imports**: `LocalAbstractCoreLLMClient` imports `create_llm` robustly in monorepo namespace-package layouts
- **Token Limit Metadata**: `_limits.max_output_tokens` falls back to model capabilities when not configured (runtime surfaces explicit per-step output budget)
- **Token-Cap Normalization Boundary**: Removed local `max_tokens → max_output_tokens` aliasing from AbstractRuntime's AbstractCore client (AbstractCore providers own this mapping)

### Testing

- **25 new/modified test files** covering:
  - Active Memory functionality
  - MCP worker (logging, security, stdio communication)
  - Evidence recorder
  - Memory query rich filters
  - Tool executor (canonicalization, filename aliases, timeouts, error detection)
  - LLM client tool call parsing
  - Runtime configuration and subworkflow handling
  - Packaging extras validation

### Statistics

- **33 commits** improving memory systems, MCP integration, evidence capture, and tool execution
- **45 files changed**: 5,788 insertions, 286 deletions
- **6,074 total lines changed** across the codebase
- **3 new modules**: `active_memory.py`, `evidence/recorder.py`, `mcp_worker.py`

## [0.2.0] - 2025-12-17

### Added

#### Core Runtime Features
- **Durable Workflow Execution**: Start/tick/resume semantics for long-running workflows that survive process restarts
- **WorkflowSpec**: Graph-based workflow definitions with node handlers keyed by ID
- **RunState**: Durable state management (`current_node`, `vars`, `waiting`, `status`)
- **Effect System**: Side-effect requests including `LLM_CALL`, `TOOL_CALLS`, `ASK_USER`, `WAIT_EVENT`, `WAIT_UNTIL`, `START_SUBWORKFLOW`
- **StepPlan**: Node execution plans that define effects and state transitions
- **Explicit Waiting States**: First-class support for pausing execution (`WaitReason`, `WaitState`)

#### Scheduler & Automation
- **Built-in Scheduler**: Zero-config background scheduler with polling thread for automatic run resumption
- **WorkflowRegistry**: Mapping from workflow_id to WorkflowSpec for dynamic workflow resolution
- **ScheduledRuntime**: High-level wrapper combining Runtime + Scheduler with simplified API
- **create_scheduled_runtime()**: Factory function for zero-config scheduler creation
- **Event Ingestion**: Support for external event delivery via `scheduler.resume_event()`
- **Scheduler Stats**: Built-in statistics tracking and callback support

#### Storage & Persistence
- **Append-only Ledger**: Execution journal with `StepRecord` entries for audit/debug/provenance
- **InMemoryRunStore**: In-memory run state storage for development and testing
- **InMemoryLedgerStore**: In-memory ledger storage for development and testing
- **JsonFileRunStore**: File-based persistent run state storage (one file per run)
- **JsonlLedgerStore**: JSONL-based persistent ledger storage
- **QueryableRunStore**: Interface for listing and filtering runs by status, workflow_id, actor_id, and time range
- **Artifacts System**: Storage for large payloads (documents, images, tool outputs) to avoid bloating checkpoints
  - `ArtifactStore` interface with in-memory and file-based implementations
  - `ArtifactRef` type for referencing stored artifacts
  - Helper functions: `artifact_ref()`, `is_artifact_ref()`, `get_artifact_id()`, `resolve_artifact()`, `compute_artifact_id()`

#### Snapshots & Bookmarks
- **Snapshot System**: Named, searchable checkpoints of run state for debugging and experimentation
- **SnapshotStore**: Storage interface for snapshots with metadata (name, description, tags, timestamps)
- **InMemorySnapshotStore**: In-memory snapshot storage for development
- **JsonSnapshotStore**: File-based snapshot storage (one file per snapshot)
- **Snapshot Search**: Filter by run_id, tag, or substring match in name/description

#### Provenance & Accountability
- **Hash-Chained Ledger**: Tamper-evident ledger with `prev_hash` and `record_hash` for each step
- **HashChainedLedgerStore**: Decorator for adding hash chain verification to any ledger store
- **verify_ledger_chain()**: Verification function that detects modifications or reordering of ledger records
- **Actor Identity**: `ActorFingerprint` for attribution of workflow execution to specific actors
- **actor_id tracking**: Support for actor_id in both RunState and StepRecord for accountability

#### AbstractCore Integration
- **LLM_CALL Effect Handler**: Execute LLM calls via AbstractCore providers
- **TOOL_CALLS Effect Handler**: Execute tool calls with support for multiple execution modes
- **Three Execution Modes**:
  - **Local**: In-process AbstractCore providers with local tool execution
  - **Remote**: HTTP to AbstractCore server (`/v1/chat/completions`) with tool passthrough
  - **Hybrid**: Remote LLM calls with local tool execution
- **Convenience Factories**: `create_local_runtime()`, `create_remote_runtime()`, `create_hybrid_runtime()`
- **Tool Execution Modes**:
  - Executed mode (trusted local) with results
  - Passthrough mode (untrusted/server) with waiting semantics
- **Layered Coupling**: AbstractCore integration as opt-in module to keep kernel dependency-light

#### Effect Policies & Reliability
- **EffectPolicy Protocol**: Configurable retry and idempotency policies for effects
- **DefaultEffectPolicy**: Default implementation with no retries
- **RetryPolicy**: Configurable retry behavior with max_attempts and backoff
- **NoRetryPolicy**: Explicit no-retry policy
- **compute_idempotency_key()**: Ledger-based deduplication to prevent duplicate side effects after crashes

#### Examples & Documentation
- **7 Runnable Examples**:
  - `01_hello_world.py`: Minimal workflow demonstration
  - `02_ask_user.py`: Pause/resume with user input
  - `03_wait_until.py`: Scheduled resumption with time-based waiting
  - `04_multi_step.py`: Branching workflow with conditional logic
  - `05_persistence.py`: File-based storage demonstration
  - `06_llm_integration.py`: AbstractCore LLM call integration
  - `07_react_agent.py`: Full ReAct agent implementation with tools
- **Comprehensive Documentation**:
  - Architecture Decision Records (ADRs) for key design choices
  - Integration guides for AbstractCore
  - Detailed documentation for snapshots and provenance
  - Limits and constraints documentation
  - ROADMAP with prioritized next steps

### Technical Details

#### Architecture
- **Layered Design**: Clear separation between kernel, storage, integrations, and identity
- **Dependency-Light Kernel**: Core runtime remains stable with minimal dependencies
- **Graph-Based Execution**: All workflows represented as state machines/graphs for visualization and composition
- **JSON-Serializable State**: All run state and vars must be JSON-serializable for persistence

#### Testing
- Run the test suite with `python -m pytest -q` (see `docs/manual_testing.md`).

#### Compatibility
- **Python 3.10+**: Supports Python 3.10, 3.11, 3.12, and 3.13

### Known Limitations

- Snapshot restore does not guarantee safety if workflow spec or node code has changed
- Subworkflow support (`START_SUBWORKFLOW`) is implemented but undergoing refinement
- Cryptographic signatures (non-forgeability) not yet implemented - current hash chain provides tamper-evidence only
- Remote tool worker service not yet implemented

### Design Decisions

- **Kernel stays dependency-light**: Enables portability, stability, and clear integration boundaries
- **AbstractCore integration is opt-in**: Layered coupling prevents kernel breakage when AbstractCore changes
- **Hash chain before signatures**: Provides immediate value without key management complexity
- **Built-in scheduler (not external)**: Zero-config UX for simple cases
- **Graph representation for all workflows**: Enables visualization, checkpointing, and composition

### Notes

AbstractRuntime is the durable execution substrate designed to pair with AbstractCore, AbstractAgent, and AbstractFlow. It enables workflows to interrupt, checkpoint, and resume across process restarts, making it suitable for long-running agent workflows that need to wait for user input, scheduled events, or external job completion.

## [0.0.1] - Initial Development

Initial development version with basic proof-of-concept features.

[Unreleased]: https://github.com/lpalbou/abstractruntime/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/lpalbou/abstractruntime/compare/v0.4.35...v0.5.0
[0.4.35]: https://github.com/lpalbou/abstractruntime/compare/v0.4.34...v0.4.35
[0.4.34]: https://github.com/lpalbou/abstractruntime/compare/v0.4.33...v0.4.34
[0.4.33]: https://github.com/lpalbou/abstractruntime/compare/v0.4.32...v0.4.33
[0.4.32]: https://github.com/lpalbou/abstractruntime/compare/v0.4.29...v0.4.32
[0.4.31]: https://github.com/lpalbou/abstractruntime/compare/v0.4.29...v0.4.31
[0.4.29]: https://github.com/lpalbou/abstractruntime/compare/v0.4.28...v0.4.29
[0.4.28]: https://github.com/lpalbou/abstractruntime/compare/v0.4.27...v0.4.28
[0.4.27]: https://github.com/lpalbou/abstractruntime/compare/v0.4.26...v0.4.27
[0.4.26]: https://github.com/lpalbou/abstractruntime/compare/v0.4.25...v0.4.26
[0.4.25]: https://github.com/lpalbou/abstractruntime/compare/v0.4.24...v0.4.25
[0.4.24]: https://github.com/lpalbou/abstractruntime/compare/v0.4.23...v0.4.24
[0.4.23]: https://github.com/lpalbou/abstractruntime/compare/v0.4.22...v0.4.23
[0.4.22]: https://github.com/lpalbou/abstractruntime/compare/v0.4.21...v0.4.22
[0.4.21]: https://github.com/lpalbou/abstractruntime/compare/v0.4.20...v0.4.21
[0.4.20]: https://github.com/lpalbou/abstractruntime/compare/v0.4.19...v0.4.20
[0.4.19]: https://github.com/lpalbou/abstractruntime/compare/v0.4.18...v0.4.19
[0.4.18]: https://github.com/lpalbou/abstractruntime/compare/v0.4.17...v0.4.18
[0.4.17]: https://github.com/lpalbou/abstractruntime/compare/v0.4.16...v0.4.17
[0.4.16]: https://github.com/lpalbou/abstractruntime/compare/v0.4.15...v0.4.16
[0.4.15]: https://github.com/lpalbou/abstractruntime/compare/v0.4.14...v0.4.15
[0.4.14]: https://github.com/lpalbou/abstractruntime/compare/v0.4.13...v0.4.14
[0.4.13]: https://github.com/lpalbou/abstractruntime/compare/v0.4.12...v0.4.13
[0.4.12]: https://github.com/lpalbou/abstractruntime/compare/v0.4.11...v0.4.12
[0.4.11]: https://github.com/lpalbou/abstractruntime/compare/v0.4.10...v0.4.11
[0.4.10]: https://github.com/lpalbou/abstractruntime/compare/v0.4.9...v0.4.10
[0.4.9]: https://github.com/lpalbou/abstractruntime/compare/v0.4.8...v0.4.9
[0.4.8]: https://github.com/lpalbou/abstractruntime/compare/v0.4.7...v0.4.8
[0.4.7]: https://github.com/lpalbou/abstractruntime/compare/v0.4.6...v0.4.7
[0.4.6]: https://github.com/lpalbou/abstractruntime/compare/v0.4.5...v0.4.6
[0.4.5]: https://github.com/lpalbou/abstractruntime/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/lpalbou/abstractruntime/releases/tag/v0.4.4
[0.4.3]: https://github.com/lpalbou/abstractruntime/releases/tag/v0.4.3
[0.4.2]: https://github.com/lpalbou/abstractruntime/releases/tag/v0.4.2
[0.4.1]: https://github.com/lpalbou/abstractruntime/releases/tag/v0.4.1
[0.4.0]: https://github.com/lpalbou/abstractruntime/releases/tag/v0.4.0
[0.0.1]: https://github.com/lpalbou/abstractruntime/releases/tag/v0.0.1
