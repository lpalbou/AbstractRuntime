# Changelog

All notable changes to AbstractRuntime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Unreleased changes live in [`changelog.d/`](changelog.d/README.md), one file per pull request;
`python scripts/changelog.py assemble X.Y.Z` writes them here at release.

## [Unreleased]

## [0.19.6] - 2026-11-14

### Changed

- Requires AbstractCore 2.35.2 and AbstractMemory 0.5.2. The Core configuration facade forwards explicit single-repository Hugging Face details and reports unsupported older Core versions clearly.

### Fixed

- Diary projection and replay retain the original resolution verdict and available target identifiers. Private titles remain omitted. Temporary projection failures retain the diary act with a labelled warning and can be retried; existing projections are not backfilled.
- Cancellation is protected against stale Runtime state publication, including completion and wait-resume boundaries, without holding the provider execution lock. Memory-owner conflicts return a refusal for the target rather than changing the executing run's identity.

## [0.19.5] - 2026-11-11

### Fixed

- **A question the visit answered is marked resolved** (`identity/visit_workflow.py`, `identity/reflection.py`,
  `identity/act_only.py`, `identity/diary.py`, `identity/chat.py`; root backlog 1206). Operator 2026-11-11 02:00: an
  entity asked him a question in a visit, he answered, nothing marked it resolved, no badge appeared, and the
  question stayed under OPEN QUESTIONS ("it's important as if unresolved, this is probably looked back into its
  memory"; "investigate and fix"). Memory already resolves a question when a later diary entry carries
  `resolves=<its entry id>` and the book confirms the claim (`resolves_status`), but the visit lane never asked for
  it. Now:
  - the visit's look-back lists the entity's open questions that this visit asked or touched (a question it kept
    during the visit, or one a turn's recall showed it), each by entry id and title as formed, read from memory's
    own open-question set (`abstractmemory.diary.open_questions`, scope `diary`, once per close), and offers a
    ```` ```resolve ```` block: `resolves=<the question's id>`, `evidence=<the numbered moment that answered it>`,
    then the answer in the entity's words. The block is captured at the result boundary like a ```` ```diary ````
    block and written through the same `DIARY_WRITE` as a note carrying `resolves=`; its `evidence=` numbers link
    the entry to those moments. A block naming no question or giving no answer is refused, said, with the words
    kept in its marker. The model declares the resolution, the book confirms it; nothing reads the entity's words.
  - the look-back names a resolution a turn already made (the resolution-to-lesson cue the chat lane had), and the
    look-back's own capture warnings (a refused claim) join the close's `reflection_notices`.
  - the act-only capture's metadata of an entry that claimed `resolves=` carries `resolves` and `resolves_status`;
    when the book confirms an open question or problem, the `DIARY_WRITE` result and that metadata also carry
    `resolved_record_id` (the question's graph id) and `resolved_title` (its title as formed; `""` and
    `resolved_visibility: "private"` for a private question).
  - a visit turn whose diary entry resolves an open question serves a second notice beside the entry's own:
    `{"kind": "question_resolved", "record_id": <the question's graph id>, "title": <its title>}`
    (`chat.question_resolved_notice`, which a host applies to the look-back's captured `diary_entries` for the
    close). The chat lane's prompt and notices are unchanged.

## [0.19.4] - 2026-11-11

### Fixed

- **A visit with no word exchanged is not remembered** (`identity/visit_workflow.py`, `identity/track_record.py`,
  `identity/acquaintances.py`). Operator ruling 2026-11-11 00:40: "(a visit opened and closed without a word) that
  should not create a memory if no word was shared" and "a zero-turn visit ... no and it should not even be
  remembered, fix it". A visit opened and closed (by the operator, the idle deadline or the pause freeze) with no
  turn already formed no record and ran no model call (REFLECT has no sheet to look back on); that is now pinned,
  row for row over every table of the memory store. What still counted it was the life's derived counts: the
  track record listed it as a completed visit (and in the reputation's completion rate) and the acquaintances
  counted an interaction with the visitor. `visit_without_words(run)` names the rule (a visit run whose
  `_visit.turn_n` is 0 or absent; a malformed count raises), and `track_record.is_wordless_visit` keeps such a run
  out of both: it is neither a counted run nor an interaction, and an open visit counts from its first line. The
  ledger keeps the run's open and close rows (the ledger is the record of the run). A visit with one line or more
  is unchanged.

## [0.19.3] - 2026-11-11

### Fixed

- **Own time withdrawn ends on the turn in flight** (`identity/life.py`, `identity/chat.py`; operator 2026-11-10
  23:09: "if I remove the personal time, it's non negotiable; but it's ok to let the entity finish its turn").
  When the operator turned own time off (the gateway writes the grant `disabled`, then the stop command), the loop
  honored the stop at the next tick boundary but kept working past the revoke: the tick's look-back already under
  way still formed its summary, interest and diary entry, and the day's close then ran a second model look-back
  (on the live home: 22:03 revoke, look-back records at 22:07, a day summary and diary at 22:11-22:12, exit 22:12).
  Now the turn in flight finishes whole and nothing after it runs:
  - a revoke seen when the turn ends skips the tick's look-back and the authoring drain;
  - a look-back whose model call was in flight keeps nothing of its reply (`ChatSession.withdrawn_check`, set by
    the loop; `_reflect_over(abandon_if_withdrawn=True)` for the tick and day look-backs, never a salvage);
  - the day closes without its look-back.
  The day's look-back is deferred, never lost: `pending_reflection.json` carries the whole day to the next open
  (a visit, the next own-time day). "Withdrawn" is the operator's stop (the stop command, the STOP file, the kill
  switch) or, on a personal day, the grant revoked or expired; a work day answers to the stop only. A consumed stop
  command is now remembered by the loop (`_should_stop` treats `stop_command` as final, like the kill switch).
  A normal close (rest, the day's last tick) still runs its look-back.

## [0.19.2] - 2026-11-10

### Added

- **`memory_facade.community_keys_of`** (AbstractMemory 0.5.1's membership-only read, re-exported unchanged like
  `communities` and `community_of`): `community_keys_of(system, record_ids, *, scopes, owner_id)` answers
  `{record_id: topic key or None}` for many records in one call, from memory's in-process community index. A host
  that already holds records (a recall, a search result) names their topics with it instead of reading the whole
  communities payload.
- **Three visit ledger fields for a live view of the turn** (read by the Entity app's live activity panel, root
  backlog 1196). All additive; a reader without them shows what it showed before.
  - `topic` (string or null) on every handle of a visit's `memory_recall` record (`result.handles[]`, the passive
    recall and the recall of a live correction): the stable key of the topic (memory community) the memory is in
    now, null when it is in none. The visit asks for it with `with_topics: true` in the `MEMORY_RECALL` payload; the
    seam handler (`integrations/abstractmemory/seam_handlers.py`) makes ONE membership-only read per recall,
    AbstractMemory 0.5.1's `community_keys_of` with the handles' graph ids, answered from memory's in-process
    community index. Hot-path cost: about 50 µs for 44 handles at 2,000 and at 20,000 records on a synthetic home
    (54 topics); the first call in a process loads the index once (28 ms and 39 ms there). It never reads the
    communities payload (which computes every topic's facts: 19 ms and 190 ms on the same homes) and never calls
    `community_of` per handle (461 ms and 3.2 s). A lookup that fails leaves the recall whole, without `topic`, and
    says why in its `warnings`. The model's prompt is unchanged.
  - `iteration` and `max_iterations` (int) on the `effect.payload` of the `reason` model-call record (the react
    visit arm): this call's place in the turn's reasoning loop, from 1, and the turn's cap, read from the runtime's
    `_limits` when the call is planned. Metadata only: no handler sends them to a provider.
  - `hits` (int) on a `search_memory` result: the graph and book matches its text counts.
    `HomeMemoryReader.search_memory` answers a `SearchMemoryText` (a `str` with `hits`), `execute_tool_elections`
    copies it onto the new `ToolElection.hits`, and the `ENTITY_TOOLS_EXECUTE` result entries carry it. A host that
    builds its own tool result entries (AbstractGateway's visit `TOOL_CALLS` executor) copies `election.hits` as it
    copies `election.artifact_id`.
- **The own-time loop's failures stop says why** in `loop_status`: `note` = "Stopped after three failed ticks:
  <the last failure, as the provider gave it>. His memory is intact." (`failures_stop_note`, `LifeReport.last_failure`,
  `write_loop_status(note=)`; the next status write clears it).

### Changed

- **AbstractMemory floor 0.5.1** (`pyproject.toml`): the visit recall's topic stamp calls `community_keys_of`, which
  AbstractMemory 0.5.0 does not have.
- **The crashed-loop note is a plain sentence for end users** (`loop_process_status`, shown verbatim on the Entity
  app's card): "His own time stopped without a word (the process is gone: a crash or a reboot). His memory is
  intact." (`life.CRASHED_LOOP_NOTE`), in place of "loop_status said running but the process is gone (crash or
  reboot)". "process is gone" stays in it: AbstractGateway's entity_phase reads a crash from those words.

## [0.19.1] - 2026-11-10

### Added

- **The entity's self-knowledge brief: what it knows about, at a glance** (`identity/self_brief.py`,
  `life.build_consolidator`, `chat.compose_system_base`). An entity cannot search for what it does not know it has, so
  at the end of every sleep's topics stage the night makes one call to the summaries capability over the entity's
  topics (title, summary, size, newest memory, recalls of the last 7 days, largest first), its trails of thought
  (AbstractMemory's `facts.paths`: memories of one topic recalled together, written `A → B (n times)`) and what the
  pass changed, with the briefing that names the entity. The reply is stored in `<home>/self_brief.md` under a header
  (`version_seq`, `written_at`, `tokens`), never in the memory graph. Hard cap 400 tokens by the runtime's budget
  estimator (`digest.token_estimate`); a longer reply is refused, never cut. A sleep whose topics did not change keeps
  the brief without a model call. A failure keeps the previous brief and writes one authoring-failure line
  (`template_id` `runtime.self_brief`). Fenced: no call after the operator's Stop, a reply returning after it is
  abandoned, the write is a fenced write section. The night's result carries `self_brief`; `own_time.log` gets one
  line per sleep. Planned next (not in this release): each topic's line also gives
  its memory-type counts from the payload's `kinds` ("12 episodes, 3 dreams, 2 interests"), its last formed memory
  (`facts.span.last`) and its recent use (`facts.activity`, a count of recalls in the last 7 days: memory serves no
  last-accessed date), then the main trails; under the 400-token cap the per-topic line is shortened, never the list
  of topics (today an over-cap reply is refused and the previous brief kept).
- **The brief sits in every lane's system prefix** (`compose_system_base(..., self_brief=)`, read at summon by
  `self_brief_for_prompt`): `ChatSession` (hosted chat, own time and work through the life factory's re-compose) and
  the durable visit's OPEN node place the section "What I know about, at a glance (updated in my sleep):" right after
  the identity block, before everything that changes per turn, so the provider's cached prefix stays byte-identical
  across turns and changes at most once per sleep. A home without a brief gets no section. The `topics` tool's
  teaching (descriptor and tools paragraph) names the brief: "Your at-a-glance brief lists your topics; use topics to
  dig in."

### Changed

- `authoring.entity_naming_clause(entity_name)`: the naming clause of the summaries author's briefing, shared by
  `authoring_system_prompt` (unchanged output) and the brief's `self_brief_system_prompt`.

### Fixed

- **The `topics` tool description fits AbstractCore's 200-character limit with the brief's teaching added**
  (`identity/tools.py`): AbstractCore refuses a longer tool definition, so a host could not offer the entity's tools.
  A test now pins every entity tool description under the limit.

## [0.19.0] - 2026-11-10

### Added

- **An entity's cognition profile: a model per part of its thinking, how many memories a turn brings to mind, and
  how readily it revisits what it understood** (`identity/cognition_profile.py`, `docs/entity-runtime.md`). A home
  can carry `cognition.yaml` beside `substrate.yaml`. `capabilities` names the model for `summaries` (digests,
  titles, keywords, cards), `understanding` (ripples, dream narration) and `extraction`; each is `{inherit: mind}`
  by default or `{provider, model, thinking?, max_input_tokens?}` (`max_input_tokens` 4096 to 140000, 140000 when
  absent). `recall.shelf_size` (1 to 400) is the number of memories retrieved per turn in visits and own time;
  `rebuild.sensitivity` (0 to 1, default 0.5) and `rebuild.cap_per_turn` (0 to 10, default 3) set how readily
  memory work re-opens what is understood. `lookback.context` (`day` by default, or `tick`) says what the look-back at
  the end of a tick of its own time sees: the whole day so far, or this tick's exchange plus a recap of earlier
  moments; `lookback_context(home)` reads it. `resolve_capability(home, cap, mind=None)` returns
  `{provider, model, thinking, max_input_tokens, source}` with `source` `profile` or `mind`; a capability that
  inherits resolves through `resolve_home_substrate` (a host passes its resolved mind with `mind=`), and when
  nothing names a model it raises `SubstrateUnset`. `recall_shelf_size(home, default)`, `rebuild_policy(home)` and
  `read_cognition_profile(home)` (every field with its source) read it; a malformed file or field reads as its
  default with a warning naming the field. `write_cognition_profile(home, profile)` validates first
  (`CognitionProfileInvalid` carries the field and a sentence to show) and writes atomically; the gateway is its
  one caller.
- **Sleep thresholds in the cognition profile** (`identity/cognition_profile.py`). `sleep:
  {after_hours, after_events}` says when an IDLE entity goes to sleep: hours since the last completed maintenance
  pass (1..720) and new memories formed since it (at least 1); defaults 24 and 500. `read_cognition_profile` serves
  both with their sources (defaults when unset), `sleep_thresholds(home)` reads them, and a write outside the schema
  is refused with its field (`sleep.after_hours` / `sleep.after_events`) and a sentence. The gateway's idle sweep
  consumes them.
- **`build_consolidator(..., embedder=)`: the night uses the host's own embedder.** A host whose embedder is its own
  resolved AbstractCore client (the gateway's, possibly in its model-host child) - not a URL - hands it to the night,
  which passes it unchanged to `open_home` and builds nothing from `embedding_model` / `embedding_base_url`. Without it
  the night's repair refused every vectored record (`embedder_unavailable`) and the missing-embedding step embedded
  nothing.
- **An unreachable model endpoint parks the run; it no longer fails it** (`integrations/abstractcore/
  effect_handlers.py`, `integrations/abstractcore/llm_client.py`; root 1168, operator 2026-11-08). When the provider
  raises AbstractCore's `EndpointUnreachableError` (no TCP connection to the inference endpoint: the machine is off,
  asleep or unreachable), the LLM_CALL handler returns a durable EVENT wait — `wait_key = endpoint_unreachable:
  <base_url>`, `resume_to_node` = the node that issued the call, no prompt (a signal wait, presented as "Waiting
  for …", never as a question), `details {kind, message, base_url, provider, cause, detail, errno, elapsed_s,
  retryable, attempt, since, node_id}`, the reason text `the model endpoint <base_url> did not answer (<cause>:
  <detail>, after <t> s); waiting for it`. The ledger keeps one WAITING record per attempt; `_runtime.endpoint_waits`
  counts the attempts per node and is cleared when the call goes through. A `resume` with no payload re-plans the
  node and re-issues the identical call as a new effect issuance (the parked attempt is never replayed as a result);
  `cancel` ends the run as today; any other provider error fails as before. Core never counts the error in its
  breaker and the runtime records no failure. The remote client (`RemoteAbstractCoreLLMClient`) rebuilds the typed
  error from the AbstractCore server's 503 envelope (`error.code == "endpoint_unreachable"` with the seven fields,
  `detail` included; nothing is parsed out of the message); a generic 500 stays the error it
  was. The gateway's endpoint prober issues the resume; the kit's wording is its own.
  Imports `EndpointUnreachableError` by name: requires AbstractCore 2.35.0 or newer (the class, its pickle
  `__reduce__`, the 503 envelope with `detail`); the dependency floor is `abstractcore>=2.35.0`.
  `docs/architecture.md` (runtime loop) describes the wait.
- **Every tool result during entity activity rests whole as an artifact of the entity's home** (`identity/tools.py`,
  `identity/tool_effects.py`, `identity/chat.py`; root backlog 1145 step 1; operator 2026-11-08: "there should be no
  activity, either by human or by entities, that does not go through the runtime ... to enable retrospective analysis /
  search, including to rehydrate context/memory"). `execute_tool_elections(..., artifact_store=, artifact_context=)`
  stores what each executed tool returned in the home's `FileArtifactStore` through `keep_tool_result_artifact`:
  content-addressed (one blob per content, an artifact per run), no size cut, with metadata `tool`, `arguments` (the
  election body whole and its block args), `turn_id` / `tick_id`, `record_id` when known, `created_at`, `lane`,
  `run_id`, `session_id`, `entity_id`, `content_type`, `bytes`, `shown_is_view`, `failed`, and tags `kind=tool_result`,
  `tool`, `lane`, `entity_id`, `turn_id`. The result text the model reads ends with one line, `[artifact: <id>]`, so the
  verbatim and the memory records can name it; `ToolElection.artifact_id` carries it, the ENTITY_TOOLS_EXECUTE result's
  `results[].artifact_id` and the turn report's `tool_details[].artifact_id` too. Where the shown result is a bounded view,
  the artifact is the whole material: `fetch_url` keeps the whole page past its 6,000-character view, `read_file` the
  whole file (byte for byte, content type from the bytes; an image's bytes too, `tool_result_media` unchanged),
  `execute_command` the whole output past its cap. Book-adjacent tools (diary_read, diary_list, search_memory) are the
  one exception: their content is the entity's own book/graph, already in the home, and the W5 containment rule keeps
  private words out of life scope. A store failure is a loud `#FALLBACK` notice, never a lost turn. Wired on the
  ENTITY_TOOLS_EXECUTE effect lane (`tool_effects.py`, the real run's id and node) and on the home-direct chat lane
  (`chat.py`: visits driven by `ChatSession` and personal time, where the stand-in `chat-<session>` run id and the turn
  id, which is the tick id, name the origin). A lane that passes no store behaves as before (no artifact, no line).
- **Docs: [Entity topics](docs/entity-topics.md).** The page covers:
  - the night's topics stage: when it runs, the embedding first, one version per sleep, authored titles within the
    night's budget, the fence, and what each sleep records;
  - the `topics` tool, with its bodies and answers;
  - the memory facade names a host serves.

  `docs/entity-runtime.md` documents the cognition profile's `sleep` block (`after_hours`, `after_events`,
  `after_fragmented_share`) and `sleep_thresholds`. `docs/README.md`, `llms.txt` and `llms-full.txt` list the new
  page.
- **An image attached in an entity visit reaches the entity's model as an image** (`identity/visit_workflow.py`,
  `identity/tools.py`; operator 2026-11-08: images already work with every other client, so they must work for entities
  too). A visit turn's resume payload may carry `attachments`: a structured list, one record per file the visitor
  attached, with a workspace-relative `path` (and whatever the door recorded: `bytes`, `sha256`, `content_type`).
  Attachments are never parsed from the visitor's words. The runtime keeps the records whole on the turn
  (`_turn.attachments`), adds one honest line per file to the rendered message — for an image
  `(attached: shared/x.png — a PNG image, 1,204,113 bytes; attached to this message as an image; read_file re-opens it)`,
  for anything else `(attached: shared/notes.pdf — a PDF document, 88,201 bytes; read_file opens it)`, for a file that
  cannot be opened `(attached: … — could not be opened: <why>)` (a path with an embedded NUL, a name too long or a
  symlink loop reads "not a readable file path"; it never crashes the turn and never echoes a host path) — while
  `turn.text` keeps the human's raw words. The line states the fact of the call (the host attaches the file), not a
  claim about perception: whether the mind can see it is the provider's business. This
  turn's images (PNG, JPEG, GIF, WEBP by magic, within the workspace cap) are shown to the model on every call of the
  turn: on the react arm as `context.attachments` (absolute paths, cleared at HARVEST, so the next turn does not re-send
  them — a re-look is `read_file`), on the home-direct arm as the LLM_CALL payload's `media`. The appended user message
  carries `metadata.attachments` and the durable visit history entry carries `attachments`, so the transcript keeps the
  reference (path, bytes, sha256, type); the file itself stays in the workspace. A turn with files and no words is a
  turn: `turn.text` stays empty, the rendered message is the decoration alone (presence, memories, the attachment
  lines), the recall cue and the episode's verbatim name what was shown; a turn with neither words nor files still
  parks without a turn.
- **`tool_result_media(election, workspace)`** (`identity/tools.py`): the media one executed tool election shows the
  model — absolute file paths, or None when the result is text only. Declared beside the executor
  (`ToolDescriptor.result_media`); today `read_file` on an image declares its path, re-resolved and re-sniffed under
  the same containment and size cap as the read, never derived from the result text. A host that serves tool results
  as `{"success": True, "output": …}` makes `output` the dict abstractagent's sight lane reads,
  `{"rendered": <the result text>, "media": [<absolute path>]}`, and the image rides the next model call.
- **`sniff_image(data)`** (`identity/tools.py`): `(label, content_type)` for a PNG/JPEG/GIF/WEBP by magic, else None.
- **resume_to is a runtime-owned phases.yaml bucket** (`identity/life.py`).
  `resume_to: {phase: personal|work|null, written_by, written_at}` (the own-time phase an interruption took over,
  spec v23) has one writer, `write_resume_to`, read by `read_resume_to`. The writer field-merges (every other key of
  phases.yaml rides through; a corrupt file or a newer schema is refused). Until a home's bucket exists the read falls
  back to the gateway sidecar `<entities_dir>/.life_phase/<slug>.json` (`resume_to`); `migrate_phase_sidecar(home)`
  copies it once, idempotently (result `{"resume_to": "migrated"|"present"|"absent", "sidecar"}`), after which the
  sidecar is a projection only.
- **`memory_facade.records_since_maintenance`** (`integrations/abstractmemory/memory_facade.py`, ADR 0011). The
  memory facade re-exports AbstractMemory's `records_since_maintenance` unchanged (the same object), so the gateway's
  idle-sleep sweep reads an entity's records since its last completed sleep through the runtime seam instead of
  importing AbstractMemory directly. Needs AbstractMemory 0.5.0 (`sleep_cadence.records_since_maintenance`).
- **The memory facade exports the topic verbs** a host serves and runs (AbstractMemory
  0.5.0): `communities_pass` and `embed_missing` (from `abstractmemory.communities`), and `communities` and
  `community_of` (from `abstractmemory.community_index`). They are memory's own functions, unchanged, from their
  modules: memory exports none of them on its package root, where `abstractmemory.communities` is the module. The
  gateway's `/communities` route serves `memory_facade.communities(system, scopes=, owner_id=)` (payload version 2).
- **Memory's passes author their records WITH them, through the summaries capability** (`identity/authoring.py`,
  `identity/life.py`, `identity/chat.py`, `identity/entity_runtime.py`, `integrations/abstractmemory/brain_handlers.py`;
  AbstractMemory 0.5.0). `capability_formation_author(llm_for_capability, artifacts)` and
  `handler_formation_author(handler, artifacts)` build memory's `author(kind, body, context)`: one structured call
  per record, author `capability-model` / `summaries`, the client's reader, the call kept as a home artifact (its
  sources); any failure raises the client's own exception, which memory records (`author_raised`, class, message).
  Wired into every pass the runtime runs: the loop's sleep (`sleep_pass`: consolidation, world models, candidates,
  dream; the old-engine `dream_pass` rung too), a session's card updates (`world_model_update`, both sites) and the
  flow-brain `MEMORY_CONSOLIDATE` (the host's `capability_handlers["summaries"]`). `log_pass_authoring_failures`
  writes one authoring-failure line per `authoring.failures` entry a pass reports (pass, reason, class, message,
  the capability model). The runtime never calls `wake_cue_dedup_pass`.
- **A revocation of own time records who revoked it.** `write_personal_grant(..., revoked_by=)` takes
  "kill_switch" (the operator's Stop) or "operator" (his own revoke). Mode disabled then writes
  `{mode: disabled, revoked_by: ...}`, `read_personal_grant` answers it, and the next grant clears it. Naming a
  revoker on an arming write, or an unknown revoker, refuses and writes nothing.
- The runtime's own revokers name themselves: `hard_stop_loop` records `kill_switch`, and the CLI's
  `--revoke-personal` records `operator`. The gateway's revoke doors pass theirs.
- **`sleep.after_fragmented_share` in the cognition profile** (operator 2026-11-10; float, default 0.15,
  0 < share <= 1). The idle sweep also puts an entity to sleep when more than this share of the home's live memories
  are fragmented: placed in a topic only provisionally, without a vector, or with a title or summary still pending
  (memory's sleep-cadence read counts them). It is validated with a sentence and its field
  (`validate_sleep_after_fragmented_share`, `sleep.after_fragmented_share`), read with its source like the other two
  thresholds, and returned by `sleep_thresholds`. A malformed value in the file reads as its default with a warning.
  The sweep that acts on it is the gateway's.
- **The operator's start opens his first day** (`identity/life.py` `spawn_loop_process(first_day_unconditional=)`,
  CLI `--first-day-unconditional`, `LifeLoop(first_day_unconditional=)`; spec personal entered_by: "the operator's
  explicit loop start is an UNCONDITIONAL first day"). A host passes True for the operator's own acts (his grant,
  start or retry); a restore after an interruption leaves it False, so the gate governs (nothing pulling: he rests
  inside personal).
- **The ledger says when a call ran without the speculation it asked for** (`integrations/abstractcore/
  effect_handlers.py`; operator 2026-11-08: "fallback to MTP=off if MTP was selected by mistake"). When a
  provider response's `metadata.speculation.applied` is `false` (AbstractCore degraded a default-sourced
  `native_mtp` request for a provider without a native MTP adapter and generated without it; AbstractCore
  2.35.0), the LLM_CALL result carries one notice worded from that metadata — `speculation requested
  (<mode>, N draft tokens) was not applied: <message> (reason: <reason>)` — so the replay view can show why. No retry
  in the runtime; a response without the key adds nothing; an exception still propagates as before. A stored
  `native_mtp` default applied to a remote endpoint model no longer fails the call: AbstractCore 2.35.0 generates
  without it and this notice says so.
- **`EntityRuntime.readmit(fence)` / `ChatHome.readmit(fence)`: the host's Resume path for a home it keeps open**.
  An opened home now acts under one `lifecycle_fence.FenceHolder` that its handler table, memory
  commit guard, diary and artifact guards read at every check; `readmit` swaps the fence inside the home state lock
  and refuses one that is not the home's current admission (`StaleLifecycleGeneration`). The gateway's cached
  EntityRuntime per slug no longer refuses every effect forever after Stop -> Resume, and needs no reopen; readmit it
  under the per-slug lock once no run of the old generation is still acting. `ChatHome.fence` reads the holder.
- **The sleep forms the entity's topics** (operator 2026-11-10: topics "must run server side (runtime) ... during
  sleep/maintenance phase ... so that it becomes part of the knowledge of the entity"). `build_consolidator` runs one
  new stage after METADATA REPAIR and before the night voice: `identity/topics.sleep_communities`. It calls
  AbstractMemory's `abstractmemory.communities.embed_missing` first, so every vector-less record of the entity's scopes
  gets a vector, then `communities_pass` forms the night's version of the entity's communities in one guarded
  transaction. The pass gets the night's own author (the summaries capability that titles every record the night
  forms) and what the repair step left of the night's authoring budget (`authoring_cap` minus the tasks the repair
  took, in model calls). Memory owns every graph operation; the runtime holds no community code.
- The night's result carries memory's receipt under `communities` (`version_seq`, `communities`, `new`, `kept`,
  `renamed`, `dissolved`, `authored`, `pending`, `failed`, `skipped`, ...), plus the embedding counts
  (`embedding: {embedded, skipped, failed}`) and the `budget`.
  - One own_time.log line says what formed (including the unplaced records whose only neighbours are closed,
    and the detection rounds), or why memory wrote no version (unchanged, empty, nothing vectored).
  - A title the pass could not author gets its authoring-failure line.
  - Records that were not embedded get one night note, by memory's reason codes; never a failure.
  - When a version is written, `communities_version` joins the completed-maintenance detail of that night
    (memq_maintenance), with the same completion time and seq.
- The stage is fenced. Both verbs get `cancelled=fence.moved`, the author is checked before every call (no model call
  starts after the operator's Stop), and memory's commit guard refuses the version after a Stop. A stopped night
  records nothing for its topics: no version, no line, no maintenance note. The receipt says `stopped`, and the night
  ends with its one Stop line. That Stop line now also closes a night stopped during the repair step.
- A memory without the verbs produces one loud `#MISSING_VERB` line per sleep and the receipt
  `{"missing_verb": ...}`. A receipt this runtime cannot read produces one loud line naming its keys. In both cases
  the night's other work stands; the runtime never falls back to computing communities itself.
- `_sleep_authoring`'s receipt gains `considered` (the authoring tasks the drainer took).
- **The entity's `topics` tool** (tier 1, read-only, body optional; operator 2026-11-10: topics help the entity "to
  know what it knows, and let it delve deeper into subparts of its mind/memory"). It reads the records AbstractMemory
  serves (`communities`, payload version 2), never a cache of its own (`identity/topics.topics_text`, through
  `HomeMemoryReader.topics`).
  - Empty body: the list, with each topic's title, one-line summary, size and newest memory, by size.
  - A topic's `#tag`: that topic's memories by title, newest first; memories placed since the last sleep are
    marked.
  - A memory's `#tag`: the topic that memory belongs to.
  - `#tag words`: that topic's memories ranked by meaning.
  - Words: the topic closest to them by meaning, its closest memories first.

  Meaning ranks through the store's vector query (memory's VectorMatrix), one embedding per call. Without an
  embedder the tool says so and lists newest first. A topic's tag is the tail of its stable key, so it still names
  the same topic after a sleep. Every memory shown carries its `#tag`, registered for `read_memory`. An untitled topic
  shows as "(untitled yet - N memories)", never a title cut from its members.
- The tool is taught in the tools contract paragraph, declared natively, and wired on the chat lane (ChatSession)
  and the effect lane (`ENTITY_TOOLS_EXECUTE`). A lane that does not wire it says so (`#FALLBACK topics elected but
  no resolver wired`). A memory without `communities`, or a payload of another version, fails the call loudly.
- Remote transcription carries the spoken-language facts: the Core server's `language` (what the engine was told; `null` = detected) and `detected_language` ride the child result's `metadata` (and `text.metadata`), `null` when an older server omits them.
- **Who speaks in a turn: `author` on session turns, replayed history and the replay view**
  (`core/turn_author.py`, `history_bundle.py`, `session_history.py`, `replay/view.py`). One shape everywhere, every
  key present: `{kind: "entity"|"agent"|"person", name, entity_id, identity_ref, avatar: {style, color}|null}`.
  Session turn rows and every history bundle's `run` section (so session history blocs) carry `prompt_author` (who asked) and `author` (who
  answered). A host stamps them at `vars._meta.prompt_author` / `vars._meta.author` (`stamp_turn_authors`); without a
  stamp the answer is the entity of a summon run (its signed `_runtime.entity` stamp), else the agent named by
  `vars.workflow_selection.name` (then the workflow id's bundle id; never a model name), and the prompt is the
  origin's person (`_meta.origin.actor`), name null on a run older than origins. The history replayed to the next
  agent run (`session_chat_messages`) puts the author in each message's `metadata.author`, and an ENTITY's reply is
  rendered `<name> (entity): <reply>` (`speaker_line`) so an agent never reads an entity's words as its own; agent
  and person messages keep their exact bytes. Replay view messages with role `user`/`assistant` carry `author` from
  the root run (relabelled when the snapshot is folded after the records); the 15 conformance goldens are
  regenerated for the new key (cursor tokens change with the snapshot digest).
- **`repeat_of`: a turn whose prompt repeats another turn's** (`core/turn_author.py` `prompt_repeat_of`). A host
  stamps `vars._meta.repeat_of = <run id>` (the gateway does it for an `@entity` mention that answers the person's own
  turn again). Session turn rows and history bundle `run` sections carry `repeat_of` (null when none); the replay
  view's input message carries it when set; replayed history keeps the prompt (the pair stays whole, so the
  entity's reply still reaches the next agent run) and marks the user message's `metadata.repeat_of`.
- **A replayed answer carries its own time** (`session_history.py`): the assistant message's `metadata.ts` is the
  run's `updated_at` (when it answered, never before the prompt's `created_at`), so a reader sorting by `ts` never
  shows an answer above its question; the user message keeps `created_at`.
- **Work is driven by the job queue** (`identity/life.py`; operator 2026-11-09).
  `read_job_queue(home)` / `jobs_queued(home)` read the jobs queued for him, in the order he works them: the standing
  work order, then the task inbox's (`task_inbox.jsonl`, AbstractGateway's file) `taken` and `pending` tasks (parked
  and done tasks are not queued; the fold matches the gateway's). The day gate runs a work day whenever a job is
  queued, whatever the personal grant says; the decision carries `job` and `jobs_queued` (`loop_status.day_cause.kind`
  `work_order` or `task`). A task a work day opens is marked `taken`, and his ```work verdict marks it `done` or
  `parked` (`blocked:`) with the verdict line as its note (`append_task_status`, the gateway's flock-guarded
  single-line append; a plain append where there is no `fcntl`). The work day's contract names who left a task (`work_day_text`); `build_session_factory`'s
  factory takes the gate's `job`.
- **A job interrupts his own time and returns to it** (D4). When a work day opens while his own time is granted and is
  the phase the run returns to, the home remembers it (`resume_to = personal`, written by
  `loop:job-interrupts-own-time`). When every job is done the gate returns him to his own time while the grant still
  stands; otherwise own time ends idle with `stopped_by=jobs_done`. Any idle exit cancels a remembered return held in
  the runtime bucket (own time ended without an interruption).

### Changed

- **The phase graph is spec v23 and the day gate follows it** (`identity/life.py`, `identity/phase_spec.py`,
  `identity/spec/entity_phases.vendored.json`). The vendored graph is re-vendored byte for byte from abstractentity (v23: four phases,
  the statuses idle and stopped, the self-loops, no sleep->sleep). `read_day_gate` decides, in order: a visit
  posture; HIS OWN SLEEP DECISION (a ```rest block he wrote, passed as `sleep_elected`) unless the last activity
  phase was already sleep (`sleep_refusal`: no sleep->sleep) or it came from own time the operator has since
  revoked; a standing work order; personal while the operator's grant stands (a day when drives pull;
  the first day after the operator's start when the loop is started with `first_day_unconditional=True`; otherwise
  he RESTS between days inside personal, awake, zero model calls, waking early for a visit, a work order or a
  revoked grant); else `idle`: own time ends awake (`stopped_by=personal_off`, `no_return` or `edge_removed`). The
  gate never chooses sleep by itself. Every loop-written boundary consults the effective graph by its own edge
  (`personal->personal#personal_cycle`, `work->work#task_complete`, `work->personal#no_task|#operator`,
  `personal->work#operator`, `sleep->personal|work#maintenance_done`, `personal|work->sleep#self_elected`); a
  removed edge skips the leg and the floor is idle, never sleep. A self-loop writes no phase marker. 24/7 mode:
  a finished work order no longer ends the loop - the gate chooses another order (work->work), own time
  (work->personal) or idle. The day's session opens in the gate's phase (`build_session_factory`'s factory takes
  `phase`; the work contract rides only a work day).
- **No clock drives a sleep** (spec v22 retirements). Removed from the loop: the v11 timed personal
  cycle, the 6 h unattended need-check, the 1 h sleep bound over an operator sleep, the settled-desk sleep and the
  2 h grant-use floor. Their dials stay readable and are `wired: False` (`TUNABLE_RETIRED`; an edit warns that it
  governs nothing). An operator sleep lasts until the operator's wake or its maintenance's end (the gateway writes
  awake); a visit until the door's close.
- **The maintenance signals feed his decision**: when `abstractmemory.maintenance_due` says a pass is due, the day
  cue carries one line of facts (new records since the last sleep, records sharing a title, unconnected records)
  and says a ```rest block is his to write if and when he chooses. No text is read; nothing sleeps him.
- **He wakes when his sleep's maintenance is done; a visit interrupts his sleep and hands it back**
  (`identity/life.py`; spec sleep->X#maintenance_done). His own sleep no
  longer naps `rest_minutes` after the pass: he wakes as soon as the pass returns (a stop that arrived during it ends
  the loop right then). No deadline is stamped on his sleep and the night's `should_continue` reads no clock: a pass
  ends at its next stage only for a visitor, a wake, a pause or the STOP brake. A visitor who knocks during the pass
  takes the posture (the pass yields at its next stage, and the night voice, a model call, is skipped); when the
  door's close restores the interrupted sleep (`written_by=self`), the loop resumes its maintenance pass - the SAME
  sleep, counted once, no sleep->sleep - and wakes him into what the toggles allow.
- **`WorkspaceRoot.read_file` on an image returns a label instead of a refusal, and says the fact of its lane.** No
  image byte ever enters the prompt as text (the 2026-08-01 wall stands). On a lane whose host folds
  `tool_result_media` into the next model call (`execute_tool_elections(..., results_carry_media=True)` /
  `read_file(..., carry_media=True)`; the gateway's visit tool handler) the result is
  `--- shared/x.png (1,204,113 bytes) ---` then `[#IMAGE: a PNG image; it is attached to your next model call as an
  image]`. On every other lane (the in-process chat driver, the effect lane) it is `[#IMAGE: a PNG image - stored intact
  at this path; this lane cannot attach images to your model calls, so it is not viewable here. …]` — never a promised
  look that no call carries. Other binaries keep the `#BINARY` notice, which no longer claims "You have no tool that can
  view images or other binary content": on a carrying lane it says images are attached as images when read and other
  binary content is not readable here; elsewhere that binary content, images included, is not viewable in this lane.
  WEBP (`RIFF….WEBP`) is now labelled "a WEBP image" instead of "a RIFF media container".
- **There is no work toggle: work starts when a job is queued** (`identity/life.py`; operator
  2026-11-09). The day gate runs a work day whenever a work order stands, whatever the personal grant says, and never
  enters personal unless it is granted. A `work` key in phases.yaml or in the gateway sidecar (written by a
  development build) is ignored: nothing reads or migrates it, and a phases.yaml write carries it through untouched.
  `ChatSession.yield_check` stays a visitor check (True = a visitor is at the door).
- **Interest, lesson, realization and diary titles are explicit and whole** (operator 2026-11-08: "the title of any
  lesson, question, interest, world model must be explicit and easy to read, for a human and for an agent"; root
  ADR 0054). The look-back offers and the entity contract now ask for a first line `title: <a short, complete
  name>` in each ```interest, ```lesson and (optionally) ```realize block, and a `gist:` line for a question. The
  title is that line, verbatim; without one it is the body's first complete sentence (`digest.split_sentences`),
  whole. Before, these titles were the body's first 8 words, and diary titles a 60-character slug ending in "…".
  The record's digest is the words without the title line. New helper `identity.reflection.elected_title_and_words`;
  `RealizeElection.title`. Already-stored titles are not rewritten.
- **A local helper belongs to its conversation, not to the run that started it** (operator decision
  2026-11-09). A dev server an agent started in Code with `local_helper_start` died when the answer was
  delivered: each turn is one run, and the run's end closed its helpers (`browser_probe` then reported
  `ERR_CONNECTION_REFUSED`). A helper started by a run of a session is now kept by the session: it
  outlives the run, and the session's later runs reach it by its `helper_id` (`local_helper_status`,
  `local_helper_stop`). It stops when the agent or the operator stops it, when the host closes the
  session's helpers (`close_session_helpers(runtime, session_id)`, called when the session is archived
  or ends) or when the host stops (`get_local_helper_registry.close_all`); never at run completion.
  A run without a session keeps the old scope (its helpers end with it). The scope names the runtime
  that holds the run, so the same session id in another store never reaches them. The three tools'
  descriptions say the new lifetime. `register_shell_session_teardown` registers the runtime's session
  scope (`register_session_scope`).
- **A new interest's title is its title** (`identity/visit_workflow.py` APPLY, `identity/chat.py` look-back and
  mid-turn sites): the `interest: ` prefix the runtime put in front of the entity's own title is gone (operator
  2026-11-08). Interests stored before keep theirs; AbstractMemory's world-model key already strips it, so no card is
  re-keyed.
- **Default iteration budget 50, one constant** (`core/vars.py` `DEFAULT_MAX_ITERATIONS`, exported from
  `abstractruntime`): `ensure_limits`, `RuntimeConfig.max_iterations`, the visual Agent node, the agent adapter, the
  limit status/warnings and the entity visit loop read it; it was 20 (operator 2026-11-08: "By default, set it to 50
  everywhere"). Hosts and packages import the name instead of copying the number.
- **The kill switch revokes own time and cancels the remembered return; nothing is restored after it** (`identity/life.py`; operator 2026-11-09: "personal time is never given back during a resume"). `hard_stop_loop` revokes the
  personal grant (mode disabled by `kill_switch`, the gateway Stop's own write; skipped when already off; result
  `personal_revoked` / `personal_note`) and writes `resume_to = null` (`written_by=kill_switch`)
  whatever the loop's fate - a home still on the gateway sidecar is
  shadowed too - and reports it (`resume_to_cleared`, plus `resume_to_note` when a corrupt phases.yaml refused the
  write). A loop started after a kill-switch stop never returns to the own time the stopped loop was living, whatever
  `return_to` its host passes: it works the queued jobs and ends idle (`jobs_done`). Only the operator's own act
  (`first_day_unconditional`: a grant or a start) opens own time again.
- The `topics` tool reads `abstractmemory.community_index.communities` (no longer the package root). The list is that
  one read: each topic's newest date comes from memory's `facts.span`, with no scan of the entity's digests. A
  drill-down reads the digests once, for the members' titles.
- **Every memory's title and summary are authored with it** (`identity/authoring.py` new, `identity/chat.py`,
  `identity/visit_workflow.py`, `identity/life.py`, `integrations/abstractmemory/seam_handlers.py`,
  `integrations/abstractmemory/identity_support.py`, `identity/title_failures.py`; operator rulings 2026-11-09: "the
  title + summary must ALWAYS exist for any memory"; "the metadata of any newly generated memory must be generated
  along with the memory"; needs AbstractMemory 0.5.0). An elected interest, lesson or realization keeps the
  entity's `title:` line and words; what its block lacks, and every episode's and session summary's title and
  summary, the `summaries` capability model authors in the SAME step that forms the record (one structured call;
  the call is kept as a home artifact and named in `attributes.metadata_authored`). Only when that step fails does
  the record form PENDING through AbstractMemory (`MEMORY_FORM` records may carry `pending: [title|summary]`; a
  missing title is None, never a cut or a placeholder - the `untitled <kind> [<hex>]` label is retired, a
  legacy one is still never rendered), memory queues its `title.author` / `digest.author` task, and one JSON line
  (now with `missing: title|summary|both`) goes to `<workspace>/untracked/authoring-failures/<date>.jsonl` (renamed
  from title-failures; the log marker is `#AUTHORING_FAILURE`). A public diary entry without a gist forms with its
  title pending. Homes open with `task_queue=True`.
- **Pending records are retried, the sleep repairs what is left** (`drain_authoring`, `declare_missing_metadata`):
  the drainer runs only `title.author` / `digest.author`, at the own-time loop's tick boundary (inside the tick's
  writer window, after a visitor) and at a session's close, bounded by the cognition profile's `cap_per_turn`; it
  writes through `retitle` / `rewrite_digest` (which settle the task and clear the marker), defers on an unreachable
  endpoint without spending an attempt, and logs a reply without the fields. The sleep pass adds one repair step
  after its engine pass: records lacking a title or summary (or carrying the retired placeholder) are declared
  pending, then pending ones are authored; its receipt carries `authoring: {declared, refused, authored, deferred,
  failed}`.
- **Capability clients in the loop process**: `spawn_loop_process(capability_endpoint_profiles={cap: {spec,
  profile}})` hands them to the child in one env key (never logged); the child installs them in the ambient endpoint
  resolver and builds `llm_for_capability` (`build_capability_llm_factory`, the day client's patience window) for its
  sessions, its drainer and its sleep.
- **Own time needs the operator's authority, never drives alone** (`identity/life.py`). With the grant armed, `read_day_gate` admits personal only
  for the operator's start (`first_day`), a continuation of personal (`last_phase="personal"`) or a remembered return
  to it (`return_to="personal"`, or the home's `resume_to` bucket / the gateway sidecar). Otherwise own time ends
  awake: `{"phase": "idle", "cause": "no_return"}` and the loop exits with `stopped_by=no_return`. `LifeLoop(return_to=)`
  (default "personal": a direct run is the operator's act) remembers the phase its run was started to (re)enter, and a
  personal day keeps it through a sleep, a visit or a work day. `spawn_loop_process(return_to=)` / CLI `--return-to
  personal|work|none` carry it to the child; a host that passes nothing gets `none` (the home's resume_to decides).
- **Personal time says what it is for, and every look-back offers interests, lessons, questions and problems
  again** (`identity/life.py`, `identity/chat.py`, `identity/reflection.py`; operator 2026-11-08).
  - The own-time contract now reads "This time is yours to explore the tensions, the questions and the centres of
    interest that are alive in you, with the tools you have: a time to explore yourself and discover more about
    yourself and the world around you." It no longer says "there is nothing you are supposed to do with it, and no
    way to spend it wrong"; rest is "honoured when you are genuinely done" (no longer "always a valid choice").
    "Nothing owed" left the desk line and the wake cues.
  - The cue after a rest no longer repeats the rest reason ("you rested and your own time resumes."); the reason
    stays in the records (the sleep state, `own_time.log`, the tick record).
  - The desk ("Alive on your desk today") shows each offered drive's gist and its next step when the record carries
    one. The gist rule (no regex on his prose, and never the record's title, which is a machine cut of the text):
    the first non-empty line of its stored text; past 60 words (`DRIVE_GIST_MAX_WORDS`) it shows the first 60
    words, cut at a word boundary, followed by a visible " …" (read_memory opens the whole record). The offered ids still feed the first-turn commit
    exclusion, and the own-time log gains one day-open line naming the offered ids and their stored strength.
    The standing-interests note in the same cue no longer says "Yours if it pulls, never owed."
  - The model look-back runs at every visit close, every own-time day close (the zero-LLM own-time close is gone)
    and the end of every personal-time tick (not on the tick that elects rest: the day close follows). It offers
    feelings, interests, lessons, questions and problems through the same parsers as before. Two interests and two
    lessons per day of own time or per visit, counted across the turns and every look-back; a block past the cap is
    stored as refused ("[interest refused, not kept - ...: "<his words, whole>"]"), never as kept.
  - A look-back sends the same system prompt as the last turn's own call, so a provider's prefix cache can reuse
    it. What a personal tick's look-back sees is a per-entity setting, `lookback.context` in the home's
    `cognition.yaml`, read through `cognition_profile.lookback_context(home)` each time a tick's look-back is
    built, so a change applies from the next tick (and so at every day-open) without restarting the loop:
    - `day` (the default): the whole day's conversation so far. The tick's own request is then a prefix of its
      look-back, so with a prefix cache (OpenAI, llama.cpp / LM Studio, mlx-lm) each tick adds a roughly constant
      number of new tokens: the reply plus the look-back prompt.
    - `tick`: that tick's own exchange plus a recap of the day's earlier moments (the last 12). Smaller when the
      provider has no prefix cache.
    Visit and own-time day closes always see the whole discussion. Cost: one more model call per personal tick.
  - A tick's look-back stores its summary record only when it kept something (a feeling, interest, lesson, diary
    entry, realization or topic); a look-back that keeps nothing stores nothing.
  - The own-time loop reads `cognition.yaml` `recall.shelf_size` again at each day-open, so a new width takes
    effect at the next own-time day without restarting the loop.
- **The vendored phase graph is spec v24** (`identity/spec/entity_phases.vendored.json`): `maintenance_done` is
  shipped in `cause_evaluators` (the loop writes `sleep->personal|work#maintenance_done` when his own sleep's pass
  completes), so the operator's blueprint edits may now target it. Byte copy of AbstractEntity's canonical spec v24.
- **The vendored phase graph is spec v25, the five-state life cycle** (`identity/spec/entity_phases.vendored.json`,
  `identity/life.py`; byte copy of AbstractEntity's canonical spec v25): a newborn starts idle (`initial_phase: idle`, a state kept
  in `statuses`, never a phase key), the edges into work are guarded by `job_queued`, and `work->personal#operator` is
  removed with the work toggle. The loop never names that edge: the return to his own time after the jobs is
  `work->personal#no_task`, and when a job is still queued but the blueprint removed the work landing, the gate lands
  idle (`edge_removed`) instead of personal.
- **A placeholder title is marked and never shown as his** (`identity/title_failures.py`, `identity/chat.py`,
  `identity/visit_workflow.py`, `identity/prelude.py`; interim until AbstractMemory
  stores a missing title as None). A record formed with the `untitled <kind> [<key>]` placeholder carries
  `title_missing: true` and `title_failure_reason` in its attributes; the recall shelf, the turn's memory report and
  the prelude show no title for it (`is_placeholder_title`). A visit's untitled election is logged after its
  formation, so the failure line names the record it formed (or why none formed).
- **The sleep repairs legacy cards and dreams too, titles and summaries** (`identity/authoring.py`; AbstractMemory
  0.5.0 forms cards and dreams pending, and its passes refuse a home without a task queue - every runtime path that
  runs them opens the home through `open_home`, `task_queue=True`). The repair step finds records stored before that
  with memory's mechanical metadata (a card titled "World model: <subject>" without authored title provenance, a
  dream titled "Dream: ...", a card body in `mechanical-card-v2` or carrying a mechanical delta, a legacy dream's
  body) and any record lacking a title or summary, and DECLARES them pending through memory's `declare_pending`
  (AbstractMemory 0.5.0: the marker and the `title.author` / `digest.author` tasks formation gives; a declared
  title reads None until authored, the old one kept in its history). The drainer then authors their titles
  (`retitle`) and their summaries (`rewrite_digest`; a card through its own write, a dream in place). A refused
  declaration (e.g. `protected_kind`) is logged (`#REPAIR_REFUSED`) and counted; memory's `rearmed` answer (a
  missing task repaired) is a success counted with the re-arms; the receipt is
  `authoring: {declared, refused, rearmed, authored, deferred, failed}`.
- **`ChatSession.fence` replaces the host-set `cancel_check`**. The
  session takes its home's fence (`open_home(..., fence=)`) or the host's (`session.fence = <LifecycleFence>`; the
  own-time loop sets its own). **Breaking for hosts**: assigning `session.cancel_check` now raises `AttributeError`
  naming the fence, so a host that still sets it fails at once instead of running unfenced turns.
  `salvage_pending_lookback(..., fence=)` takes the door's fence (required).
- **The loop heartbeat checks the fence, not phases.yaml**. `touch_loop_heartbeat` runs before
  and after every model call of the loop; it took the home lock and parsed phases.yaml each time (~0.7 ms per call).
  It checks the loop's fence under the same lock now (one stat): no yaml parse, no subprocess; the lock (under which
  the Stop writes `stopped`) and the durable, fsynced status write stay.
- The Stop-fence tests compare a home's SQLite files by the hash of their logical dump, not by row counts (an UPDATE
  after the Stop was invisible to them), and the formation-authoring test records the write through the production
  handler table instead of re-wrapping it.
- **`open_home(..., fence=)` and `ChatSession(..., fence=)` take the fence by signature**:
  keyword-only, no default. An unfenced home or session was a silent path (the gateway's hosted chat is exactly that
  path); a reader that acts on nothing now writes `fence=None` itself, explicit and grep-able, and a ChatSession over
  a fenced home refuses `fence=None`. Every runtime caller passes its fence (the loop's day factory, the CLI visit,
  the salvage door) and the static scan covers `ChatSession` too. **Breaking for hosts**: the gateway's `open_home`
  and `ChatSession` constructions must pass the fence.
- **The Stop fence: every runtime state write carries a `LifecycleFence`, required by signature** (new
  `identity/lifecycle_fence.py`). Work is admitted at the
  home's lifecycle generation and carries the fence; `write_entity_state`, `write_personal_grant`, `write_resume_to`,
  `migrate_phase_sidecar`, `write_loop_status`, `touch_loop_heartbeat`, `record_loop_substrate` and
  `append_phase_changed` take `fence=` keyword-only with **no default** (the optional `expected_generation` is gone),
  check it inside the home state lock against the generation read there, and refuse a write decided before a Stop
  (`StaleLifecycleGeneration`, now a `StoppedByOperator`, itself a `TurnCancelled`: a clean "stopped by the operator"
  record, never a failure). Only `KILL_SWITCH` (paused, own time revoked, the return cleared, loop status stopped) and
  `RESUME` (awake, nothing restored) pass unchecked, and only for those writes. **Breaking for hosts**: every gateway
  call of these writers must pass a fence (an unfenced call raises `TypeError` and writes nothing).
- **The sidecar migration is one locked, fenced step**: the bucket's absence, the sidecar read and the write happen
  inside the home state lock, so a migration racing a Stop can no longer bring back an old personal return.
- **`spawn_loop_process(..., fence=)` takes the start's admission (required)**. The spawn
  checked eligibility, cleared a stale STOP and fast-forwarded the command inbox, and only then admitted itself - a
  Stop in between became the spawn's own generation, so the reservation, the log and the child appeared after the
  Stop. The host passes the fence it decided the start at: the spawn refuses at once when it moved, clears the
  control files only inside `fence.commit`, and reserves at that admission. **Breaking for hosts**: the gateway's
  spawn calls must pass the fence (an unfenced call raises `TypeError`).
- **A missing title is recorded, never guessed** (`identity/reflection.py`, `identity/title_failures.py`,
  `identity/chat.py`, `identity/visit_workflow.py`, `identity/diary.py`,
  `integrations/abstractmemory/identity_support.py`; operator ruling 2026-11-09: "we shouldn't need a fallback. if
  it fails, it should add information to a temporary error file in untracked/"). An elected interest, lesson or
  realization takes its title only from the block's `title:` first line, and a public diary entry only from its
  `gist:` line; his first line (or first sentence) no longer stands in. `parse_elected_title(block)` returns
  `(title | None, words, reason)` with reason `missing`, `empty`, `multi_line` (two `title:` lines), `malformed` (the
  `title:` line is not first) or `too_long` (over `TITLE_MAX_CHARS` = 200); `elected_title_and_words` returns
  `(title | None, words)`. The words stay whole. Each failure appends one JSON line (observed_at, entity, record_id,
  kind, raw_block, template_id, template_version, provider, model, reason, stored_title, plus session_id / turn_id /
  entry_id) to `<workspace>/untracked/title-failures/<YYYY-MM-DD>.jsonl`, the workspace being the parent of the data
  dir that holds the home; no environment variable is read. When no such folder resolves the same line goes to the
  home's `own_time.log` after `#TITLE_FAILURE` (and to the session's output). AbstractMemory requires a non-empty
  title, so until it accepts a missing one the record carries the visibly-untitled label `untitled <kind> [<key>]`
  (the key: 6 hex of the block's sha256, none of his words). Block bodies now keep his line breaks (the checks read
  lines). `build_diary_effect_handlers(home_dir=...)` locates the failure file; a host that does not pass it gets the
  marked line in the projection's warnings.
- **A new lesson's or realization's title is its title** (`identity/chat.py` mid-turn, look-back and realization
  sites; `identity/visit_workflow.py` APPLY): the `lesson: ` and `realization: ` prefixes are gone, as `interest: `
  already was. Records stored before keep theirs; the Entity app's labels already strip them.
- **What a turn formed is served beside the reply as structured notices; the served reply carries no bracket
  markers** (`identity/chat.py`, `identity/reflection.py`, `identity/visit_workflow.py`, `history_bundle.py`,
  `replay/view.py`; contract agreed with the Entity app owner 2026-11-09). One shape everywhere:
  `[{kind: "diary"|"question"|"lesson"|"interest"|"realization", record_id: <the GRAPH id of the formed record; a
  diary entry's projection, never only its diary_ id>, title: <as formed, "" while it is still being written>,
  text?: <his words, whole>}]`, in the order the acts stand in his reply; a private diary entry is `title ""`,
  `visibility: "private"`, no words; a visit-lane lesson/interest/realization staged for the close is
  `record_id: null, staged: true`. Served on: `TurnReport.reply_notices` (chat lane; `ChatSession.turn` returns
  the served reply), the visit turn's `_turn.reply_notices` / `_turn.reply` and its ANSWER_USER payload
  (`message` = the served reply, `notices` = the list: the ledger is the source every app reads), the replay
  view's answer message (`notices`), and every session turn row (`turn.notices`, read from the run output's
  `notices`; `[]` when none). The served reply is the marked reply without exactly the markers those notices
  replace (`[kept in diary - ... - reread: diary_read diary_...]`, `[kept a private diary entry ...]`,
  `[learned: "..."]`, `[kept interest: "..."]`, `[realized: "..." - held for sleep]`; removed once, a marker
  alone on its line takes its line); a refusal marker stays (nothing formed to point at). The STORED reply keeps
  every marker (root ADR 0054): `TurnReport.marked_reply` / `_turn.marked_reply`, the episode verbatim and the
  session history he re-reads. The `notices` strings stay the operator's log lines. Reflection gains the marker
  builders `interest_marker`, `lesson_marker`, `realization_marker` (the parsers use them) and `without_markers`.
- **A visit's episode is formed with its authored title and summary** (`identity/visit_workflow.py`,
  `identity/entity_runtime.py`, `identity/authoring.py`; operator 2026-11-09: metadata generated with the memory).
  The durable visit workflow cannot call a model inside a node, so its formation step issues the authoring call as an
  LLM_CALL effect tagged `payload.capability = "summaries"` (`authoring_llm_call_payload`: system prompt, one user
  message with the exchange, `response_schema` {title, summary}, `params.thinking = "off"`, `_absorb_failure`)
  in a new `VISIT_AUTHOR` node between COMMIT and FORM; FORM forms the episode with the authored title and summary
  (`attributes.metadata_authored`), and only a failed call forms it pending (memory's task) with a failure line
  written once the record has its id. `open_entity_runtime(capability_handlers={cap: handler})` routes a tagged
  LLM_CALL to its capability handler and stamps the echo `result.capability`; a capability nobody serves is answered
  `{ok: False, capability_unrouted: <cap>}` with an error in the log (reason `capability_unrouted`), never by the
  mind. The fields are read from `result.data` (structured output) or strictly from `result.content` as JSON.
- **Every record a visit's close forms carries its authored metadata** (`identity/visit_workflow.py`; operator
  2026-11-09). The look-back summary (title and summary) and each interest, lesson or realization elected without a
  `title:` line (title; his words stay the summary) get one LLM_CALL tagged `capability: "summaries"` in the APPLY
  step right before the record forms (`_reflect.authored.<summary|interest-i|lesson-i|realize-i>`), and form with
  the authored metadata (`attributes.metadata_authored`; a floored summary loses its mechanical-floor label once
  authored). Only a failed or unrouted call forms the record pending (memory's task), with its failure line written
  at the next APPLY once the record has its id. A visit never fails on it.
- **The summaries author's briefing names the entity and sets the title style** (`identity/authoring.py`:
  `authoring_system_prompt`). Every authored record (an episode, a summary, a diary entry, a dream, a card, a topic, a
  birth record) is now briefed with "<name> is the entity whose memories these are: refer to <name> by name, never as
  'the agent', 'the assistant' or 'the speaker', and write in the third person" and "Write the title in sentence
  case, with no trailing period and no quotes."; `capability_formation_author`, `handler_formation_author`,
  `author_metadata` and `authoring_llm_call_payload` take a required keyword `entity_name` (`ChatHome.name`; an empty
  one raises), and `drain_authoring` reads it from the home's `spark.yaml` (`entity_name_of`).

### Removed

- **The runtime's own communities compute is retired** (`identity/communities.py`; operator 2026-11-10: memory owns
  the graph operations). The request-time Louvain, the keyword bridges, the γ sweep and the keyword LABELS (a regex
  heuristic naming topics with no model) are gone. Topics are AbstractMemory's community records, formed during the
  sleep with authored titles (the night's topics stage) and served as payload version 2. The keyword extractor in
  `identity/digest.py` stays this wave: world-model cards and the candidate miner read its keywords.
- `memory_communities` remains only to fail loudly. It raises `CommunitiesRetired`, an `ImportError` the gateway route
  already answers with "501 communities unavailable", with a sentence naming what to serve instead. It never returns
  the old shape. The module is deleted once its last caller, the gateway's `GET /entities/{name}/communities`, has
  switched. A test pins that no community algorithm or keyword label remains in the runtime's source.

### Fixed

- **A visual Agent node's agent can stay out of a message addressed to entities** (with AbstractAgent
  0.6.1 and the AbstractGateway release that writes `_runtime.mentions`). The gateway writes `_runtime.mentions`
  (`{entities: [{entity_id, name}], line}`) on the run when the person's message mentions entities,
  but an Agent node runs its loop in a child run whose `_runtime` is built fresh, so the child never
  saw it and asked again after an empty reply instead of ending the turn silently. The Agent node now
  carries `mentions` into its child (a copy; an explicit child value wins). The gateway's context line
  already reaches the child's messages the way the conversation does (the node's `context` pin, or
  `use_context`).
- **An authored summary is no longer stamped mechanical** (`identity/chat.py`, `identity/visit_workflow.py`). The
  visit episode formed with `digest_method: mechanical-v2` before its authored title and summary were merged, and
  the chat lane's `_author_records` kept the formation's mechanical label after the summaries capability wrote the
  summary; memory therefore read a model-authored summary as mechanical and queued a `digest.rewrite` for the sleep
  (tokens spent rewriting authored words). When the summary is authored the record now says
  `digest_method: authored:summaries` (`chat.AUTHORED_SUMMARIES_METHOD`, one helper `authored_digest_attributes`
  for both lanes, the visit close summary included); a failed authoring keeps `mechanical-v2` (its provisional
  digest is the mechanical one) and a title-only authoring leaves the digest's label alone.
- **An authoring failure line says what failed and on which model** (`identity/authoring.py`, `identity/chat.py`,
  `identity/visit_workflow.py`). When the summaries capability client RAISES, the line's reason is `client_error`
  (was `malformed`) and it carries `error_class` / `error_message` (the client's own exception, not the wrapper),
  `provider` / `model` of the capability client used (the drainer wrote None) and `capability: summaries`.
  `malformed` now means only a reply that came back and did not fit the title and summary (missing fields, a pydantic
  `ValidationError`); `memory_refused` is a refused authored write (`AuthoringRefused`). The visit lane names the
  capability call's own provider/model when its result carries them, never the visit's mind in their place.
  `authoring_failure_fields(exc, llm)` is the one classifier.
- **A pending record whose authoring tasks all ran out gets a fresh one** (`identity/authoring.py`,
  `identity/life.py`; operator 2026-11-09: a title and a summary must always exist). A capability failure that
  exhausted a `title.author` / `digest.author` task's attempts left the record pending with no live task, forever.
  `rearm_exhausted_authoring(ms, scopes=)` finds those records with memory's `pending_records(live_task=False)` and
  re-queues them with memory's `rearm_pending` (the marker stays, its `rearmed` count grows): at the start of every
  drainer pass (every scope) and in the sleep's repair step (the home's scopes; receipt `authoring.rearmed`). A
  refusal is logged `#REARM_REFUSED`; a memory without the verbs is said once per process (`#MISSING_VERB`), and
  nothing is written around memory.
- **Memory's birth labels are placeholders, repaired by the sleep**.
  `is_placeholder_title` also matches memory's exact birth labels for an unnamed spark
  item - `trait-N`, `value-N`, `purpose-N`, `limit-N` (the stem, one hyphen, digits; never words that merely contain
  them) - so renderers show no title for them and the sleep's repair declares those legacy records pending; the
  drainer then authors their titles like any record's.
- **A public diary entry's summary is authored, never his first line standing in** (`integrations/abstractmemory/
  identity_support.py`, `identity/diary.py`, `identity/chat.py`, `identity/act_only.py`, `identity/entity_runtime.py`,
  `identity/visit_workflow.py`, `identity/authoring.py`; AbstractMemory 0.5.0, which takes an
  authored summary on a public book projection). `diary_record_metadata` is the one rule: title = his elected gist,
  whole (a mechanical note's gist is never a title), else authored; summary = authored, formed with its reading
  (`attributes.understanding`: capability-model author, reader, the authoring call as its source; `digest_method`
  `authored-v1`). ChatSession's DIARY_WRITE and the visit lane's act-only capture author the summary (and the title
  without a gist) BEFORE the write; a failed or absent authoring forms the missing fields pending (his gist / first
  line only their provisional words) with one failure line, and the warning riding the run is word-free. The session
  sheet, the resolution notes and the visit capture metadata name the entry by its record's title (or say it is
  still being written), never by his first line. The sleep's repair declares a legacy public diary record's
  template title ("Diary entry (<kind>) — ...") pending and asks memory to declare its stand-in summary pending
  (AbstractMemory 0.5.0 declares it for a public book projection and the drainer authors it; an entry memory refuses
  with `protected_kind`, such as one claiming a chain position, keeps its stored summary, and the refusals are said
  once per pass). Private entries are unchanged
  (root backlog 1186).
- **A public diary entry's record carries no template: his gist, else an authored title** (`integrations/abstractmemory/
  identity_support.py`, `identity/diary.py`, `identity/chat.py`, `identity/act_only.py`, `identity/entity_runtime.py`). The projection no
  longer builds "Diary entry (<kind>) - <date>[: <gist>]" titles, entry-id tails or the "Wrote a diary entry ...; no gist
  elected" body. His elected gist is the title, whole; a gistless public entry (and a writer-declared mechanical note,
  whose gist is machine words) gets its title from one summaries capability call BEFORE the write - ChatSession's
  DIARY_WRITE (`authored_metadata` on the payload, the call kept as a home artifact) and the visit lane's act-only
  capture (the host's summaries capability handler). The record's summary is always his words (his gist, else his
  first line, whole): a diary entry is his act and memory never authors its body. Only a failed (or absent) authoring
  forms the title pending, with one authoring-failure line naming the record, the reason, the error class and the
  capability's model. Private entries are unchanged (content-free act frame) pending the operator's ruling.
- **Every failed authoring attempt of the drainer leaves its failure line** (`identity/authoring.py`). Only a malformed reply was recorded; an unreadable task packet (`packet_unreadable`), a record with no words
  to author from (`no_source`) and an unreachable capability (`unavailable`, deferred without spending an attempt) now
  write one line too, each with the record id, the `outcome` (retry / defer), the original exception's class and
  message and the capability model when one was built.
- **A failed visit turn keeps what it already did** (`identity/visit_workflow.py` `VISIT_TURN_FAILED`). The failed turn's messages were deleted from the adapter transcript, erasing completed tool results
  and operator guidance, and the error answer reported `tools_ran: []`. Now the transcript keeps them and an explicit
  failed-turn outcome closes the turn (`[this turn could not be answered: <error>]`, metadata `kind: turn_failed`),
  the visit history folds the turn with that outcome, and the error answer reports the tools that really ran.
- **The hard stop signals only a live own-time loop, never its caller, and says what it did**
  (`identity/life.py` `hard_stop_loop`, `write_loop_status`; gateway slice finding 2026-11-09). After a freeze the
  status file was written by the CALLING process, so it named the caller's pid and start time; a second hard stop
  then read the caller as the loop and SIGTERMed it (it killed the gateway's test process). Now the status names
  the loop it is about (`write_loop_status(..., owner=(pid, token))`, or no pid), `stopped_by=kill_switch`, and
  before any signal the pid must not be the caller's own, must be alive, must match its start-time token, and its
  command line must run the loop module (`-m abstractruntime.identity.life`). A process that has exited (reaped, or
  a zombie its parent has not reaped) counts as gone. The result is explicit: `{killed, pid, signal:
  "SIGTERM"|"SIGKILL"|None, reason, requested_reason}` (`killed` only once the process is confirmed gone), plus the
  legacy keys (`frozen`, `was_running`, `escalated_to_sigkill`, `requested_by`, `status`).
- **The hard stop signals only this home's verified loop and says killed only on a confirmed exit**
  (`identity/life.py` `hard_stop_loop`). Before any signal the incarnation is
  verified - alive (not unknown), not the caller, the recorded start-time token equal to the OS's - and the process
  must be THIS home's loop invocation: `<python> -m abstractruntime.identity.life --home <this home>`, the module
  right after the interpreter (the module name as a later argument, another home's loop, a missing token, an
  unreadable start time or command line all refuse: fail closed). Before escalating to SIGKILL the incarnation is
  verified again. `killed` is true only on a confirmed exit (reaped, ESRCH, or a zombie); a denied permission
  (EPERM) or an unreadable status is "unknown", never a death, and gets no SIGKILL.
- **A sleep handed back by a visit always resumes; the loop never spins over a held state** (`identity/life.py`). With nothing pulling, the gate chose the personal rest; the rest returned at once
  because he was asleep (a visit's close had handed his own sleep back) and the loop repeated forever without
  resuming the pass. A held state (asleep, the visiting posture, paused) now goes straight to the state gate, before
  any leg is chosen: his own sleep resumes its maintenance and wakes, an operator sleep, a visit or a pause idles.
- **One lock and one generation for a home's phase state: a late writer can no longer undo Stop**
  (`identity/life.py`). `home_state_lock(home, timeout_s=30.0)` is the one
  cross-process lock (fcntl.flock on `<home>/.phase_state.lock`, re-entrant per thread, `timeout_s=None` waits
  without a limit, `HomeStateLockTimeout` otherwise); the gateway takes it INSIDE its own per-home lock and no runtime
  path calls out while holding it. Every phases.yaml write (`write_personal_grant`, `write_resume_to`, the
  migration) re-reads and merges against the current file inside it, and so does `write_entity_state`. The lifecycle
  generation (`lifecycle_generation` in phases.yaml; `read_lifecycle_generation`, `bump_lifecycle_generation`
  returning the new value) fences them: those writers take `expected_generation=` and raise
  `StaleLifecycleGeneration` (writing nothing) when it moved. `hard_stop_loop` revokes personal, clears resume_to and
  bumps the generation in ONE critical section before any signal (result `lifecycle_generation`). The own-time loop
  fences its state and resume_to writes on the generation it started at and ends (`stopped_by=kill_switch`) when it
  moves.
- **The requested iteration budget is the one the agent obeys** (`visualflow_compiler/visual/executor.py`,
  `core/runtime.py`). A flow whose On Flow Start node has a `max_iterations` pin with an authored default (the
  shipped basic-agent: 20) used to send that default through the data edge into the Agent node, which overwrote the
  run's `_limits.max_iterations` in the react child: a run that asked for 50 through `_limits` (the gateway default,
  the kit field) ran 20. The pin default no longer stands in for the run's budget; an explicit flat `max_iterations`
  input still counts as the request. START_SUBWORKFLOW now carries the parent's `_limits.max_iterations` into a child
  flow (an explicit child value wins), so one value governs the whole run tree.
- **One iteration-limit order everywhere: the person's request > the host's stored default > the workflow's own value >
  50** (`core/vars.py`, `core/runtime.py`, `visualflow_compiler/compiler.py`, `visualflow_compiler/visual/executor.py`). An Agent node's own `max_iterations` pin default (the Flow editor seeds one on every
  Agent node) beat a requested `_limits.max_iterations` (request 100 ran 20), and a deliberate On Flow Start pin value
  never applied, because `Runtime.start` always seeds `_limits.max_iterations` (pin 5 ran 50). `Runtime.start` now
  stamps `_limits.max_iterations_declared` (true when the caller named `_limits.max_iterations`, i.e. a request or a
  host default; `declared_max_iterations(limits)` reads it): a declared budget wins at the start node, at every Agent
  node (over its edge, pin default and config) and in child flows (over a subflow pin's flat value); otherwise the
  workflow's own value (start-pin default, Agent-node pin default or config) wins over the seeded
  `DEFAULT_MAX_ITERATIONS`. An explicit child `_limits.max_iterations` (a delegated sub-agent's narrower budget) is
  kept. Note: a shipped workflow that authors its own value (basic-agent's start pin 20) now runs that value when
  nothing is requested and no host default is stored.
- **The sleep's repair authors a legacy public diary entry's summary, not only its title** (`identity/authoring.py`;
  AbstractMemory 0.5.0, which declares a legacy public book projection's summary pending under formation's rule,
  `diary_summary_refusal`). A public diary record formed before authored summaries carried
  his first line as its summary: the repair's declaration is now taken (pending marker + `digest.author` task, the
  first line kept as provisional version 0), and the drainer authors the summary in the same step, beside the
  template title. Private projections and owner-direct entries are never declared. A summary memory still refuses
  (a projection claiming a chain position) is said once per night per rule memory names, with that rule
  (`protected_kind - ... (hash_chained)`), instead of the stale "until memory accepts the declaration".
- **No length cap anywhere on stored text** (root ADR 0026/0054; operator 2026-11-08: stored text is never cut or
  capped, what is shown is each app's fold). The remaining bounds are gone:
  - the automation `title` (`automations/models.py` `validate_title`, `automations/wakes.py`): any non-empty text,
    whitespace trimmed; the 120-character refusal is what made the Assistant web cut titles before saving
    (root/assistant 1154); a note wake's derived "Reminder: <note>" title is whole. `TITLE_MAX` is gone;
    `docs/automations.md` says so;
  - a published event's `title`, `body` and `subject` (`core/publish.py`): non-empty / text, no 120 / 2,000 bound;
  - the mechanical record title (`identity/digest.py`): the chosen sentence whole (it kept 14 words and a "…"), and
    the words whole when no sentence qualified (was 8 words);
  - the `[felt: …]` and `[realized: …]` markers in the stored reply (`identity/reflection.py`) carry the record's
    words, his reason and the realization whole (they were marked cuts at 60 / 80); `_receipt_gist` is gone, and the
    two visit-workflow refusal notices that used it quote the realization whole;
  - a failed occurrence's attention body is the whole error (`automations/controller.py`, was 2,000);
  - `#FALLBACK` notices (`identity/reflection.py`) quote the skipped line whole — notices rest in run vars and in the
    close report, so they are stored (were 80 / 60);
  - `identity/life.py`: the loop status `day_cause.detail` (was 200), the entity's `next:` cue (was 400) and the rest
    reason written to the sleep state (was 400) are whole; the standing-interest offer in the day cue folds visibly
    with `drive_gist` instead of a silent 110-character cut;
  - `identity/chat.py`: the turn report's tool `arg` (was 120), the resolution gist handed to the reflection prompt
    (was 100) and the sheet line's first line (was 120) are whole.
  Left as log-only (console `out` echoes): `chat.py` lesson/interest/realization echoes at 70, `life.py` tick
  `reply_head` at 160 and work verdict at 120. Left as prompt currency under a token budget: the recall handles
  (`chat.py` ~1960, title 120 / digest 280) and the recall path labels. Still bounded and MARKED, not changed here:
  the reflection sheet line (`chat.py` `_SHEET_LINE_MAX = 160`, `#TRUNCATION`) and the world-model card text
  (`chat.py` `_CARD_TEXT_MAX = 1600`, `#TRUNCATION`), both stored; a ruling on them is pending.
- **A call refused before it ran saves no prompt artifacts**.
  After a Stop landed mid-turn, the refused model call's node trace still carried its whole payload, and the trace's
  bounding offloaded ~48 KB of the prompt (messages, system prompt, tools) into artifact files after the Stop. The
  fence's refusal at dispatch now marks its cancelled outcome `ran: False`, and the Runtime's node trace records such
  an effect without its payload (type, result key and the cancellation kept; nothing offloaded). The ledger's own
  records of the cancelled act are unchanged.
- **Turning personal off ends own time awake, never asleep** (`identity/life.py`). In 24/7 mode the day gate's `no_grant` leg wrote `asleep`
  (`written_by=grant-gate`) with a 6 h wake, and the visit door refused him. Now the loop ends own time in both modes
  with `stopped_by=personal_off` (was `personal_disarmed`) and writes no sleep: he stays awake and visitable. An
  ending loop no longer lands him in a `loop-exit` sleep; it wakes a sleep it wrote
  for itself, and a new loop wakes one an earlier loop left (`LOOP_OWN_SLEEP_WRITERS`: `self`, `personal-cycle`,
  `grant-gate`, `day-gate`, `loop-exit`). An operator sleep, a visit posture and the kill switch are never touched.
  A supervised settled desk ends the run without writing a sleep.
- **Reflection summaries store their digest whole** (`identity/chat.py`, ADR 0054). The look-back summary's digest
  was cut at 280 characters with a marker; it is now stored as written, for visits, own-time days and ticks.
- **A diary entry with no elected gist stores its whole first line as its digest** (`integrations/abstractmemory/
  identity_support.py`, ADR 0054). It was cut at 120 characters mid-word; the look-back is now the main producer of
  questions and problems.
- **Queued work starts with own time off** (`identity/life.py`). The spawn door
  (`spawn_loop_process`) and the CLI start door refused every start without the personal grant, so a job queued
  while own time was off never ran. `loop_start_refusal(home)` admits a start when a job is queued or the grant is
  armed, and refuses otherwise (`no personal time: <why> - and no job is queued`); the ownership checks (one start
  at a time, a running loop) are unchanged.
- **A replayed answer is the stored bytes, whitespace included** (`session_history.py`, `history_bundle.py`). Session turns stripped the answer before replay, so an entity re-reading its own session
  got "indented answer" for "  indented answer\n\n" (not what it said; prefix caches missed). The answer now replays
  as stored; `.strip` only decides whether there is one.
- **The session sheet and the authored world-model card are stored whole** (`identity/chat.py`,
  `identity/visit_workflow.py`; root ADR 0054, memory owner 2026-11-08: abstractmemory has no length caps on its
  columns and `author_world_model` stores text whole). The 160-character `_sheet_line` cut on each sheet moment
  (run vars, `pending_reflection.json`) and the 1,600-character `_bounded_card_text` cut on an authored briefing are
  gone; the sheet line is the record's digest byte for byte and the card is what the entity wrote. A PROMPT that
  lists the sheet (the look-back, the card-authoring prompt) folds each moment visibly at read time
  (`sheet_prompt_line`, `SHEET_PROMPT_LINE_MAX = 160`, "… [folded for this prompt; the record is whole]"): the
  prompt sees the whole conversation beside the sheet, so the sheet is an index of the moments and an unfolded one
  would add ~5x tokens per moment and no new words. The feel-marker renderer keeps reading the whole lines, so what
  rests in the reply is the record's words, never the fold.
- **An activated skill is always readable** (`integrations/abstractcore/skill_roots.py`, `effect_handlers.py`;
  operator 2026-11-08: an agent in Code, with a skill activated for its session, called `read_file` on the skill's
  `SKILL.md` and got `Error: File '…' is ignored by .abstractignore policy`, because AbstractCore's file tool applies
  the workspace `.abstractignore`, which prunes the gateway data folder where the shelf lives). A `read_file` whose
  path lies inside one of the run's own skill folders is now answered by the runtime itself — the whole file, UTF-8,
  in the same line-numbered shape AbstractCore renders, `start_line`/`end_line` honoured — before AbstractCore's tool
  runs, so neither the workspace `.abstractignore` nor the workspace scope can refuse a skill the model was told
  about. The folders come from the run's own record, never from the model's words: `skills` as the workforce payload
  (`shelf`, `imported_shelf`, `roots`, `ABSTRACTCODE_SKILLS_ROOTS`, with the `active` names), `_runtime.skills_resolution`
  (`shelf`/`imported_shelf`/`roots` when the host records them, with its `active` names), and the `<path>` each skill
  carries in `_runtime.skills_block` (AbstractSkill 0.3.11 renders the SKILL.md it loaded); a folder counts as
  `<root>/<name>` or `<root>/skills/<name>` only when it holds a `SKILL.md`. The requested path must lie under a
  folder lexically and, once resolved with `realpath`, still under that folder's real path: a symlink leading out is
  refused with a typed error, never served and never handed to AbstractCore. Any other path reaches AbstractCore
  exactly as before.
- **The sleep's repair now reaches episodes and session summaries formed before formation authored them**
  (`identity/authoring.py`; operator 2026-11-09: every memory has an authored title and summary, the sleep repairs
  leftovers). `declare_missing_metadata` selects them BY STAMP, never by their text: a record of kind `episode` or
  `summary` that carries no authored provenance (no `metadata_authored`, no authored `title_provenance`, its current
  reading not written by a capability model, the operator or the entity) and whose reading memory derives
  (`reading_provenance`) as `mechanical` (a `mechanical-v1` / `-v2` / `-floor-v1` / `-flow-v1` / `-dedup-v1` /
  `-close-v1` digest_method or stamp) or `unrecorded` (no label at all: the Entity app's "who wrote it was not
  recorded") is declared pending for BOTH its title and its summary through memory's `declare_pending`. The stored
  title and text are kept as version 0; nothing is cut. A summary titled with the runtime's own pre-4eefddb template
  `session reflection: <session id>` (matched by that prefix only) has its title declared even when its summary was
  already authored. A ripple's consolidation record is never taken.
- **A closed record is no longer scanned by the repair**: memory refuses `closed` (its replacement is the live
  record), so a superseded episode used to cost one `#REPAIR_REFUSED` line every night. The receipt counts them in
  `closed`.

The drainer authors the declared fields with the summaries capability model and stamps the capability model's
provenance on each (`retitle` -> `title_provenance`, `rewrite_digest` -> `understanding`, `digest_method
authored-v1`: author `{kind: capability-model, id: summaries}`, the client's reader, the call kept as a home
artifact). BUDGET: one task per field, at most 64 per sleep pass (`build_consolidator(authoring_cap=64)`, about 32
records with both fields), plus `rebuild.cap_per_turn` (default 3) at each own-time tick boundary and at a session's
close. A home with thousands of such records is repaired over many nights, its declared titles reading None until
authored.
- **An entity re-reading its own session reads its own replies bare** (`core/turn_author.py`,
  `session_history.py`). Replayed history labelled every entity reply
  "<name> (entity): ...", including the replies of the entity that was reading it (a re-summoned entity
  replays its own session): a model may copy the prefix, and the bytes no longer matched what it said,
  so every prefix cache of that session missed. `session_chat_messages(..., reader_entity_id=...)` and
  `speaker_line(author, text, reader_entity_id=...)` now leave the reader's own replies exactly as
  stored; every other reader still sees who spoke. `entity:<name>`, `entity:<name>@<home_id>` and a bare
  name are the same reader. Hosts that seed an entity run's history pass the run's entity id
  (AbstractGateway: the summon seeding in `bundle_host`).
- **No authoring model call starts after a Stop**. Every authoring path resolved its
  capability client and called it with no check in between, so a Stop landing during the resolution (or anything
  between the outer check and the call) still sent the model a call. `ChatSession._author_records` /
  `_author_diary_entry`, the authoring drainer, and memory's formation authors (`capability_formation_author`,
  `handler_formation_author`, which now take a REQUIRED `should_stop` - the admission's fence) check immediately
  before each model call and after it returns, and let the operator's Stop through instead of turning it into a
  deferral or an authoring failure.
- **The loop's bookkeeping and the chat lane's pending look-back marker never land after a racing Stop**. The loop's spend, personal-time usage, task status (taken / done / parked) and work-order archive, and
  `ChatSession`'s write-ahead marker (`pending_reflection.json`), checked the fence and then wrote outside any lock:
  a Stop in that window landed a file after the paused row. Each is one `fence.commit` step now (the home state lock
  and the generation re-read there): a Stop arriving meanwhile waits for the write, and a write decided before a
  Stop never lands after it (about 0.4 ms once per tick or day close, never per effect).
- **No model call after a Stop, retries included, and a late reply lands nowhere**: `_generate` checks the fence before EVERY attempt (a harmony-400 retry after the Stop used to call the model
  again), maps a call that failed after the Stop to the Stop, and abandons a reply that returned after it (not
  counted in spend, nothing formed). A client that honours `cancel_event=` (abstractcore's providers) gets the
  fence's event: the Stop in the bumping process stops the decode within a token.
- **Every chat effect is fenced around its authoring**: `_effect` checks before the authoring call
  and again before the handler, and `_author_records` / `_author_diary_entry` abandon an authoring reply that
  returned after the Stop (no artifact kept, no write).
- **The CLI visit writes nothing after a Stop**: it is admitted at the home's generation; its
  auto-yield posture, its timeout restore and its close restore are fenced, and the restores check the posture and
  write in one locked step (the timeout restore used to write awake unconditionally); a Stop mid-visit ends it with
  one plain line, without a look-back.
- **The visit lane's durable diary capture after a Stop**: a fenced book refusal is the Stop - no
  rescue copy of the reply is written into the home - and the public entry's authoring call to the host's summaries
  handler never runs after it.
- **A parent run ends CANCELLED when its child was stopped**. The fenced effect table checked
  only a model call's outcome after it returned, and the Runtime's own built-in handlers (`START_SUBWORKFLOW`,
  `ASK_USER`, `ANSWER_USER`, the waits, `MEMORY_QUERY` ...) were not in the fenced table at all: a child the Stop
  cancelled came back as `completed({cancelled: True})` and the parent went on to end COMPLETED. Every effect's
  outcome is now checked after it returns, and `open_entity_runtime` fences the Runtime's whole table, built-ins
  included (two fence checks per effect).
- **A process that bumped a home once still sees a Stop by another process**. The bumping
  process used to skip the signal stat for that home forever after its first Stop, so a later Stop by another process
  (a CLI `hard_stop_loop`, a service twin, a second gateway) passed every fence check until the next truth read. Every
  process now pays the one stat per check (about 1-3 us); the bump still updates its own process's table at once.
- **The fence fails closed on every signal failure**. A home that was never stopped had no
  signal file, and "absent then absent" read as unchanged in every process, so another process's Stop whose first
  signal write failed was hidden; a failed removal of a stale signal hid it too; a corrupted phases.yaml left
  `moved` False; and a failed truth read kept the cached generation standing. Now: admission (`LifecycleFence.admit`,
  inside the home state lock) establishes a valid signal - written when missing or behind, refusing admission when it
  is ahead of phases.yaml; a missing signal reads as moved; the check also observes phases.yaml's own signature (two
  stats; the file is parsed only when it changed), so a bump whose signal failed or a corruption is seen at the next
  check; and a failed truth read drops the cached generation.
- **A home opened with its lifecycle fence refuses every write after the operator's Stop** (AbstractMemory 0.5.0). `open_home(..., fence=)` passes `fence.check` as memory's
  `commit_guard` (called inside every write transaction, before COMMIT: a Stop rolls the write back), and guards the
  diary book (`DiaryStore(guard=)`, before an entry is appended) and the artifact store (`FileArtifactStore(guard=)`,
  before a store, metadata update, access record or delete). Its effect handler table is wrapped once
  (`lifecycle_fence.fence_handlers`): an effect after the Stop is refused without running, a model reply that returned
  after it is abandoned, and a handler whose commit the guard refused reports the Stop - always
  `EffectOutcome.cancelled` with `KILL_SWITCH_REASON`, never failed and never retried. The loop's day sessions and its
  night open their home with the loop's fence.
- **`open_entity_runtime(..., fence=)` is required**: the visit lane's Runtime runs over a fenced home and its FINAL
  handler table (the host's LLM and tool handlers, the capability routing) is wrapped, so a Stop ends the run
  CANCELLED with the operator's reason. The run's own ledger keeps an unguarded artifact store (the records of a Stop
  are written after it). **Breaking for hosts**: the gateway's visit and mail lanes must pass the run's fence.
- **A Stop inside formation authoring is never an authoring failure**: `ChatSession._author_records` /
  `_author_diary_entry`, `authoring.author_metadata` and `diary_authoring_payload` let the operator's Stop through
  (it used to become a pending record with a failure line), and `ChatSession._effect` raises `StoppedByOperator` for
  a fenced refusal instead of "<effect> failed". A Stop that memory's guard raises inside the night's engine ends the
  night with its one clean line.
- **One loop per home through a slow start: the spawn reservation**.
  `spawn_loop_process` used to release its lock after a fixed 0.6 s whatever the child had done, so a second start
  could spawn a second loop over a home whose first child had not registered yet. Now, under the home state lock and
  before `Popen`, it writes the reservation `loop_status = {phase: starting, generation, spawner_pid,
  spawner_started_at}` (the child pid joins it after `Popen`, never over a child that already registered);
  `read_loop_status` reads it running while the recorded child - or, before `Popen` returned, the spawner - is alive
  (one `waitpid` + `os.kill(pid, 0)`, never `ps`, never a clock). A second start, concurrent or after the first
  returned, is refused ("already starting"); a hand start (`python -m abstractruntime.identity.life`) honors a running
  or reserved loop too. A child that dies at once releases the reservation (`stopped_by start_failed`); one refused
  by a Stop meanwhile surfaces as `StoppedByOperator`, not as a failed start. `LOOP_PHASES` gains `starting`.
- **A loop spawned before a Stop and booting after it writes nothing**: the spawner passes
  `--generation G` (the reservation's); the child checks it before anything else and exits "(stopped before it
  started ...)" with zero writes, and its first write (the boot claim replacing the reservation with its own
  `between`) is fenced at G. A kill switch that preceded the reservation is still known to the booting child
  (`after_stopped_by`), so nothing is restored.
- **The night's completion is fenced**: `build_consolidator` takes the loop's
  `fence` (required). A moved generation ends the night at the engine's next stage (`should_continue`), memory
  records no completed maintenance (`sleep_pass(cancelled=fence.moved)`, AbstractMemory 0.5.0: the idle-sleep
  boundary is not reset), the repair step declares, re-arms and authors nothing (`_sleep_authoring(should_stop=)`,
  also handed to the drainer), and neither the night voice nor failure lines follow - one clean line instead.
- **A memory commit whose guard passed lands before the Stop's bump, never after it** (AbstractMemory 0.5.0). Memory's commit guard runs under its database locks and the bump under the home
  state lock, so "guard passes -> another process bumps -> COMMIT lands" was possible. `bump_lifecycle_generation`
  now holds, in this order, the home state lock -> memory's write barrier (`MemorySystem.write_barrier` over a
  handle opened for the bump: BEGIN IMMEDIATE on the journal, graph and task-queue files) -> the runtime's own write
  barrier (exclusive) -> the generation write, released in reverse. The barrier waits at most
  `MEMORY_BARRIER_TIMEOUT_S` (10 s) for a stuck writer and then bumps anyway: the Stop never hangs, and its receipt
  says `barrier_note: "barrier timed out: <the held lock>"` (`bump_lifecycle_generation(receipt=)`,
  `hard_stop_loop`'s result). Accepted residue, documented: a LanceDB store takes no lock (one write per concurrent
  LanceDB writer whose guard passed may land after the bump; its next write is refused), and memory's two SQLite
  autocommit bookkeeping writes (`_write_pin` without a cursor, the off-file `mark_journal_applied`) stay outside a
  barrier taken through another handle. The runtime's AbstractMemory floor is 0.5.0.
- **A loop never claims a home another live loop holds; a hand start reserves like a spawn**. The
  boot claim merged its `between` over whatever stood, so a reservation naming another live child, or another live
  loop's registration, was replaced (two loops over one home). The claim is one locked, fenced read-check-write now
  (`LoopAlreadyRunning`, writing nothing, when the record is another live process's - the pid alive and the same
  incarnation). The hand start (`python -m abstractruntime.identity.life` without `--generation`) used to check
  status outside any lock and never reserve: it reserves under the spawn lock now, naming itself as the child.
- **A readmission serves new work only; a write still in flight from before the Stop stays refused**. `readmit` swapped the one fence every guard read, so a memory write of an effect admitted before the Stop,
  suspended inside its handler, passed its commit guard at the new generation once Resume had readmitted the
  runtime. An operation now keeps the admission it started under: the fenced handler table fixes each effect's
  admission for its whole duration (`lifecycle_fence.admitted`, per thread and home; a child run inside a
  subworkflow inherits it), a `ChatSession` runs its entry points under its own fence, and the holder's checks - the
  memory commit guard, the diary and artifact guards, the authoring callbacks - resolve to it. A host that stamps
  `lifecycle_fence.RUN_GENERATION_VAR` into a run's vars keeps every effect of that run judged against the
  generation it was admitted at.
- **A cached runtime authors public diary entries again after Resume**: the visit lane's
  diary authoring checked the fence the runtime was OPENED with, so after Stop -> Resume -> readmit every new turn
  carrying a public diary entry was cancelled. It checks the operation's admission now.
- **The own-time loop fails closed and writes nothing after a Stop**: an unreadable phases.yaml
  used to leave the loop running "not fenced", and any read error counted as no Stop; now the loop does not start (no
  status, no state, no wake), and a generation that moves or becomes unreadable ends it at its next boundary. After a
  Stop mid-day the loop no longer runs the day's look-back (a model call), and never writes its status, state, spend,
  personal-time usage, phase markers or task status over the Stop's records; it says so once, never as `#FALLBACK`.
- **The check is an int compare**: a process-wide table plus `<home>/.lifecycle_generation`, replaced
  atomically by `bump_lifecycle_generation` under the home state lock after phases.yaml. The bumping process sees its
  own Stop with zero I/O (measured 0.1 us per check); any other process pays one stat per check (about 2 us), reading
  the int only when it changed - instead of parsing phases.yaml at every cancellation check (about 300 us).
- **`write_loop_status` no longer runs `ps` to merge the prior status** (it reads the raw file, and the start-time
  token is read before the lock).
- **A spawn reservation whose pid was reused no longer reads "already starting" forever**. The
  `starting` liveness fold checked the pid only; it now compares the recorded start-time token exactly as `day` and
  `between` do (the spawner's `spawner_started_at` before `Popen` returned, then the child's `pid_started_at`, which
  the spawner now records with the child pid - one `ps` per spawn and per status read, never per effect).
- **A failed signal write never hides the Stop**. The bump wrote `.lifecycle_generation` before
  publishing the new generation, so an OSError there (a full disk) left phases.yaml bumped while this process's
  fences and every other process's stat still read the old generation, and `hard_stop_loop` reported "not bumped".
  The generation is published first now; a failed write removes the stale signal (a process that saw it reads
  "vanished", i.e. moved) and raises `LifecycleSignalUnwritten` with the generation the Stop moved to, which
  `hard_stop_loop` reports with its note.
- **A start the operator's Stop overtook is never reported started**. A Stop landing between the
  spawn reservation and `Popen`'s return, with the child still alive at the 0.6 s check, let `spawn_loop_process`
  return `{pid, generation}` while `loop_status` already said `stopped / kill_switch`. It now raises
  `StoppedByOperator` (the child refuses itself at its boot, or the Stop kills it).
- **A diary or artifact write whose fence check passed always lands before a racing Stop, never after it**. The diary book and the artifact store checked the fence before their write and
  outside any lock the bump takes, so "check passes -> another process bumps -> the write lands" left a write after
  the paused row. The home now has a write barrier (`lifecycle_fence.write_barrier`, a cross-process flock on
  `<home>/.write_barrier.lock`): a store write is one `FenceHolder.writing(where)` section - the barrier held shared,
  the fence checked and the write made inside it - and `bump_lifecycle_generation` holds it exclusive around the
  generation write (inside the home state lock: the home lock outer, the barrier inner). `DiaryStore(guard=)` and
  `FileArtifactStore(guard=)` take the home's fence holder (a `.writing` section), or a host's callable guard
  together with `barrier_home=` (the home whose barrier the write holds); a bare callable is refused. **Breaking for
  the gateway's shared home**: its `DiaryStore(guard=self._commit_guard)` adds `barrier_home=self.home_dir`. Memory's own
  commit points join through memory's `write_barrier`.
- **What the identity runtime stores is whole; what a screen shows is the app's fold** (root ADR 0054; operator
  2026-11-08: "the text stored should never be truncated, otherwise it's not a verbatim; the text shown is
  UI/app dependent"). Six write paths cut text to a fixed length on the way to the record; none does now:
  - `identity/digest.py`: when no whole sentence of a side fits the digest budget (one long run-on sentence,
    prose without terminal punctuation), the stored digest carries that side whole, not its first 200 characters.
  - `identity/reflection.py`: the `[kept interest: "…"]`, `[learned: "…"]` and `[topic: "…"]` markers written
    into the stored reply carry the elected words whole (they were cut at 80 characters, silently).
  - `identity/act_only.py`: a captured diary entry's `gist` (run vars, the visit sheet) is his elected gist whole,
    else the entry's first non-empty line whole (120 characters was a cut).
  - `automations/wakes.py`, `automations/models.py`, `automations/attention.py`: a note wake's `notify.title` and
    `notify.body`, the definition's `notify` and every attention item's title and body are stored whole (they were
    cut at 120 / 2,000, and the default body at 280). The automation `title` keeps its validated 120-character
    bound; `NOTIFY_TITLE_MAX`, `NOTIFY_BODY_MAX` and `NOTIFY_DEFAULT_BODY_MAX` are gone. `docs/automations.md` says so.
  - `identity/life.py`: the day-open cue's quote of an open question or problem (a prompt, so it may fold) folds
    visibly with the desk's `drive_gist` rule (first line, 60 words, " …", reread command beside it) instead of a
    silent 120-character cut mid-word.
  Tests: `tests/test_stored_text_whole_adr0054.py` (one per site, each red against the old cut) and
  `tests/test_schedule_run.py` (a 2,000+ character note reaches the attention item whole).
- **A tick look-back's summary record has an authored title, never "tick reflection: <session> #<n>"**
  (`identity/chat.py`). The tick look-back passed that template as the title and only the summary
  was authored. Both are authored now in the formation step; the tick's label is kept as structure
  (`attributes.lookback_label = "<session> #<n>"`).
- **An untitled lesson, interest or realization keeps its whole first line as its title** (`identity/reflection.py`
  `elected_title_and_words`; the diary projection's slug in `integrations/abstractmemory/identity_support.py`):
  without a `title:` line, the title is the entity's whole first line - his own line break, never a sentence split.
  The sentence split cut "Dr. Smith taught me patience." to "Dr." and "I learned that U.S. Courts are slow." after
  "U.S.", and that cut title was stored for good. A `title:` line still wins; a display folds a
  long title, the stored one stays whole.
- `search_memory` and `read_memory` never serve a topic as if it were a memory. `HomeMemoryReader`'s
  direct abstracts reads (`digest_assertions_all`, `find_tag_in_home`) skip memory's `community` records; memory keeps
  them off recall, probes, vectors and FTS itself (AbstractMemory 0.5.0).
- **A cancelled turn stops at its next effect boundary and forms nothing** (`identity/chat.py`, `identity/tools.py`,
  `identity/authoring.py`, `identity/life.py`). `ChatSession.cancel_check` (set by
  a host: the gateway's Stop fence; the own-time loop sets it to its moved lifecycle generation) is checked before
  every model call, before a tool batch and before EACH tool in it (`execute_tool_elections(should_stop=)`; the
  elections left carry `cancelled`), and before the turn forms anything. The turn raises `TurnCancelled(report,
  where)`: the report keeps only the tools that really ran (and their files), ends with a `#NOTE the turn was
  cancelled ...` line, and nothing is formed (no elections, diary, commit, episode or authoring). The authoring
  drainer takes `should_stop=` too: checked before any re-arm or claim, before each task and again after the
  capability answers - a Stop then writes nothing and hands the tasks back unspent. The own-time loop ends a
  cancelled tick with `stopped_by=kill_switch` (no failure, no backoff).
- **A failed model call fails that turn, never the visit** (`identity/visit_workflow.py`; operator 2026-11-09 "a
  visit never fails"; gateway slice finding: a failed mirrored turn ended the visited entity's whole visit run).
  Every model call of a visit turn (the v0 single call and each call the ReAct middle plans) is routed through
  `VISIT_MODEL_GATE`: its final failure, after the effect policy's retries, is absorbed into its result key
  (`_absorb_failure`) and the turn goes to `VISIT_TURN_FAILED`, which answers that turn with `level: "error"`,
  `message: "This turn could not be answered: <error>"` and `turn_error`, takes the failed turn's visitor message
  back out of the adapter transcript (the next turn never stacks two user messages), forms nothing, and parks: the
  visit continues and the next turn is answered.
- **A visitor reaches an entity within one step of its own time** (operator 2026-11-09: a visit never fails; the
  entity is reachable within a minute or two). The own-time loop used to look for a visitor only between ticks, and a
  tick with web searches and page fetches runs for minutes, so the visit door gave up first. The loop now hands its
  session a `yield_check`: once a visit door has written the visiting posture, a reply's newly elected tools do not
  run (`[tool call not run: <tool> - a visitor arrived]`, with a `#NOTE`), no correction call, format nudge, speak-now
  or liveness call runs, the turn ends on what the entity said, the tick's look-back is skipped (the day's deferred
  look-back covers it), there is no pause before the next boundary, and the loop yields right after. A visitor who
  knocks on the day's last tick, or on a tick that elects rest or a work verdict, also gets the fast close (no inline
  look-back), and a rest elected as the visitor knocks is not slept over the visit.
- **`await_loop_quiescent(timeout_seconds=None)` waits without a limit** (root ADR-0027), and a `day` status that
  `read_loop_status` does not count as running (dead or recycled pid, stale heartbeat) is quiescent.
- **A dream whose authored title starts with "Dream: " is no longer re-declared every night**
  (`identity/authoring.py`). The sleep's repair read any dream titled "Dream: ..." as a legacy mechanical dream and
  declared its title and summary pending again, so a dream the summaries capability model had titled that way was
  re-authored night after night; the rule now applies only to a title with no authored stamp (the capability model,
  the operator or the entity).
- **The night's topics keep their own share of the authoring budget** (`identity/life.py`, `identity/topics.py`). The
  metadata repair ran first with the whole `authoring_cap` (64 calls), so on a home with hundreds of fields to repair
  it spent every call and the night's new topics formed untitled; the repair now runs within the cap minus a reserved
  quarter (`TOPICS_AUTHORING_SHARE`, 16 calls of 64), the communities pass authors within what it left, and the
  night's topics line says the budget and the calls spent (`authoring budget 16 calls, 12 spent`).

### Security

- **A skill folder is served only under a root the host registered** (`integrations/abstractcore/skill_roots.py`,
  `effect_handlers.py`). The `read_file` interception that serves an activated
  skill's files took its folders from run variables a client writes (`skills` mapping roots, `_runtime.skills_resolution`
  and the `<path>` of `_runtime.skills_block`), so any client could read every file beside any `SKILL.md` on the machine
  (another account's imported shelf, `~/.claude/skills/*`). The folders now come from the host: it registers its skill
  roots once per process with `register_skill_roots([...])` (`registered_skill_roots`, `clear_skill_roots`), and a
  folder is served only as `<root>/<name>` or `<root>/skills/<name>` of a registered root, for a skill NAME the run
  selected (`_runtime.skills_resolution.active`, else the `skills` names, plus the `<name>`s of `_runtime.skills_block`).
  With no root registered nothing is served and every path takes the normal `read_file` path (fail closed). A host
  other than AbstractGateway that relied on the interception must now register its roots.
- **An unsigned entity stamp never makes a turn an entity's** (`core/turn_author.py`).
  `answer_author` labelled any run carrying `_runtime.entity.entity_id` as that entity's answer, so a client sending
  `{"_runtime": {"entity": {"entity_id": "entity:example"}}}` made an agent's reply replay as "example (entity): ...".
  The fallback now counts only a FINALIZED stamp, the shape the host's door signs (`entity_id`, `channel`,
  `session_id`, `nonce`, `run_id` and `sig` all present); anything else is the agent's turn. The runtime cannot check
  the signature (the key is the host's): the host must never let a client write `_runtime.entity` (AbstractGateway
  drops it at `/runs/start`).

## [0.18.4] - 2026-11-07

### Changed

- **Entity LLM calls made by the runtime wait at least 1800 seconds** (`identity/chat.py`, `identity/life.py`,
  ADR-0027 §3). The entity Ephemeral's own-time loop stopped itself after three turns in a row hit "timed out after
  120.0s (configured timeout)" on a slow but healthy remote model. The runtime's entity lanes (an entity's own time,
  its sleep narrator and the command-line visit) now read one patience window: `ABSTRACTRUNTIME_ENTITY_LLM_TIMEOUT_S`
  defaults to 1800 (was 120); a value below 1800 is raised to 1800 with one stderr line; a negative or unreadable value
  is treated as a typo and gives 1800; `0` still means no client timeout. The retry budget
  `ABSTRACTRUNTIME_ENTITY_RETRY_BUDGET_S` defaults to 1800 (was 180) and is never shorter than one attempt, so an
  attempt that used its whole window is not retried while a fast failure (a reset connection) still retries. Visits
  hosted by the gateway are not in these lanes: they use AbstractCore's `default_timeout` (7200 seconds by default).

### Fixed

- **A slow but healthy own-time day is never read as dead** (`identity/life.py`). The loop wrote its "alive" status only
  between ticks, and one tick can make up to about 22 LLM calls. With 1800-second calls, a healthy tick could outlast
  the dead-loop bound, and the gateway's repair sweep could then start a second loop on the same entity. The loop now
  refreshes its status before and after every LLM call (only its own open day, never a stopped or foreign status), and the
  bound `LOOP_STATUS_STALE_SECONDS` is two full attempts plus ten minutes (4200 seconds). A loop whose process exited
  is still detected at once. The status file is now written atomically, so a reader never sees a half-written file.
  With `ABSTRACTRUNTIME_ENTITY_LLM_TIMEOUT_S=0` (no timeout, opt-in) a single call can still outlast the bound.

### Known limit

- While an entity's own time is in the middle of a long LLM call, a visit can be refused until the call ends: the
  gateway visit door waits 55 seconds for the loop to pause. Letting a visit interrupt the loop's call is planned for a
  later release.

## [0.18.3] - 2026-11-07

### Fixed

- **A voice engine that cannot run says why in plain words** (`integrations/abstractcore/discovery_queries.py`,
  framework backlog 0968). When the engine raised while `local_list_tts_models` was building the listing, the
  listing's `error` was the bare exception text (for example "No module named 'kokoro'"). It is now AbstractVoice's
  own `unavailable_reason`: from the catalog already returned, or else from the engine-free providers-only catalog.
  Only when neither gives one is the exception kept, worded for a person: "the text-to-speech engine could not list
  its models: ...".

## [0.18.2] - 2026-11-07

### Fixed

- Embedding imports preserve their original error and use the public AbstractCore module path.


## [0.18.1] - 2026-11-04

### Added

- **A public update names the identity its run started with** (`replay/view.py`, ADR 0052, framework backlog 1119).
  The replay view copies the run's `vars._runtime.identity_ref` (`{"of", "rev"}`, stamped by the gateway on visit,
  summon, meet and session runs) onto each of that run's `public_update` messages as `identity_ref`, so a host serves
  cached views with the ref already in place. A run without a ref, or with a malformed one, gives no field, never an
  invented one; a snapshot read after the updates were folded stamps them then; the ref is part of the snapshot
  digest, so the cursor token (an ETag) changes with it, and a run without a ref keeps the token it had.

## [0.18.0] - 2026-11-04

### Added

- **Acquaintances of an entity or a session** (`identity/acquaintances.py`, framework backlog 1119):
  `entity_acquaintances(home_dir, run_store)` and `session_acquaintances(session_id, run_store)` answer two
  lists, the other entities and the humans it interacted with, each with first and last seen, the number of
  interactions and the task kinds. Computed from existing records, never stored: the participants the
  gateway stamps on visit, summon and meet runs, the person who started a run (`_meta.origin`), and who left
  a task in the task inbox. Identifiers only; interactions with no usable identifier (an `agent:` participant,
  a participant without a namespace, a person-started run without an actor) are listed under `not_counted`,
  never guessed. See `docs/track-record.md#acquaintances`.
- **An entity's digital identity: its look and its voice, one record in its home**
  (`abstractruntime.identity.digital_identity`, framework ADR 0052 draft, backlog 1119; `docs/digital-identity.md`).
  `<home>/identity.json` holds the avatar design, its colour and the voice (`schema_version` 1). `write_identity` checks
  every bound (`IdentityInvalid` names the field), runs the gateway's `check` hook, appends an `identity_changed` row to
  `state_history.jsonl` (the `phase_changed` field names) before the file is replaced atomically, and never gives a
  `rev` twice: one write at a time per home (`<home>/.identity.lock`, held from the read to the file replace). A
  file it cannot read in full raises `IdentityUnreadable` on read and on write, never "no identity".
  `default_identity` is the recorded default anyone can recompute (SHA-256 of the entity id, eight engraved Watch
  designs, an HSL colour, the gateway's default voice as a value); `record_default_identity` writes it once and never
  overwrites. `identity_at_rev` answers an old rev from the biography. The reference `entity:castor#3` rides
  `vars._runtime.identity_ref` beside the signed entity stamp (`stamp_run_identity_ref`, `read_run_identity_ref`,
  `parse_identity_ref`). `identity_fingerprint` derives the individual's `actor_id` from the manifest, with the diary's
  chain head. Homes without `identity.json` are untouched.
- **The summon tells the entity when its look or voice changed**: one sentence ("Your voice changed on 4 November (by
  laurent): now fr-male-rene.") in the summon prelude, after the identity core, when the change is newer than the
  previous summon (`render_summon_prelude(home_dir=, identity_since=)`, `ChatSession(identity_since=)`,
  `build_visit_workflow(identity_since=)` or `_visit.identity_since` seeded by the door).
- **The idle unload reaches the host: `set_model_idle_unload(seconds)`** (host facade, `LocalAbstractCoreLLMClient`,
  `MultiLocalAbstractCoreLLMClient`; framework backlog 1115). It relays AbstractCore's process-wide idle-unload time
  (ADR 0012, default 60 minutes): a model no call has used for that long is unloaded, a locked model never is, and a
  running call is never interrupted. `None` or `0` turns it off; a refused value answers `ok: false`. Local residency
  rows of in-process text models carry `idle_unload_s`, `last_used_at` and `idle_unload_at`, and pass through Core's
  `in_use` and `locked_by`, so a host can show a model that is in use, or kept loaded by another account.
- **`set_model_residency_locks(locks)`**: the locked models become exactly the listed ones (resident ones locked,
  missing ones loaded then locked, the others unlocked), each pair with its own outcome. A lock names a provider and a
  model; the `base_url` of a configured endpoint is not part of the key. A host stores its "Keep loaded" set and
  replays it at start; as a `set_` name it is replayed when the gateway's model host rebuilds a client.
- **Track record and reputation of an entity or a session** (`identity/track_record.py`, ADR 0052 §5-§6,
  framework backlog 1119). `entity_track_record(home_dir, run_store, ledger_store, *, book_head)` and
  `session_track_record(session_id, run_store, ledger_store)` answer sessions, runs, successes, failures,
  cancelled, in progress and tokens (input, output, total), per task kind (`<phase or role>/<workflow_id>`,
  from ids only) and overall; `compute_reputation(record)` gives trust per task kind,
  `(successes + 1) / (successes + failures + 2)` with its support, the support-weighted overall, and the
  last 20 outcomes as evidence ("no experience yet" when there is none). Everything is computed from
  existing records; nothing is stored. Only work from outside counts: turn roots a person or a trigger
  started and automation occurrences (once each, however many attempts); sub-runs, controllers, the
  session memory run and runs another run or the system started never do. Outcomes come from the run
  status, the occurrence's `automation.completed` decision, an entity task marked `done`, and the work
  order verdicts (`done:` / `blocked:`). Feedback from people and agents (`feedback_user`,
  `feedback_agent`) is typed but has no source yet, so every answer says `basis: "completion-rate only"`.
  Each answer names its subject's `identity_ref` and, for an entity, its `identity_fingerprint`.
  See `docs/track-record.md`.
- **`metrics_many` also returns `tokens_input` and `tokens_output`** (SQLite ledger): `usage.input_tokens`
  (else `prompt_tokens`) and `output_tokens` (else `completion_tokens`) of completed `llm_call` records.
  Existing keys are unchanged.
- **`identity.life.read_work_order_verdicts(home_dir)`**: the verdicts the work lane archived in
  `work_order.done.md`, oldest first.

### Changed

- **AbstractCore 2.34.0 or newer** (base install, `apple` and `gpu` settings): the idle unload and its residency rows
  come from it.
- **A load with `ttl_s` or `keep_alive` on an in-process model** still lists them under `unsupported_options`; its
  warning now names the idle unload that applies instead ("the idle unload setting applies (60 min unused, never while
  it is kept loaded)"), or says it is off.

### Fixed

- **A run store whose index cannot attribute a session is refused, never started as a plain chat**
  (`core/run_attribution.py`, framework backlog 0938). A third-party store with `list_run_index` but no `session_kind`
  in its rows (an index written before automations v1) used to report "no kinds" and let a `discussion-session:` start
  through as an ordinary chat, skipping the discussion's read-only anchor. It now raises `SessionAttributionError`
  naming the missing column; a store with no index at all names both methods it lacks. Built-in stores are unchanged.
  Upgrading a custom store: add `session_kind` and `role` to its index rows, or implement `session_kinds(session_id)`
  (see `docs/automations.md`, "Storage guarantees").
- **A tool that reports its failure as a JSON object keeps its explanation** (`integrations/abstractcore/tool_executor.py`,
  framework backlog 1005). A failed result's `output` used to be emptied when the tool returned its failure as a JSON
  string, so the agent and the run ledger saw only the error line (for a web search: "No results found", never the
  backends tried, the bot-challenge status or the hint). The parsed object is now the result's `output`, as for a tool
  that returns a dict; a plain text error (`Error: ...`) still keeps an empty output and its message in `error`.
- **The visual-flow Agent node shows the model that explanation** (`visualflow_compiler/compiler.py`). A failed call
  whose `output` is an object is rendered as its error line followed by the object (`[web_search]: Error: No results
  found` then the JSON), under the same `agent_observation_max_chars` limit; a plain text failure still shows its error
  line alone.
- Failure payloads are now recorded in the run ledger and the history bundle like success payloads: whatever a tool puts
  in its failure JSON (its parameters included) is stored with the run.

## [0.17.0] - 2026-11-03

### Added

- **`EffectInterrupted`: a host stop interrupts an effect, it never fails it** (`core/effect_cancellation.py`,
  exported as `abstractruntime.EffectInterrupted`; gateway backlog 1103). A host that stops on purpose (a gateway stop
  or restart, a model-host restart for an update) raises it into the effect in flight. The runtime writes no terminal
  record and re-raises: the ledger tail stays STARTED, the run stays RUNNING at its last durable step, and the next
  tick (after the host is back) sends the effect again under the same idempotency key as attempt 1, so no retry is
  consumed and nothing reaches an automation's outcome. A BaseException like `EffectKilled`, so no `except Exception`
  on the way turns it into a failure. The interrupted call's live text ends `cancelled` with `detail: "interrupted"`.
- **Public updates carry their type in replay** (`replay/view.py`). A `public_update` message has
  `update_kind`: `"finding"` when the actor typed the event `kind: "finding"` (a fact or piece of
  evidence it established), otherwise `"progress"` (what it is doing), including events written
  before the field existed. Clients use it as given, for example to list the findings of a
  conversation. See [docs/replay.md](docs/replay.md#public-conversation-messages).
- **Hosts tell model errors apart through the discovery facade** (`integrations/abstractcore/discovery_facade.py`):
  `ModelNotDownloadedError` and `ModelNotFoundError` are AbstractCore's own classes, so a host (the gateway's
  "model not downloaded" sentence) catches exactly what Core raises and reads `model` / `provider` without importing
  AbstractCore.
- **Replayed tool calls keep the model's call id** (`replay/view.py`). An `activities[].tool_calls[]`
  row keyed by a runtime call id (`id`) also carries `call_id`, the model's own id, which is what a
  public update's `source_call_ids` cite; a client can now resolve a finding to the results it cites.

### Changed

- **Compare reads text next to a number or true/false exactly** (`visual/builtins.py` `_exact_from_text`, ADR 0049 as
  amended 2026-11-03: "we must have automatic conversion"). When one side is a number or true/false and the other is
  text, the text is read as that type by the Flow editor's literal grammar (a decimal number, ASCII, no `_`, finite,
  a whole number kept exact beyond 2**53; or `true`/`false` in any case), so `5 == "5"` and `"9" < 10` are true. This
  covers flows saved by AbstractFlow 0.13.0 (a typed value stored as text next to a wired number) and text arriving at
  run time from a form, an API input or a model, with no re-save. `"1_000"`, `"0x10"`, `"inf"`, `"nan"`, full-width
  digits and empty text are refused. Text against text is never converted (`"10" > "9"` compares text). The table is
  `tests/fixtures/visualflow/compare_text_operands.json`.
- **Compare refuses values it cannot compare instead of answering False** (`visual/builtins.py` `control_compare`,
  ADR 0049). Text that is not a number next to a number, or not `true`/`false` next to true/false (`5 == "five"`,
  every operator; text that is a number or true/false is read as one, see "Compare reads text next to a number or true/false exactly" above), and `<`/`<=`/`>`/`>=` between values Python cannot order
  (`None > 3`, a list and a number), now fail the step with an error naming the node and pins: "Compare 'Is five'
  (node same) ==: b is text ('five'), not a number; a is a number (5)." Before, these answered False silently.
  Equality of same-type values and Python's equality across other types (`None == 5`) are unchanged; true/false
  compares with numbers as Python's booleans do (`True == 1`).
- **AbstractCore floor 2.33.0** in the base install and the `apple` and `gpu` extras (was 2.32.0).

### Fixed

- **An automation occurrence whose agent completed with `success: false` names the agent's reason**
  (`automations/attention.py`). The attention item and the `automation_failed` email said "completed" (the run's status
  word); the reason is now read from the Agent node's `meta.error`, and when there is none it says so plainly.

## [0.16.0] - 2026-11-02

### Added

- A package agent skill documents operating durable workflows, waits, storage and structured execution through the public Python API.

- **`config_facade.model_catalog(..., hub_page=, hub_page_size=)`** (backlog 0864) passes a page of Hugging Face search
  results through to AbstractCore, which reports it in `hub.search`. It is passed only when asked. An AbstractCore
  whose catalog has no `hub_page` raises `AbstractCoreTooOld` in words instead of answering page 1.
  `AbstractCoreTooOld` accepts a `required=` description. `extended=True` (every quant of a row's own GGUF repos,
  AbstractCore 0865) is passed the same way and refused in words on an AbstractCore without it.
- **Fit refusals (AbstractCore ADR 0010) reach every surface.** A load refused with `ModelDoesNotFitError` (the model
  does not fit even after evicting what is not locked) is never retried (a retry would only evict more); its
  sentence is the effect error, its fields (needed/free bytes, kept models, suggestions) are kept on the ledger
  record (`result.error_details`, `EffectOutcome.error_details`) and in the residency diagnostics (`refusals`, the
  last five), and the default pair keeps the sentence instead of the configuration hint.
- Opt-in public actor updates and explicit human-guidance acceptance messages share durable replay history, with host-origin checks and exact original human words.
- Conversational ASK_USER policies distinguish indefinite essential questions from 120–300 second clarifications with predeclared safe fallbacks. Fresh question keys, strict server cutoffs, durable outcomes, retained late replies and pause/restart behavior reuse the existing due-wait indexes.
- Participant-checked live entity-visit corrections use the existing durable sidecar and fresh attributed recall before the next actor reasoning boundary.
- The package agent skill documents these question and guidance contracts.
- **A decision is an `LLM_CALL` whose output spec says `modality: "decision"`** (`integrations/abstractcore/llm_client.py`, `effect_handlers.py`). Payload `{"prompt": <state>, "output": {"modality": "decision", "questions": {...}, "truncate"?, "model"?}}`; the result is the usual normalized shape plus `decision` (the System One body: `model`, `answers`, `usage`, `latency_ms`[, `truncated`]) and `answers` (= `decision.answers`), with `content` the answers as JSON, `outputs`/`resources` empty and `provider`/`model` the decision model that answered. No artifact, no store, no `generated_media` tag; the resolved action id is `decision`. Local: the resolved spec (the `output.decision` capability default, or the spec's own route) reaches `generate(prompt, output=[spec])` and core runs it through its decision pool on any text provider; the pooled client serves a decision-only request on the capability host without building the text model. Remote: `POST {core}/v1/decisions` with `model` = `provider/model`, `state`, `questions`, `truncate`; the server's body maps to the same result.
- **`decide`: a calibrated decision as a durable child run and a VisualFlow node** (`integrations/abstractcore/run_facade.py`, `visualflow_compiler/visual/executor.py`, `visualflow_compiler/compiler.py`). `run_facade.decide(run_id, state=..., questions=..., model=None, truncate=None)` starts a child `LLM_CALL` whose output spec is `{"modality": "decision", "questions": ...}` (plus `truncate` and `model` when given); a `model` written as `provider/model` (`mlx/jaredpalmer/kev-4b`) travels whole and AbstractCore splits it. The `decide` VisualFlow node asks one typed question (inputs `state`, `question`, `question_type` = `choice` | `noul` | `score`, `options` / `levels`, `provider`, `model`, `truncate`) and fills that type's output pins (`answer`, then `choice`/`confidence`/`probabilities`, `noul`/`verdict`, or `score`/`level`/`confidence`/`probabilities`, plus `model`, `success`); a wired `questions` object asks several System One questions at once and fills `answers`. It lowers to the same `LLM_CALL`; the result sync also carries `decision` (the System One body), `content` and `meta` (`abstractflow.decide.v1.meta`).
- **A local model's loaded context, through the config facade** (`integrations/abstractcore/config_facade.py`).
  `configured_model_context(provider, model)` and `set_model_context(provider, model, tokens)` read and set the
  per-model context AbstractCore loads a model with (AbstractCore ADR 0010 point 8: a set context is a pin, loaded
  as is or refused, never shrunk). The gateway's Resources page "Set context" calls them instead of importing
  AbstractCore. On a Core without `abstractcore.config.model_contexts` (every release up to 2.31.1) they raise
  `AbstractCoreTooOld` naming what is missing.
- Tool approval waits can carry optional `details.public_action` (`text`, `update_id`, `run_id`,
  `iteration`, `call_ids`) supplied by an actor adapter alongside its proposed effect batch.
  Runtime validates its owning run and exact call identities, narrows it to the actual gated
  subset, and withholds mismatched or private-fenced context. The canonical replay preserves
  this public explanation; approval decisions and policies are unchanged.
- Hosts can start Core's independently authenticated HTTP listener through a lazy Runtime lifecycle facade; unused server dependencies stay unloaded.

### Changed

- **Changelog entries are fragments** (`changelog.d/`, `scripts/changelog.py`). A pull request adds
  `changelog.d/<pr-or-slug>.md` instead of editing `CHANGELOG.md`'s `[Unreleased]` section, which conflicted on every
  merge of main; the release assembles the fragments (`python scripts/changelog.py assemble X.Y.Z`) and CI fails a pull
  request that changes `src/` without one.
- **AbstractCore floor `>=2.32.0`** (`pyproject.toml`, `integrations/abstractcore/output_specs.py`): the `decision` output modality (`output.decision` route, `/v1/decisions`, `MultimodalGenerateResponse.decision`, `output_has_generated_media` False for it).
- **Voice transport is not a conversation attachment** (`history_bundle.py`, `conversation_search.py`). A session turn in
  a history bundle no longer lists the recording of a spoken question or the read-aloud audio of an answer among its
  `attachments` and `artifacts`; `is_voice_transport(tags)` recognizes them (`purpose` `voice_playback` or
  `voice_dictation`, `artifact_space` `voice`, a `voice/` or `client:voice/` path). The bundle's own `artifacts`
  section still lists every artifact. Clients that showed voice recordings as turn attachments stop doing so.

### Fixed

- **A console change to a non-text default no longer reloads the text model, and a runtime never
  leaves a dropped copy resident.** Found on the Linux test server (two accounts, a 9B GGUF):
  changing the embedding default evicted the admin runtime's whole text pool; the boot-warmed copy
  was dropped, its eject skipped because both accounts' runtimes named the same default, and the
  copy stayed in memory with no pool holding it ("resident ~5.4 GiB + 8.3 GiB cache ... still in
  use by runtime client 70a5b844f350, runtime client 70a72f833890; not ejected"), with nothing
  running. Now: (1) a capability-only refresh keeps every pooled text client and hands it the new
  routes in place (a move of the `input.text` route itself still rebuilds); (2) a configured default
  the client has not built is no longer a residency claim (a loaded one is its `pool` claim, one
  loading is `building`); (3) when another runtime really keeps its own copy, the dropping runtime
  still unloads its OWN HuggingFace copy (`released_own_copies`; MLX shares weights and is left to
  the process-wide eject); (4) holders are named in words: `MultiLocalAbstractCoreLLMClient
  .residency_owner_label` (hosts set it; the gateway: "the runtime of account default/bob"), claims
  carry `owner_label`, residency records carry `held_by`, and the skip reason reads "the runtime of
  account default/bob (loaded) still uses it". HuggingFace process rows also carry the loaded
  `context_length` and `kv_cache_bytes` (AbstractCore 2.31.1).
- **A standalone `LocalAbstractCoreLLMClient` claims the model it holds** (ROOT backlog 0897): another
  client's default switch or a server unload no longer ejects weights it still uses (its next call
  silently reloaded them). A pool unregisters the clients it adopts (its own claims cover them).
  A dropped HuggingFace copy is never lent to another request again, a copy a call just started on
  is freed when that call ends, and one waiter per model frees every copy a later switch drops.
- Conversational answer and retained-reply replay preserves original human whitespace and the actual question generation identity, including after another question has opened.
- Completed flows with an explicit structured `result` and no prose now retain a final JSON message in replay. Generic agent envelopes are not serialized; internal-bearing results are withheld from the public projection while their technical run output remains available.
- **A pin expression that is a single bare word explains how to pass text** (`visualflow_compiler/visual/pin_expressions.py`).
  `hello` is read as a Python name; its `NameError` now says to write `'hello'` with quotes or to set the pin's value
  instead of an expression.
- **The residency listing carries the model's slot pool** (`integrations/abstractcore/llm_client.py`, core ADR 0011, OVH 8080 pass four). A GGUF row now carries `slots`, `slots_max`, `active`, `queued`, `copies`, `slot_bytes`, `shared_weights` and `slot_refusal_reason`, folded from Core's process row. Before, only the gateway's `/host/state` showed them, so the Resources table could not say that requests were waiting, or why another slot is not built. A row whose holders share one weights instance no longer says "each a full copy of the weights".
- Structured generation carries JSON schemas across execution-client boundaries and creates validators where inference runs, preserving nested validation and defaults without serializing Python classes. Remote Core calls forward their structured response format and explicitly refuse Python validator classes that cannot cross HTTP. Local explicit validator classes remain supported. Request serialization failures are not retried, and execution diagnostics include the actual ledger step identity.

## [0.15.3] - 2026-10-29

### Fixed

- **An index build no longer announces its journal replay as changes** (`JsonFileRunStore`; a 0.15.2 regression).
  0.15.2's first sync after a build re-applies the journal tail before the build's mark (backlog 0863) and noted every
  replayed record on the change ring: a plain read (`list_run_index`, `warm_session_index`) moved `change_token` and
  `changed_run_ids` then named every run in the tail as changed, so a change-driven consumer re-read every run after
  each build (the gateway's mail notification check loaded all 400 finished runs of its index test on every full
  pass: AbstractGateway backlog 0965, red CI since 0.15.2). Replayed records are still filed into the session,
  children and active indexes (the 0863 guarantee stands). A replay is quiet only when it says nothing new: its file
  was known to the store at the build's mark (the scan memo, the sidecar included) and is unchanged since; a file
  that landed or changed during the build (the record precedes the file, J53-1), a known run now gone, and every
  record after the mark are announced as before, so a consumer whose token already covers the record still learns
  of it; a store with no sidecar announces its first replay (a bounded one-off). Tests:
  `tests/test_json_file_run_store_change_token_after_build.py` (red on 0.15.2).

## [0.15.2] - 2026-10-28

### Fixed

- **An index build no longer loses a run another store object creates during it** (`JsonFileRunStore`, backlog 0863).
  A new run's `+` journal line is written before its file is published; a session/children/active index build that
  marked the journal between the two and scanned the folder before the file landed missed the run: the listing until
  the next reconciliation, the session and children indexes for the life of the store object (the reconciliation
  repaired only the listing). Seen as a gateway that answered 404 `session_not_found` when archiving a session written
  while it booted (AbstractGateway backlog 0963); a split-mode runner creating a run during the API process's boot
  hit the same window. The first sync after a build now re-applies the journal's last 16 KiB before the build's mark
  (`JOURNAL_REPLAY_TAIL_BYTES`; idempotent, once per build), and the reconciliation re-files a run it repairs into
  the session and children indexes too (and drops a vanished one from them).

## [0.15.1] - 2026-10-28

### Changed

- **Is Empty (`is_empty_string`) accepts lists.** Nothing (unset / `None`), an empty text, and a list, tuple, set or object with no elements are empty; whitespace, `0`, `False` and a list with any element are not. Before, a list was turned into text (`"[]"`) and was never empty, so a flow checking a list variable needed Array Length + Compare (operator QA 2026-10-28, the email-triage flow). Backlog 0862 (numbered 0857 in the 0.15.1 package). AbstractFlow lets the node take a list once this is released.

## [0.15.0] - 2026-10-28

Requires AbstractMemory 0.4.0 or newer (`abstractmemory.graph`). ADR 0011, backlog 0856.

### Upgrade notes (one-way)

- AbstractMemory 0.4.0 rewrites a SQLite home's JSON-text vectors as float32 BLOBs on first write; a 0.3.x reader then
  finds no vectors (zero semantic hits, no error). Every process sharing a home or a plane moves to AbstractMemory 0.4.0
  (and so AbstractRuntime 0.15.0) together. See the AbstractMemory 0.4.0 changelog.
- The JSON-file ledger writes a `ledger_<run_id>.idx` row index beside each ledger; older readers ignore it, and a
  missing or stale index is rebuilt from the ledger.

### Added

- **Ledger row address** (on 0.14.0's per-run `seq`, ADR 0010): `LedgerStore.get_record(run_id, seq)` returns one
  record ready to read (a terminal row's `$slim` markers resolved against its STARTED row, offloaded leaves
  rehydrated); `find_started_record(run_id, step_id, before_seq=)`. SQLite reads by primary key; the JSON-file backend
  keeps a per-run line-offset index and checks the seq persisted on the line, so a point read is a seek and one line
  parse (0.05 ms at 10,000 rows instead of a 58 ms full parse), and a rewritten file is read by persisted seq, never
  by position.
- **Step context**: an effect handler reads its own step with
  `abstractruntime.core.step_context.current_step_context(run.run_id)`: `run_id`, `step_id`, `seq` of the attempt's
  STARTED row, `node_id`, `attempt`, `effect_type`. Out of band like the progress callback; `None` outside a runtime
  step; a nested runtime shadows it.
- **Memory facade** (`abstractruntime.integrations.abstractmemory.memory_facade`): `MemoryFacade` (`open_mind`, `stamp`,
  `stamp_current_step`, `reconstruct_source`, `history`, `sources_of`, `derived_from_source`, `apply_revision`),
  `is_real_run`, and the AbstractMemory names hosts use re-exported unchanged (`GATEWAY_NAMES`). Hosts no longer import
  `abstractmemory`. `reconstruct_source` answers `ok`, `source_gone` or `unsupported`, never raises for a missing
  source. See [AbstractMemory integration](docs/integrations/abstractmemory.md).
- **Provenance stamping**: `MEMORY_FORM`, `MEMORY_KG_ASSERT` and the `DIARY_WRITE` graph projection add a `sources`
  list to the provenance they write: the `step` ref `(run_id, seq, step_id)` of the writing effect, an `artifact` ref
  for the verbatim / `span_id`, a `diary_entry` ref for a projection. The legacy flat keys stay. A synthetic run id
  (`chat-<session_id>`, `diary:<entity_id>`, ...) is never a `run`/`step` ref: the home-direct chat path carries
  artifact or diary-entry refs only.
- **`MEMORY_REVISE`** (`EffectType.MEMORY_REVISE`, VisualFlow node `memory_revise`): one revision of a mind (assert /
  close / narrow ops) applied atomically through `MemoryGraph.apply_revision` as a workflow step, its provenance
  stamped with the step that applied it. Handler from `build_memory_kg_effect_handlers` (new optional `journal=`); a
  store that is not a graph home answers a failed outcome naming AbstractMemory backlog 0057; an ungrounded revision
  fails with `ProvenanceRequired`.

- **Source integrity** (architect pass): every ref the runtime stamps carries a `digest` (step: sha256 of the STARTED
  record as written, also on the step context; artifact: the sha256 of its bytes; diary entry: the chain
  `record_hash`), and `reconstruct_source` verifies it: new statuses `mismatch` (the bytes changed since the memory
  named them) and `partial` (a `$slim` marker or an `{"$artifact": id}` leaf whose artifact is gone or uncheckable;
  never `ok` with a hole). A step ref resolves only to its STARTED row (a rewritten JSON-file ledger answers
  `source_gone`, never another row).
- **Owner partitions**: `MEMORY_REVISE` and `MEMORY_KG_ASSERT` write only the run's own owners (its run, its session,
  the global owner); `MEMORY_REVISE` refuses `recorded_at` (record time is the store's); a replayed revision reports
  the memory's sources, not the retry's step.
- **Persisted JSON-file row index** (`ledger_<run_id>.idx` beside the ledger): a fresh process reads a 100,000-row
  ledger's row in 39 ms instead of 320 ms.

### Changed

- `AbstractMemory>=0.4.0` (was `>=0.3.2`).
- SQLite `find_started_record` reads newest-first with `LIMIT 1` (0.01 ms at 10,000 rows instead of 3.5 ms).

## [0.14.0] - 2026-10-27

Two features: the replay view with a durable per-run ledger cursor, and the session tool policy.
Upgrade note (one way): hash-chained JSONL ledgers written by 0.14.0 carry `seq` on each line, and
an older `verify_ledger_chain` reports those lines as tampered. Upgrade writers and verifiers together.

### Added

- **Replay view** (`abstractruntime.replay`, ADR 0010, backlog 0854; framework root 0993). One
  pure, deterministic reducer turns a run tree's ledger records and snapshots into the view every
  client renders (`replay_view_v1`).
  - What the view holds:
    - the run tree through delegation waits, with the workflow each run executes (bundle,
      version, flow, the root's entrypoint);
    - the status;
    - the person-facing interaction: a delegated child's approval, a question, or a prompted
      event, with the resume command to send;
    - verbatim messages with their provenance (run, steps, seqs, artifacts);
    - one activity per node execution (AbstractFlow's `executionIdFor` rules), with tool calls
      matched to results;
    - artifacts and usage.
  - `$slim` pointers are resolved, and a windowed history is named in `warnings`.
  - A wait is over when a resume OR a later step of the same run exists.
  - `ReplayView.apply` folds newer records and returns a delta. `apply_delta` and `diff_views`
    are the client side. `replay(all)` equals the fold of any chunking.
  - Wait kinds agree with `automations.attention`. The person-facing kinds are tool_approval,
    ask_user, and an event that carries a prompt. The non-person-facing kinds are delegation,
    timer, job, pause, automation_wake and signal. The interaction carries `answer_payload`.
  - A session tool policy's `abstract.tool_policy_decision` record is an activity
    (`tool_policy`: allowed or denied by the conversation's policy, its revision and who set it),
    never an interaction. The tool rows it names carry `decided_by: "session_policy"`.
  - Every delta carries typed `changes` (status, interaction opened or closed, run, message,
    activity, artifact, each with its run and seq), for push dispatch.
  - Cursor tokens name each run by a collision-free prefix that is safe in a header.
    `resolve_cursor_token` maps a token back.
  - The conformance cases ship with the package (`abstractruntime.replay.conformance`), so a host
    or a client port can run them. Guide: `docs/replay.md`.
- **Durable ledger cursor** (gateway backlog 0082). A JSONL append after a torn last line (a
  crash mid-write) now starts a new line; before, the fragment and the whole new record were
  dropped together:
  - Every ledger store numbers a run's records (`seq`). `JsonlLedgerStore` persists the number
    on each line, allocated under an advisory file lock, so two processes never take the same
    seq. Lines written before the field existed read as their position.
  - `LedgerStore.list_after(run_id=, after=, limit=)` works on every store: an indexed read on
    SQLite, a byte-offset seek on JSONL. `append()` returns the seq.
  - `ObservableLedgerStore` passes `seq` to subscribers.
  - `SqliteLedgerStore.import_records` copies a ledger in and keeps its `seq` values (migration).

- **Session tool policy**: a conversation's standing "approve / deny without asking", applied before any approval
  wait. `abstractruntime.storage.session_tool_policy` (`SqliteSessionToolPolicyStore`,
  `InMemorySessionToolPolicyStore`, `session_policy_names`), `Runtime(session_tool_policy_store=...)` /
  `Runtime.set_session_tool_policy_store()`. The TOOL_CALLS handler reads the run's session row once per batch: an
  approved call runs with no wait, a denied call is refused with no wait, and each decision writes an
  `abstract.tool_policy_decision` ledger record (who set the policy, when, which selector matched). It applies to
  every run of the session (the run tree's root session: delegated children, resumes, restarts); approvals never
  reach untrusted-input runs or automation occurrences (a deny does, and also wins over an approval that was shown
  before it); never beyond the run's `allowed_tools`. Before this, "approve all" lived in each
  client, which answered the wait after it arrived: the gate flashed in every app and in the ledger. See
  [tool-approval.md](docs/tool-approval.md#session-tool-policy-a-conversations-standing-word).

### Changed

- Hash-chained ledgers: `seq` is persisted on each JSONL line and excluded from the hash by this
  version's `compute_record_hash`. A verifier from an older version includes it, so it reports
  every line written by this version as tampered. Upgrade every process that verifies a chain
  (`verify_ledger_chain`) together with the writers.
- `LedgerStore.list()` records carry `seq`. A consumer that compared records to `asdict(record)`
  sees one more key. The hash chain excludes `seq` (it is the store's position, assigned after
  the hash).

## [0.13.1] - 2026-10-27

### Added

- `email_facade.resolve_srv_default` (AbstractCore's mailbox SRV lookup) is re-exported, so a host's mail code imports
  it from the runtime facade like every other mail name.

## [0.13.0] - 2026-10-27

Requires AbstractCore 2.30.0 or newer. Personal voice stores need AbstractVoice 0.19.0.

### Added

- The voice library runs on the capability registry the runtime's synthesis uses (same voice manager, settings and
  stores); a model pool manages voices on its media host. `clone_voice(..., language=)` forwards the recording's language;
  `clone_voice` answers the stored record, flat, with its `voice_id`, locally and remotely; a remote refusal keeps the
  server's status (400, 401, 403, 404, 409, 413, 422, 429, 501, 503); an empty voice id is refused (422).
- **Per-runtime voice stores**: `voice_store_dir` and `voice_library_dirs` in a runtime's AbstractCore client settings give
  it a personal voice store in front of read-only libraries; its synthesis, voice catalog and voice library use the same
  stores. `deploy_voice_to_library(voice_id)` on the voice library facade (local runtimes). See
  [Voice library](docs/integrations/abstractcore.md#voice-library).

- **Voice library facade**: `get_abstractcore_voice_facade(runtime)` (`AbstractCoreVoiceFacade`, `VoiceLibraryError`) lists the languages each voice engine accepts, lists cloned voices (designed and built-in ones included, optionally by language), reads, renames, replaces the reference transcript of and deletes a cloned voice, creates a voice from a recording (`clone_voice`) or from a written description (`design_voice`), and returns the capability catalog (which engine and model supports `speed`, `one_shot`, design... and how). Local runtimes run it in process; remote runtimes call the AbstractCore server's `/v1` voice routes. Refusals carry an HTTP-like `status_code`. See [Voice library](docs/integrations/abstractcore.md#voice-library).
- **Voice catalog by language**: `get_voice_catalog(..., language=)` keeps the voices that can speak a language, native speakers first (local and remote).
- **`one_shot` on remote TTS**: a remote runtime forwards `one_shot` on buffered speech, as local runtimes already did.

### Changed

- A voice request the voice plugin refuses (a plain `ValueError` or `KeyError` on a `modality: voice` output, such as
  `one_shot` over an engine's length limit or an unknown voice id) and a missing capability
  (`CapabilityUnavailableError`) fail the `LLM_CALL` effect at once instead of being retried: one engine call, the same
  message. Other media lanes keep their retries.
- A remote AbstractCore server's 4xx refusal (other than 408, 409, 425, 429) is no longer retried: the status is read from
  the HTTP error's response.

## [0.12.0] - 2026-10-27

Requires AbstractCore 2.30.0 or newer (`EmailClient` mailbox verbs; Gmail labels on message summaries) and AbstractMemory 0.3.2 or newer.

First published release since 0.10.6: it also ships the 0.11.0 and 0.11.1 sections below.

### Added

- **Mail verbs for flows**: fixed-verb nodes `mail_search`, `mail_get`, `mail_mark`, `mail_label`, `mail_move`, `mail_delete`, `mail_send`, `mail_reply`, `mail_forward` and `mail_attachment` compile to `TOOL_INVOKE` effects (`mail.*`) the runtime runs on the run's mailbox (default: the trigger's, else the primary; `mailbox` names another by id, or by the address of the run's own or the primary mailbox; `ids` / `id` take one id, one Email record or a list), without an approval prompt and never mintable by a model. Built on AbstractCore 2.28.0's `EmailClient` verbs (`search` with a typed query, `get_many`, `mark`, `label`, `move`, `delete`, `download_attachment`, `session(write=...)`) and `EmailContext.send` / `reply` / `forward` (through `guarded_send`): one connection per verb, reads read-only, mailbox changes and sends refused on a read-only mailbox (`email_mailbox_read_only`). See [Mail verbs for flows](docs/email.md#mail-verbs-for-flows).
- **Publish to channel**: `EffectType.PUBLISH_EVENT` (node type `publish_event`) with `{channel, type?, title, body?, subject?, data?, alert?}`: the runtime validates it and hands it to the host (`Runtime.set_publish_event_handler`, `handler(run, payload, step_id)` with the effect's idempotency key as `step_id`, the same on a replayed step); the host appends a CloudEvents event with the `channel` extension to the user's stream (and an attention item when `alert`) and returns `{event_id, sequence, alerted}`, the node's result. Channels are free names applications subscribe to. See [Publish to channel](docs/automations.md#publish-to-channel).
- **The Email record**: one message shape for trigger inputs, samples and the mail verbs: `{id, message_id, mailbox, folder, thread_id, from, to, cc, subject, date, body_text, body_html, attachments: [{index, name, size, type, artifact}], seen, flagged, labels}`. `labels` merges the IMAP keywords with the message's Gmail labels (AbstractCore 2.29.0). The trigger's `messages` use it: their `id` is now the message's locator (`"<uidvalidity>:<uid>:<folder>"`) and the Message-ID moves to `message_id`.

- **Publish to channel: `recipient` and `channels`** (framework ADR-0041 follow-up, Notify me's Who / Where). The `PUBLISH_EVENT` payload gains `recipient` (default `"me"`; a name, so another account can be named later; the host decides) and `channels` (a non-empty list from `tray`, `watch`, `phone`, `email`, `console`, deduplicated in that order; absent or `"all"` = every channel; refused without `alert: true`), validated by `validate_publish` / `normalize_channels` (`DELIVERY_CHANNELS`). Defaults are omitted from the payload the host gets, and `handler(run, payload, step_id)` is unchanged, so older hosts keep working. The visual executor and compiler pass both from the node's pins or, else, its `effectConfig` (`publish_config_defaults`: Notify me saves `effectConfig.channels`). Tests: tests/test_publish_event.py (validation, the host receives the set, the compiled workflow passes the saved choice, a compiled Notify me end to end).

- **RSS 1.0 (RDF) feeds** for `feed.item@1`: `parse_feed` reads `<rdf:RDF>` documents with the RSS 1.0 channel and items (namespace `http://purl.org/rss/1.0/`), stdlib only, into the same item record as RSS 2.0 and Atom: `id` = the item's `rdf:about` (else the link), `published` = Dublin Core `dc:date` (else none), `summary` = `description` (else `content:encoded`), `author` = every `dc:creator`, joined with commas; `title` falls back to `dc:title`. The parsed feed's `kind` is `rss1`. An RDF document without an RSS 1.0 channel is refused (`FeedParseError`, so `feed_unreadable` on the connection). The fetcher also accepts `application/rdf+xml`. Nature (`https://www.nature.com/nature.rss`), Science (`https://www.science.org/rss/news_current.xml`, the journal's table of contents) and Deutsche Welle's RDF feed were refused before and now read.

### Changed

- **Dependency floors**: AbstractCore 2.30.0 (the published successor of 2.27-2.29, with the latest
  AbstractVoice, AbstractVision, AbstractMusic and Abstract3D plugins) and AbstractMemory 0.3.2.

- **Mail verbs refuse in two plain ways.** A mailbox the person keeps Read only (AbstractCore's explicit `EmailAccount.access`, no longer "has SMTP") refuses `mark`, `label`, `move`, `delete`, `send`, `reply` and `forward` with `email_mailbox_read_only`: "This mailbox is read-only: automations only read it." / "In My email, turn on Read and write for <address> to change it or send from it." A Read and write mailbox without an outgoing server refuses `send`, `reply` and `forward` with the new `email_mailbox_cannot_send`: "This mailbox can't send: it has no outgoing server." / "In My email, add the outgoing server (SMTP) for <address>." Both before connecting.

### Fixed

- **A compiled Publish to channel / Notify me node runs.** The visual compiler adds `resume_to_node` (and the effect type under `type`) to every effect payload; `validate_publish` refused it ("invalid publish request: unknown publish field(s): ['resume_to_node']"), so every Publish to channel and Notify me node of a VisualFlow failed its step, and a node's own `type` pin replaced the effect type the compiler reads. The runtime now strips that envelope before validation (`core/publish.strip_visual_envelope`) and the executor carries the `type` pin as `event_type`.
- **A process that serves no model does not load one.** `MultiLocalAbstractCoreLLMClient` builds its default client at construction (a warm-up that, for a local MLX default, loads the weights: 15 GB for a 27B model) and again on a default switch. The gateway's split API process builds the same runtime as the runner process, so it loaded a second copy at boot. `set_default_model_warmup(False)` (`integrations.abstractcore.llm_client`, process-wide, no environment variable) defers the default client to its first call; `default_model_warmup()` reads it. Default on: nothing changes for a single process. Tests.
- **Events reach runs parked by another process.** In a split gateway (`serve --no-runner` + `abstractgateway runner` on one folder) the runner process parks a run in WAITING(EVENT) by re-saving a run it already saved as running, which writes no creation-log line, so the API process's event-wait index never heard of it and an event delivered there missed the waiter (same hole for a run that declared `events_mailbox` later). `JsonFileRunStore` now refreshes both indexes from the store's change signal before every lookup: when the creation log or the folder's mtime moved, or at least every 2 s (`EVENT_INDEX_REFRESH_S`), it stats the running and waiting runs and re-files the ones another process changed. Two stats when nothing moved; own saves never re-read; `event_index_stats()`. Tests: a real second process parks a run, an event delivered in this process resumes it (and the mailbox case); the 2 s floor; no pass when nothing moved.
- **Run listings come from memory (the trusted run index).** `JsonFileRunStore` answers every listing that is not RUNNING/WAITING or session-scoped (no status, a finished status, `role=`, `automation_id=`, `latest_occurrence_row`: about 61 call sites in the gateway) from its in-memory record of every run's listing fields once one complete pass has read the folder, instead of globbing and stat()ing every run file per call (11,500 files: a 0.25 s pass alone, minutes beside a busy thread). Changes reach it through the store's own writes, the creation log (new line `=<run_id>`: a finished run's file changed, written after publication), a re-read of its RUNNING/WAITING entries when the folder moved, and a background reconciliation (at most every 60 s when the folder or the log moved, at least every 10 min) that repairs and logs any difference. `trust_index=True` (default) / `False` (verify: the scan answers and is compared, `run_index_stats()["shadow_mismatches"]` counts differences); `reconcile_run_index()`; `OffloadingRunStore.run_index_stats()`. Memo hygiene on the way: a scan's read never overwrites a newer entry from this store's own save, a parsed file is memoized with the token of the file actually read (fstat), and the prune keeps entries whose file exists (created after the glob). Tests: differential (2,000 runs, two writers, after every step and after reconciliation), idle minute with zero run-file stats, second process through a store and bypassing the log, shadow mode.
- **Automation listings read the present, not the history.** `pending_waits` reads the run index's WAITING rows
  attributed to the automation (occurrences and descendants) instead of walking every occurrence tree, and the new
  `pending_waits_by_automation` answers a whole page from one read. The new `latest_occurrence(runtime, id)` (the
  row of `list_occurrences(limit=1)`) reads the controller ledger backwards from its end and stops at the newest
  admission (`automation_records_newest_first`, a lazy newest-first reader of JSONL ledgers);
  `list_occurrences` resolves prompt artifacts for the returned page only; `list_attention` reads back to the seen
  floor only. `normalize_occurrence_output(..., lazy=True)` resolves just the values it reads (an agent's output can
  hold thousands of offloaded values). `list_automations(..., index_rows=...)` takes controller rows a host already
  holds, and `automation_generation(run_store)` exposes the change counter to key them on. Measured on a copy of the
  operator's plane (11,500 run files): the gateway's automations list went from 2.4 s to 0.22 s cold.
- **Ledger change signal.** `ObservableLedgerStore.change_token()` (a counter every append moves) and
  `changed_run_ids(token)` (the runs whose ledger grew since, from the last 8,192 appends; `None` beyond), so a periodic
  reader learns which ledgers grew without re-reading them. Process-local, like the subscriptions. The bounded record
  behind both stores' `changed_run_ids` is `abstractruntime.storage.change_ring.ChangeRing`.
- **"Did any run change?" is one stat.** New `JsonFileRunStore.change_token()` (creation-log size + active-index counter,
  moved by every save, create and delete) and `changed_run_ids(token)` (the runs behind the last 8,192 changes, or
  `None` when unknown). The gateway runner compared a glob + stat of every run file every 0.5 s instead (32 ms on a
  warm 11,500-run copy, 0.18 s on the operator's folder): the token costs about 1 µs and reads no file.
- **A session's artifacts without reading the whole catalog.** New `ArtifactStore.list_session_candidates(session_id, owner_run_id=)`:
  the artifacts a session may see (owned by its memory run, tagged `session_id`, or tagged `shared`), newest first.
  `FileArtifactStore` answers from three catalog indexes (two new expression indexes over the tags, built by the first
  lookup in the caller's thread, never when the store opens): 1-7 ms on a 106k-artifact catalog, where reading the
  catalog to filter in Python took 3.2 s. A catalog error raises; there is no fallback to reading every artifact.
- **"Who listens?" no longer reads every automation's run file.** `connection_usage`, `stream_consumers` / `wake_stream_consumers`, `email_trigger_consumers` / `wake_email_automations`, `connection_trigger_consumers` / `wake_connection_automations`, `prune_email_inbox` and the feed poll used to list the whole run index and load every controller run on every call: per connection row, per feed poll, per mailbox poll and per wake (measured on the operator's 11,499-run plane: 0.5 s per call quiet, 2.2 s beside a busy thread, 6-9 s for a 3-connection listing; 190% CPU while idle). They now answer from one **active-consumers index** per runtime (`abstractruntime.automations.consumers`: `{automation_id: trigger}` for the ACTIVE automations), in microseconds. It is rebuilt only when something changed, on the run store's automation generation (`invalidate_active_consumers`, keyed by the store's folder or database file): a JSON-file store bumps it on every save whose controller index row (the automation's status and trigger) is new or different, so any path of the process (a command, a decision, a controller failing or cancelled) is seen at once and ordinary controller saves bump nothing; `create_automation`, every applied automation command and any decision or reconciliation whose delta changes the definition or the `paused` / `exhausted` state bump it explicitly for every store; a 30-second TTL catches another process writing the same folder. A rebuild lists only the RUNNING and WAITING controller rows (a JSON-file store answers from its active index) and reads no run file: a controller's index row now carries `automation_consumer: {status, trigger}` (an older scan sidecar's controller rows are re-parsed once). New `connections_usage(runtime, connections)` answers a list from one snapshot. Answers are unchanged.
- Trigger messages keep their flags: the mail watcher stores each new message with its `flags` (\\Seen, \\Flagged, IMAP keywords) and Gmail `labels` (AbstractCore 2.29's `MessageSummary.labels`, on a Gmail server) as read, so an occurrence's Email record carries `seen`, `flagged` and `labels` like a sample's (they were null). Nothing else in the payload or the cursor changes.

## [0.11.1] - 2026-10-25

Requires AbstractCore 2.27.0 or newer (`EmailClient.session()` / `get_many()`).

### Added

- **The mail watcher reads a burst of new mail on one connection.** `EmailInboxFeeder.poll` lists and reads the new messages inside one `EmailClient.session()` and reads them with `get_many` (one header FETCH per 100, one body FETCH per text section) instead of one connection per message (~10 s each on a remote server). Payloads, the cursor (advanced message by message, only after each durable append), nothing marked read and the per-message failure count are unchanged; a message that breaks the batch read is read alone so only it fails. Needs AbstractCore with `EmailClient.session` / `get_many` (the release after 2.26.4).

- **Named occurrence inputs for flows**: an automation's occurrence receives its trigger's data as flat, named inputs beside `input_data.event`: `scheduled_at` and `occurrence` (schedules), `messages` and `mailbox` (`email.received@1`; `messages` stored as an artifact like the mail bodies), `items` and `feed` (`feed.item@1`), `calls` (`webhook.received@1`); a manual run adds none. Adapters implement `occurrence_inputs(...)`. A stored webhook call now keeps its headers (lower-cased, without the signature and credentials). The input and output contract (`response`, `success`, `notify`) is documented in [Occurrence inputs and outputs](docs/automations.md#occurrence-inputs-and-outputs-the-flow-contract).

- **Sample inputs for testing a triggered flow**: `sample_occurrence_inputs(runtime, source_id, source_version, config, count=100, principal=...)` returns the named inputs and `event` a real occurrence would receive, from real data and read only: the newest matching messages of the bound mailbox (20-second fetch budget, `truncated`), the latest feed items, the last stored webhook calls (or one documented sample call), `scheduled_at` = now for schedules. The event carries the CloudEvents extension `abstracttest: true`. Mail is read on one IMAP connection (`batched_newest_messages`: AbstractCore's `EmailClient.session()`, `search` and `get_many`; a body over 1 MB is skipped with a typed record, never cut; attachments never downloaded; nothing marked read), so 100 messages take a few round trips. See [Testing a triggered flow](docs/automations.md#testing-a-triggered-flow-with-sample-inputs).

### Changed

- **Feed connections are polled only while something listens.** `TriggerConnection` gains `paused` (default false) and `always_listen` (default false), set through `TriggerConnectionStore.create(...)` / `update(...)`; `status` keeps its meaning. `connection_usage(runtime, connection)` derives `{listening: "paused" | "listening" | "idle", used_by: <active automations reading the stream>}`, and `connection.to_dict(usage=...)` adds both fields. `poll_feed_connections` fetches only listening feed connections (an idle one is not fetched and stores nothing; its first poll after an automation binds again is a baseline) and its reports carry `listening` and `used_by`. `append_webhook_event` refuses a paused connection like a disabled one (409 `connection_disabled`); an idle webhook connection still accepts and stores calls (its `runtime=` argument is accepted and unused). See [Trigger connections](docs/automations.md#trigger-connections).
- `feed.item@1`'s connection fields read in plain language for forms: "Feed address", "Check every" (a number and a unit), "Most new items per check" and "When added" ("Ignore what is already there" / "Also deliver the latest items", `enumNames`); the field names and values are unchanged.

## [0.11.0] - 2026-10-25

### Added

- **`email.received@1` on a further mailbox** (framework backlog 1044): `config.account` accepts a mailbox id of the user (`"self"`, the default, and `"default"` stay the primary mailbox). The automation reads only that mailbox's inbox streams and its occurrences act for it (`_runtime.email_account` from `email_binding_for_trigger`); `message_matches` checks the mail's `account_ref` too. The descriptor's `account` schema is a pattern with default `"self"` (no longer `enum: ["self"]`). See [The email.received@1 trigger](docs/email.md#the-emailreceived1-trigger).

- **A note wake reaches the person** (framework backlog 1045 follow-up): `SCHEDULE_RUN` takes `notify` (default true for a `note`, false for a `task`); a notifying wake's definition carries `notify.always: true` with `notify.title` (the note) and `notify.body` (the note), so its completed occurrence raises an attention item (kind `notify`, body = the answer, else the note) whatever the answer's own `notify` says. `notify.always` / `title` / `body` are new optional fields of every definition's `notify`. The effect result reports `notify`.

- **Wakes: `EffectType.SCHEDULE_RUN`** (framework backlog 1045): a run schedules work for later (`at`, `after_s` or `on` a trigger event) with a `task` or a reminder `note`, as a new turn of its own session by default (`mode: same_session`) or in a fresh one. The runtime plans a one-shot automation (`automations.wakes.plan_schedule_run`): tools at most the caller's ceiling intersected with the request, never `schedule_run` itself, written as the occurrence's explicit `_runtime.allowed_tools`; approval `ask`, the caller's tool policy copied, untrusted callers make untrusted wakes; `origin_run_id` recorded and the occurrence's origin is `agent`. The host creates it through `Runtime.set_schedule_run_handler` (usually with `create_scheduled_run`) and appends `wake.scheduled` / `wake.cancelled` to the user's stream (`wake_scheduled_event`, `wake_cancelled_event`). Creation requests accept `session_id` (where a growing automation's occurrences run; the controller keeps `automation:<id>`), `origin_run_id`, `one_shot` and `untrusted_origin`; an occurrence's `session_kind` may be `chat`. `schedule_run` is classified as a delegation tool. See [Wakes](docs/automations.md#wakes-schedule_run).

- Several mailboxes per user (framework backlog 1044): `email_binding_for_trigger(binding, trigger)` binds an `email.received@1` automation whose `config.account` names a further mailbox of the same user to that mailbox (`<tenant>:<user>:<account_id>`; `self`, `default` or no value keep the runtime's own binding), with `account_ref_with_id` / `account_id_of_ref` (`abstractruntime.email`).

- **Webhook trigger `webhook.received@1`** (framework backlog 1042, runtime half): a webhook connection with a generated, sealed signing secret (`whsec_…`) or the creator's own, an optional `subject_path` and a replay window. `verify_standard_webhook(secret, headers, body, now, tolerance_s)` checks Standard Webhooks signatures (pure; reference vector tested), and `append_webhook_event(inbox, connection, body, headers)` maps one verified HTTP call to one event on the connection's stream (`webhook-id` as `subject`/`dedupekey`, so a replayed id is refused) and returns the status the host's route answers with (202 / 409 / 401 / 413 / 415). See [`webhook.received@1`](docs/automations.md#webhookreceived1).

- **Feed trigger `feed.item@1`** (framework backlog 1043): runs an automation on new items of an RSS 2.0 or Atom feed connection (`{url, every >= 60s, max_items, initial_items: skip|deliver}`). Standard-library parsing, conditional GET, item identity (Atom id / RSS guid / link) as `subject` and `dedupekey` so each item is appended once, the cursor in the connection stream's state, typed failures on the connection with a capped backoff. The host's plane runs the poller: `poll_feed_connections(runtime, store)` polls the due connections and wakes their automations (no thread started here). The items of one poll become one occurrence, with the items in `input_data.event.data.events` and, for a string prompt, in a fixed untrusted frame. A `network` pytest marker for the live test. See [`feed.item@1`](docs/automations.md#feeditem1).

- **CloudEvents trigger envelope** (framework backlog 1039): `TriggerEnvelope` carries the CloudEvents 1.0 attributes (`specversion`, `id`, `type` = `abstract.<source>.v<version>`, `source`, `time`, `subject`, `datacontenttype`) and extensions (`bindingid`, `connectionid`, `dedupekey`, `sequence`, `recordedtime`, `abstracttrust`; `correlationid`, `expirytime`, `traceparent` reserved) beside its original keys, which stay for this release. Every built-in source populates `subject`, `dedupekey` and `abstracttrust`; schedules add `payload.series = {rule, occurrence, scheduled_at, next_at, coalesced}`. Occurrences receive the envelope as `input_data.event` (`to_cloudevent`; `from_cloudevent` is the inverse; `CLOUDEVENT_JSON_SCHEMA`). The prompt header is unchanged. See [Occurrences](docs/automations.md#occurrences).

- **Run origin** (framework backlog 1038): `vars._meta.origin = {kind: person|trigger|agent|system, actor, client, ref}` records who or what started a run. `abstractruntime.core.run_origin.origin_for(...)` builds it for hosts (person and system runs), `stamp_origin` writes it once and `origin_of` reads it (None for older runs). Every automation occurrence now carries one from the controller: `trigger` with `ref {automation_id, occurrence_index, event_id, source}`, or `person` for a manual run now; the actor is the automation's owner (`_meta.automation_owner`, the `"<tenant>:<user>"` of the creation request). See [Run origin](docs/automations.md#run-origin).

- `wake_stream_consumers(runtime, stream)` / `stream_consumers(runtime, stream)` (`abstractruntime.email`, framework backlog 1042/1044): wake the automations that read ONE inbox stream, decided by their binding alone (`automation_reads_stream`): `trigger:<connection_id>` for a connection-bound binding, `email:<tenant>:<user>:<account_id>:<folder>` for an `email.received@1` binding of that mailbox (`config.account`; `self` or unset is the runtime's bound mailbox). `wake_email_automations(runtime, account_id=…)` and `email_trigger_consumers(runtime, account_id=…)` take an optional mailbox; `prune_email_inbox` also keeps the events a connection-bound automation has not read yet.

- `MailboxQueryableRunStore` (backlog 1046): `list_mailbox_runs(names=[…])` returns every running or waiting run
  that declares one of the names as `events_mailbox`, and `next_wait_deadline()` the earliest wait deadline over
  every waiting run. The in-memory, file (a lazily built mailbox index maintained on save and delete, every hit
  re-verified on disk) and SQLite stores implement both; `OffloadingRunStore` forwards them. They let a host fan
  durable events out and schedule wakes without a window of the newest runs.

- **Trigger connections** (framework backlog 1040): per-user instances of a trigger source, registered once and reused by any automation. `TriggerConnectionStore(plane_dir)` keeps one JSON file per connection (`<plane>/triggers/connections/<id>.json`, atomic writes), listed and changed per principal (`"<tenant>:<user>"`; another principal's connection reads like an unknown one), with secret fields and generated secrets sealed by AbstractCore's `SecretVault` (the mailbox credentials' sealing). Descriptors gain `needs_connection`, `connection_schema` and `connection_provider` (`schedule@1/2`, `manual@1`: none; `email.received@1`: a mailbox of the mail accounts). Each connection has one inbox stream `trigger:<connection_id>`; `test_fire` appends a synthetic `{data: {test: true}}` event to it and `wake_connection_automations` wakes the bound automations. `create_automation` and `automation.revise` refuse a binding whose `config.connection_id` is unknown, another principal's, of another source or disabled (`Runtime.set_trigger_connection_store`). `ConnectionEventAdapter` is the shared adapter of connection-bound sources (one coalesced admission per batch of stream events). See [Trigger connections](docs/automations.md#trigger-connections).

## [0.10.6] - 2026-10-18

### Added

- `abstractruntime.conversation_search.ConversationSearch`: keyword search over a plane's conversations: each turn's question and answer (as the session replay builds them) and its media, the artifacts it generated (kind and generation prompt, attributed to the turn whose run made them) and the attachments it was sent with (kind and file name); read-aloud audio and internal artifacts are not media. A hit on an artifact names it (`artifact`: id, owning run, kind, content type), so a client can show it. Its own SQLite file per plane, an FTS5 `trigram` index over text lowercased and without accents: words and parts of words, every word required, words of one or two letters as whole words, and a fuzzy pass that keeps a turn holding at least 60% of each word's trigrams (a typo still finds it). One hit per session (its best turn, a snippet, the match positions, where it matched: question, artifact or answer). Tool usage, tool outputs and intermediate thinking are not indexed: [backlog 0849](docs/backlog/proposed/0849_conversation_search_tool_usage_outputs_and_thinking.md). `sync(budget_s=…)` indexes what changed since the last call, newest sessions first, and reports what is left. Works with the file and the SQLite run stores; without the `trigram` tokenizer (SQLite < 3.34) it indexes whole words with prefixes.

- Automation definitions accept a named `policy.tool_policy` snapshot for per-tool Auto and Approve choices. The target tool ceiling remains authoritative; unspecified tools ask for approval. Messaging and untrusted-input restrictions remain enforced. See [Tool approval](docs/automations.md#tool-approval).
- Streamed read-aloud recordings (`_finalize_tts_stream_artifact`, the streamed-voice path of the AbstractCore integration) are stored under the voice namespace: their artifact tags carry `purpose=voice_playback`, `artifact_space=voice`, a `filename` and a logical `path` of `voice/tts-<id>.wav`, so a host can list a user's generated documents and attachments without the playback audio while playback and replay keep the same artifact ref and provenance. Test: `tests/test_streamed_voice_namespace.py`.

## [0.10.5] - 2026-10-18

### Changed

- The pooled client wires every client it builds in one place (`_adopt_client`): the residency core, the
  Core execution context, the host's endpoint-profile resolver and the pool-wide token callback reach the
  text clients, the per-call override clients and the media host through the same path, and a setting
  applied later reaches all of them through `_owned_clients`. No behaviour change. Tests:
  `tests/test_endpoint_profile_media_end_to_end.py` (an image spec on an `endpoint:<id>` profile reaches
  that endpoint through the `LLM_CALL` handler, the pooled client and AbstractCore's capability host, with
  and without a text provider on the call).

## [0.10.4] - 2026-10-18

### Fixed

- An image, video or voice spec whose provider is an endpoint profile (`endpoint:<id>`) keeps that provider
  all the way to the media plugin. The `LLM_CALL` handler resolved the first output spec's endpoint as if it
  were the call's text provider: it rewrote the spec to the bare provider family and put the endpoint's URL
  and key into text-client routing params that the media path never reads, so an image generation routed to
  a gateway endpoint profile (the web console's sandbox, an image route set to an OpenAI-compatible relay
  serving `gpt-image-2`) reached AbstractVision as a keyless `openai-compatible` request with no base URL
  and failed with "Missing vision_base_url / OPENAI_BASE_URL" -- in the one-shot media subprocess, since
  the spec no longer said `endpoint:`. Only the call's own `provider` (`_provider`) is resolved now; the
  spec's endpoint is resolved by the media plugin through the host's resolver, in process. The pooled
  client's media host (the client every media-only call runs on) now carries the host's endpoint-profile
  resolver, whether it is built before or after `set_provider_endpoint_profile_resolver`. Affects every
  host that attaches an endpoint-profile resolver (AbstractGateway). Tests:
  `tests/test_endpoint_profile_media_specs_keep_provider.py`.

## [0.10.3] - 2026-10-18

### Fixed

- Image and video specs whose provider is an endpoint profile (`endpoint:<id>`) run in process instead of in
  the one-shot local media subprocess. The subprocess isolates local pipelines and has no host
  endpoint-profile resolver, so a gateway profile (for example an OpenAI-compatible relay serving
  `gpt-image-2` selected on the gateway's image route) could not be resolved there; in process,
  AbstractVision resolves it through the resolver the host attached to the client. Tests:
  `tests/test_endpoint_profile_media_in_process.py`.

## [0.10.2] - 2026-10-16

### Changed

- **Code nodes can import Python's standard library.** In the `sandbox` permission mode, `import json`, `re`,
  `datetime`, `math`, `collections`, `itertools`, `statistics`, `textwrap`, `hashlib`, `urllib.parse`… now
  work, at the top of the code and inside functions, in both the RestrictedPython lane and the fallback lane, and
  inside the function bodies of flow function libraries. The sandbox still refuses third-party packages, private
  (`_`-prefixed) modules, and the standard-library modules that reach the machine or go around the sandbox's own
  checks: processes and the operating system (`os`, `subprocess`, `threading`, `signal`, `ctypes`, `platform`…), files (`io`,
  `pathlib`, `shutil`, `tempfile`, `glob`, `sqlite3`, `logging`, and the parsers that read a file by path: `xml`,
  `configparser`, `argparse`…), the network (`socket`, `http`, `urllib.request`, `smtplib`…), code loading or
  interpreter internals (`sys`, `importlib`, `pickle`, `types`, `inspect`…) and attribute access by name,
  evaluated annotations or frames (`operator`, `typing`, `functools`, `traceback`). A refusal names the module
  and the reason; `from x import *`, relative imports and the `__import__` call stay refused.
- **A Code node's return value must be JSON data.** `transform` returning a module, a function, a set, bytes or
  any other value JSON cannot carry now fails with a sandbox error naming the value and where it sits
  (`transform returned a value that cannot be sent as JSON: module 'time' at output[2]`), in every permission
  mode, instead of a serializer crash later in the gateway's simulate reply or the run's persisted state. `full_access` (host policy) imports anything, as before.
  The rule is applied twice: at compile time by `validate_code`, and at run time by the `__import__` bound into
  the sandbox builtins. `get_code_execution_policy()` now lists `imports.denied_modules` for thin clients.

## [0.10.1] - 2026-10-16

### Changed

- **The JSON run store answers "which runs are running or waiting" from an in-memory index.** Every such listing (`list_runs` / `list_run_index` with status running or waiting) used to glob and `stat()` every run file in the store on every call — 10,678 files on a long-lived store — and each `stat` releases the GIL, so the scan queued behind any busy thread: 0.2 s alone, 3 s with two concurrent, 67–147 s beside a generating model. The store now keeps the set of active run ids, built from one scan, maintained by `save()`, `delete()` and the creation journal, and re-verifies every candidate against disk at listing time, so a run that ended or vanished is never listed and a stale entry heals itself; a background rescan at most every 60 s catches a writer that bypasses the journal. The rows, their order, filters, limits and the `updated_at` sort are exactly the scan's — proven by a differential harness against a copy of a live store under concurrent writers and a busy thread, and by `tests/test_active_index.py`. `JsonFileRunStore(..., use_active_index=False)` keeps the plain scan.
- The creation journal gains a `~<run_id>` line: a run that becomes active through one store object (resumed, re-opened, first save after a restart) is announced to every other store object on the same folder, like `+` for creation and `-` for deletion. Older readers ignore the line.

## [0.10.0] - 2026-10-15

### Added

- Trigger source `schedule@2`: `{kind: "daily", at: "HH:MM"}`, `{kind: "weekly", days: [...], at}` and `{kind: "monthly", day: 1..31 | "last", at}` evaluated on wall time in the config's `time_zone` (IANA, converted with `zoneinfo`: a skipped wall time runs at the shifted instant, a repeated one runs once; a day beyond the month's length runs on its last day), plus `{kind: "every"}` and `{kind: "once"}` with `schedule@1`'s meaning (a `once` may give a wall time `at` in `time_zone`). Same coalescing and pause policy as `schedule@1`; event ids `schedule@2:<binding_id>:<tick>`. `validate_time_zone(name)` and `time_zone_names()` are exported from `abstractruntime.triggers`. See `docs/automations.md`.
- `integrations.abstractcore.config_facade.voice_input_hint(route, host=None)`: AbstractCore's served one-line hint for a configured speech-input route (`abstractcore.config.recommendations.voice_input_hint`, AbstractCore >= 2.26.0), so a host such as AbstractGateway serves it without importing AbstractCore itself. The AbstractCore floor is 2.26.0.
- A limit on an execution wire: `edge.data.loopLimit = {"max": N}` (the editor's ↺N badge on a wire that makes execution repeat). Before compile, `visual/loop_limit_lowering.py` turns it into a `loop_limit` node on that wire (`again` to the wire's target; `limit_reached` to the other branch when the wire leaves an If/Else and that branch is not itself a limited wire, else nothing: the path ends at the limit), re-keying the target's per-path values and entry list. A run tree shows the lowered node as "Loop limit N". `VisualEdge` now carries `data` (older runtimes never read it, so a limited flow runs unlimited there).
- VisualFlow `get_vars` node (Get Variables): a data node with one output per variable, its pin id the variable name (dotted paths allowed); each output reads the run variable's current value, or null when it is not set.
- VisualFlow `loop_limit` node (Loop Limit): exec outputs `again` (taken while fewer than `max` passes were let through) and `limit_reached` (then the count starts again from 0), data output `count` (the pass number). The count lives in `run.vars._temp.loop_limits.<node id>`, so it survives pause/resume. It belongs to the current iteration of the enclosing ForEach / For / While (each item, and each new entry of that loop, starts from 0); with no enclosing loop, to the run. Its `max` comes only from its own pin, never from the previous step's output.
- VisualFlow `add_to_list` node (Add to List Variable): `name`, `item` (added as one element), `items` (each element added); a missing or null variable starts as `[]`; a variable that holds something else, or `items` that is not a list, fails the step naming it; reads and writes in one step; output `list`.
- String Template: a placeholder reads an input pin of its name, then the `vars` object, then the run variable (top-level names not starting with `_`). A list or object with no filter is written as JSON (it was Python's `str()`).

### Changed

- An execution wire that loops back out of a ForEach / For / While body (from inside the body to the loop node itself or to anything before it) is refused when the flow compiles: "Wire <source>.<pin> -> <target> goes back out of <loop>'s loop body: the run would never end. A path inside a loop body can only repeat within the body; to repeat the whole loop, wire the loop-back after the loop's done." Such a wire re-entered the loop with its frame still on the control stack, so the stack grew by one frame per pass and the run never ended (a Loop Limit on it never reached its max). A loop-back that goes through the loop's `done`, or that stays inside the body, is unchanged (`visual/loop_back_check.py`).
- `schedule@1` is unchanged: its configs normalize byte for byte as before and existing automations keep their tick grid (`tests/test_schedule_v1_corpus.py` freezes the 0.9.1 answers on real automation configs).

### Fixed

- Set Variables (and Add to List Variable) only take what their own pins carry (a wire, a value set on the node, an expression). Before, keys of the previous step's output were written as variables too.
- Set Variables (`set_vars`): every data input pin besides `updates` is one variable, written under the pin's id (dotted paths nest) with its value: a wire, else the pin's default. These win over the same name in the `updates` object. AbstractFlow edits them as rows (name, type, value), so one node initialises several variables, wired or literal. A flow using such rows needs this runtime: an older one writes only `updates`. Test: `tests/test_visual_set_vars_pins.py`.
- VisualFlow reroute ("relay") nodes: a dot in AbstractFlow that lets one output feed several inputs along a tidy path. They are visual only: before compiling, every chain `source -> reroute (-> reroute ...) -> target` is rewired `source -> target` and the reroutes are removed (`visualflow_compiler/visual/reroute_lowering.py`, run before the multi-entry lowering). A reroute with no source feeds nothing (its targets keep their defaults); per-path values that name a reroute point at its real source. Data wires only: an execution wire through a reroute, a second input into one, or a cycle of reroutes is refused with a message naming the wire. A flow with a connected reroute needs this runtime (older ones refuse the unknown node type). Test: `tests/test_visual_reroute_lowering.py`.

## [0.9.1] - 2026-10-08

### Security

- `browser_probe`: a local target (path or `file://` URL) now carries the run's `_sandbox` stamp, the same scope the commands get. AbstractCore 2.25.1 serves the page from a loopback origin limited to that scope instead of `file://`, so a page cannot load refused files as subresources. Needs AbstractCore 2.25.1 (an older `browser_probe` refuses the unknown argument and runs nothing). Test: `tests/test_browser_probe_target_walling.py`.

### Fixed

- `stream_voice(...)` is never a wait. It used to start the child run parked on `WAIT_EVENT abstractcore.voice.tts.stream:<uuid>` for the whole stream: hosts folded that waiting record into the CALLER'S run (a Read aloud during a live turn read as "Waiting for an event"), and a host that died mid-stream left the child waiting forever. Now nothing durable exists while audio streams; when the stream ends (done, error, cancel or the consumer closing it) the child run is created already COMPLETED with the outcome. Its id is allocated up front: `runtime_start` carries it as `child_run_id` (schema `abstractruntime.tts_stream.start.v2`, no `wait_key`), as does the trace metadata. A stream cut by a crash leaves no run.
- `AbstractCoreRunFacade.close_interrupted_voice_stream(child_run_id, *, reason)` finishes a legacy streamed-speech wait left by an older process (completed, `errors[0] = {"code": "interrupted", "message": reason}`); `is_interrupted_voice_stream_wait(waiting)` and `LEGACY_VOICE_STREAM_WAIT_KEY_PREFIX` recognise one (exported from `integrations.abstractcore`). AbstractGateway calls it at startup.

### Changed

- Requires AbstractCore 2.25.1 or newer (`abstractcore>=2.25.1` in the base install and the `[apple]`/`[gpu]` extras): the Linux command sandbox reaches an allowed folder inside a refused one, and `browser_probe` accepts the run's `_sandbox` stamp.

## [0.9.0] - 2026-10-05

### Added

- `Runtime.set_occurrence_input_resolver(fn)`: a host hook called at each automation occurrence's admission, as `fn(definition, input_data) -> input_data`. Its answer is frozen into the occurrence's prepared inputs, so replays stay byte-identical; a non-dict answer fails the admission; with no resolver set nothing changes. AbstractGateway uses it to resolve the workspaces of an automation that follows its owner's default at each run, instead of a snapshot taken at save time. Test: `tests/test_automation_occurrence_input_resolver.py`.

- Command sandbox: `execute_command`, `shell_exec`, `local_helper_start` and AbstractAgent's `execute_python` (0.3.18 or newer; an older one fails on the unknown argument and runs nothing) receive the run's effective workspace set as the hidden `_sandbox` argument (`sandbox_stamp(scope)`, built on every call from the same workspace vars the file tools read; `execute_python` starts in the stamp's private workspace; a model- or payload-supplied `_sandbox` is always replaced or dropped), and run inside AbstractCore's OS sandbox (`abstractcore.tools.sandbox`: macOS `sandbox-exec`, Linux `bwrap` or Landlock). `local_helper_start` and the entity exec tool apply it themselves (the entity's workspace and whitelisted mounts). With no sandbox on the host, or an AbstractCore without the sandbox module, the call is refused with one sentence before anything runs and the run continues.

- `integrations.abstractcore.command_sandbox_host`: hosts configure the command sandbox through the runtime (`configure_host`, `host_policy`, `host_sandbox_kind`, `KIND_LABELS`, `KIND_NONE`, `reset_host_for_tests`; pure re-exports of `abstractcore.tools.sandbox`), so a host never imports AbstractCore directly. Used by AbstractGateway's `serve`.
- `MappingToolExecutor` never drops the `_sandbox` stamp: a tool whose signature cannot take it (an AbstractAgent older than 0.3.18) gets a failed result naming the stamp and does not run. Its argument-name normalization never maps a model key (`sandbox`, `registry_namespace`, ...) onto a hidden `_`-prefixed parameter.
- CI: a `linux-sandbox` job installs bubblewrap on Ubuntu and runs the command-sandbox tests end to end inside it (the end-to-end tests now run inside the host's OS sandbox: macOS `sandbox-exec` or Linux bubblewrap); the job fails when any of them is skipped.
- `workspace_writable_paths`: writable exceptions inside read-only roots (`workspace_read_only_paths`), the more specific rule winning. The tool scope, `path_is_read_only` and VisualFlow writers honour them; child runs and file nodes inherit the parent's exactly and never add one. Used by AbstractGateway 0.13.0 for "Allow everything, refuse listed workspaces" with a read-only default.
- `integrations.abstractcore.structured_facade` (also flat from `integrations.abstractcore`): `parse_response_format` and AbstractCore's own `ResponseFormatError`, so hosts check an OpenAI `response_format` without importing AbstractCore. Used by AbstractGateway 0.13.0's `/v1/chat/completions`.
- Workspace context of a tool-using `LLM_CALL`: after the default working directory (the run's own workspace), `Allowed workspaces:` lists each allowed workspace outside it with its mode (`(read-only)` / `(read & write)`, from `workspace_read_only_paths` and `workspace_writable_paths`), and under `all_except_ignored` `Everything else: (…)`; refused paths are never listed as allowed. Used by AbstractGateway 0.13.0 on every run start.
- `automation.unarchive`: an archived automation comes back paused with its history intact; a controller that ended because it was archived restarts and parks on its wake wait (observation `automation.unarchived`).

### Changed

- Workspace nesting: the most specific row wins. Among `workspace_root`, `workspace_allowed_paths` and `workspace_ignored_paths`, the longest real-path prefix of a path decides (a refusal wins a tie); a refused folder no longer refuses an allowed workspace inside it, and a refused folder inside an allowed one still refuses its subtree. Built-in protection stays absolute. Same rule as AbstractGateway's validation and the sandbox profile.
- The workspace context now says "Shell commands run inside an OS sandbox limited to these workspaces." and describes refused paths as "Excluded paths (a more specific allowed workspace inside one stays reachable)".
- Requires AbstractCore 2.25.0 or newer (`abstractcore>=2.25.0` in the base install and the `[apple]`/`[gpu]` extras): the command sandbox and `structured_facade` use it.

### Removed

- `workspace_shared_path` / `SHARED_WORKSPACE_KEY` / `shared_workspace_path` (added during 0.9.0 development, never released): AbstractGateway 0.13.0 has no shared workspace any more (round 11), so the workspace context names no "Shared workspace"; the allowed workspaces and their modes are listed as before. A stale key in a run's vars is ignored.

## [0.8.5] - 2026-10-03

- Hosts can serve AbstractCore ASGI requests through `integrations.abstractcore.server_facade.serve_core_request`, with independent request-scoped authentication preserved throughout streaming. Requires AbstractCore 2.24.0 or newer.

## [0.8.4] - 2026-10-02

- Automations accept `context.growing.max_tokens` at creation and revision, defaulting to 50,000 tokens. Whole-turn history and admitted retry inputs are preserved.
- Email-enabled automations produce a result notification for every completed occurrence, with the full answer and configurable `notify.recipients`.

## [0.8.3] - 2026-10-01


### Added

- `integrations.abstractcore.mcp_facade`: opens a stdio or HTTP MCP client (a stdio server starts
  with exactly the environment given), turns a server's tool into an `mcp::<server>::<tool>` spec
  without its URL, and runs `tools/call` with the runtime's result mapping. The gateway calls
  registered MCP servers through it.

### Changed

- An automation run started by an email from someone else (untrusted input) is never offered an MCP
  tool: every `mcp::...` name leaves its tool lists and is never granted, even when named.

## [0.8.2] - 2026-10-01

Pausing an automation also stops its retries. Requires AbstractCore 2.20.2 (unchanged).

### Changed
- `automation.pause` also stops retries: an occurrence waiting in retry backoff is cancelled
  (quietly, like `stop_current`), and a scheduled attempt that fails while the automation is paused
  completes `failed` without a retry. An attempt already running still finishes, and a manual run
  (`run_now`, allowed while paused) keeps its retries.

### Notes
- A retry that is waiting in backoff when you pause an automation is cancelled instead of starting
  after its backoff. Resume the automation, or use `run_now`, to run it again.

## [0.8.1] - 2026-09-30

Email automations never trigger themselves, speech and images work without a buildable text model,
and a chat's mail to its owner runs without asking. Requires AbstractCore 2.20.2.

### Added
- Mail sent by an automation run (controller, occurrence, any run it starts, the send-email action)
  is automatic mail: the run's `EmailContext` gets `automation_marker =
  "automation:<automation id>/run:<run id>"`, so every send carries `Auto-Submitted` and
  `X-AbstractFramework-Automation`. `automation_marker_for(vars, run_id)` is exported from
  `abstractruntime.email.binding`. Chats and discussions send ordinary mail.
- `email.received@1` config `auto_submitted`: `"skip"` (default) never admits a message whose
  `Auto-Submitted` header is present and not `no`; `"admit"` lets it reach the filters. Stored
  configs without the key behave like `"skip"`.
- `EmailInboxFeeder(..., is_own_sent=callable)` and `PollReport.own_automatic`: the account's own
  automatic mail (the framework marker from its own address, or a Message-ID the host recorded as
  sent) never enters the event inbox.
- `abstractruntime.integrations.abstractcore.run_facade.inline_run_active(run_id)`: true while the
  run facade executes a child run in this process; host runners skip such runs.
- `ToolApprovalPolicy.default_require_approval_tools`: the names on the require list only because
  no list was given (the default caution list).

### Changed
- The pooled local client runs media-only requests (image, video, voice, music, transcription, no
  tools) and streaming TTS on AbstractCore's capability host instead of building the default text
  client: text-to-speech no longer fails when the text default cannot be built (for example an
  MLX route on an install without MLX), and no text model is loaded for a media request.
- Without a per-run tool policy, the `send_email_recipient@v2` refiner now also lowers a name that
  is on the executor's default caution list (`send_email` under `ToolApprovalPolicy()`): a chat
  run's send to its owner's own address runs without an approval wait. An explicitly passed
  require list and a per-run policy's still win; email-triggered runs under `approval: "ask"` are
  unchanged.

### Dependencies
- `abstractcore>=2.20.2` (also `[apple]` and `[gpu]`).

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
  and, for a string `prompt`, inside a fixed untrusted frame whose markers carry a boundary token drawn at random
  for each occurrence (a body cannot fake the end of its email). The messages and the framed prompt are stored as
  artifacts at admission: `automation.admitted`, `pending_occurrence` and the dispatch effect carry refs, metadata and
  event ids, never bodies, and the occurrence run resolves the refs when it starts. Creating this trigger on a
  runtime without an artifact store is refused (`unsupported_feature`).
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
- **`policy.untrusted_input_tools`**: tools named one by one that an email-triggered occurrence may run without asking
  although "allow all tools" does not cover them there; message-sending tools are never granted this way.
- **`START_SUBWORKFLOW` `payload.resolve_vars`**: top-level vars whose artifact refs are resolved for the child only
  (strict: a missing artifact fails the start), so the parent's ledger and node traces keep the refs.
- **Tool facts for untrusted input** (`tool_effects`): `TOOL_NETWORK_REACH` (`none`, `configured`, `open`, `send`) and
  `TOOL_WRITE_SCOPE` (`workspace`, `workspace_only_mode`, `host`, and for the memory tools `run` and `memory`) for
  every tool the runtime exposes, with `untrusted_input_grantable(...)`.
- `list_tool_catalog(email_off_reason=...)` names why email is off (`not_connected`, `admin_disabled`,
  `not_available` — the administrator has not made agent email tools available to this user — and `agent_tools_off`).

### Changed

- **Requires AbstractCore 2.20.0 or newer** (`abstractcore.comms.email`, the email tools on a per-run resolver).
- **Email-triggered automations act only within the user's mission** (operator decision 2026-09-30). Under
  "allow all tools", an automation triggered by `email.received@1` grants by kind, decided on tool facts and never on
  names: only tools with no network egress beyond configured services, no code or command execution, no message
  sending, no writes outside the run's workspace and no delegation run unattended (file reads, workspace-confined
  file writes, mailbox reads, `get_email_attachment`, `recall_memory` and `update_plan`). `fetch_url`,
  `browser_probe`, `skim_url`, `skim_websearch`, `web_search`, `execute_command`, `shell_exec`, `execute_python`,
  `delegate_agent`, `channel_fs_write`, `agora_post_message`, `agora_send_dm`, the camera tools, the memory-writing
  tools (`remember`, `remember_note` including `scope: "global"`, `compact_memory`: they write the user's lasting
  memory, outside the run's workspace, so a stranger's email cannot plant a note later runs obey), MCP tools and
  every other tool ask, unless the user named them one by one in `policy.untrusted_input_tools`; message-sending
  tools (`send_email`, `reply_email`, `agora_post_message`, `agora_send_dm`, WhatsApp and Telegram sends) are never
  granted, named or not. Schedule and manual automations keep the full grant. `reply_email` is withheld like
  `send_email` everywhere.
- **Email-triggered automations under "ask" run nothing unasked** (re-gate 2026-09-30). Their occurrences always carry
  a per-run tool policy: under `policy.tool_approval: "ask"` it auto-approves only the tools the user named in
  `policy.untrusted_input_tools` (sending tools never) and requires approval for every other tool, so neither the
  executor's static defaults (which auto-run `skim_url`, `web_search`, the agora posts and the Telegram sends) nor a
  per-call refiner decide there. The policy replaces any `tool_policy` the target inputs carried
  (`replaced_target_policy: true`) and is marked `untrusted_input: true`, beside the run flag
  `_runtime.untrusted_input`. Child runs inherit both, the parent's value winning over the child's own. Defence in
  depth: for a run with that flag the approval executor never consults its static policy or a tier ceiling, runs a
  listed tool unasked only when the user named it or (under "auto") its facts pass `untrusted_input_allow_all`,
  never runs a sending tool unasked, and applies the per-call refiners only under "auto".
- `send_email` / `reply_email` `attachments` and `get_email_attachment`'s `output_dir` are confined to the run's
  workspace in every workspace access mode (ignored paths and the host's protection still apply);
  `get_email_attachment` is a `write` tool.
- A run policy that only withholds tools (`tool_policy.withheld_tools`) is applied like any other run policy, so the
  per-call refiners run for it.
- The per-call refiners (`send_email_recipient@v2`) also run when a run carries no `_runtime.tool_policy`, on top of
  the executor's static policy (its `require_approval_tools` still wins), and decide per call: one `send_email` that
  needs a person makes the batch wait.
- No environment variable enables the email tools: `ABSTRACT_ENABLE_EMAIL_TOOLS` is gone and
  `ABSTRACT_ENABLE_COMMS_TOOLS` enables WhatsApp and Telegram only; pass `email_enabled=True`.
- The untrusted frame tells the agent not to follow links or instructions contained in the emails and to act only
  on the automation's mission.
- After a folder rebuild with nothing to resynchronise, the feeder stores the new baseline AbstractCore returns
  (`reset` and `baseline`) instead of fetching one itself.
- Large structured tool outputs (for example a `read_email` with a big HTML body) are offloaded to session
  attachments under the same inline limit as text outputs; the result carries artifact references.
- The process-local email helpers (`list_email_accounts`, `list_emails`, `read_email`, `send_email`) are no longer
  exported from `abstractruntime.integrations.abstractcore`; they remain on the host facade for single-user installs.

### Removed

- `emails.config.example.yaml` (environment-variable email setup) is no longer shipped; email is configured through
  the host's per-user settings.

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
