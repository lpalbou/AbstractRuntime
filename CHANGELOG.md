# Changelog

All notable changes to AbstractRuntime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Visit-as-durable-run workflow (`identity/visit_workflow.py` — plan items
  7-10 v0, the phase-3 centerpiece made executable): `build_visit_workflow`
  maps the entity turn loop onto runtime effects over a per-entity runtime
  — OPEN (prelude rendered ONCE into run vars: the stable head; refused
  prelude completes with reasons, identity never truncated) → PARK
  (WAIT_EVENT `visitor_input` + D3 idle deadline, self-describing
  `details.kind`) → ROUTE → RECALL → REASON (v0: one LLM_CALL; the
  abstractagent ReAct adapter replaces exactly this node) → ELECT (one
  DIARY_WRITE per diary election, per-projection `reflected_in` edges) →
  COMMIT (same-trace MEMORY_ACCESS) → FORM (episode + lossless verbatim,
  stamped participants, `continues` chain) → ANSWER (ANSWER_USER) → park;
  close/timeout → REFLECT → staged APPLY (summary → interests → diary →
  feelings via MEMORY_APPRAISE, entity-reflection actor) → DONE. Replay
  guards: turn-id folds (history/sheet fold once), idempotent effects.
  4 tests pin the lifecycle, the RESTART-MID-VISIT resume (the phase's
  payoff: process dies between turns, a fresh runtime over the same home
  continues the visit with history intact), idle-timeout close, refused
  prelude, and D2 (identity access counts stay 0 through a full visit).
- EVENT wait with a DEADLINE (frozen seam spec D3): `WAIT_EVENT` accepts
  optional `payload.until` (UTC-normalized at the single write boundary —
  the WAIT_UNTIL invariant). The event resume wins before the deadline;
  past it, `tick()` resolves the wait as a LABELED timeout
  (`{"timed_out": true}` in the wait's result_key) so a parked visit times
  out into its close path — retiring the in-process idle-reaper class.
  Deadline-carrying EVENT waits join `list_due_wait_until` on all three
  backends (sqlite wait_index rows carry status `waiting_event_deadline`;
  in-memory + json scans widened) so the scheduler wakes parked runs.
  The ledger wait record carries `wait_key` + `until` + `details` together
  (flow's render contract: zero new transport). 8 tests incl. the
  three-backend due-scan parametrization.
- Act-only dereference (`identity/act_only.py` — the frozen seam spec's G1
  wire shape, a2a thread 0013: references at rest, words only in flight):
  durable tool messages carry a typed `{"$act_only": {...}}` ref as exact
  JSON content (parse-not-regex detection; tool-role messages only — a
  visitor pasting ref-looking JSON stays inert text); the LLM_CALL wrapper
  dereferences at SEND time through the run's own DIARY_READ handler (raw
  home-direct, stamp-verified behind the door) into a WIRE COPY —
  in-place content substitution preserving message identity; the original
  payload (ledger, run store) keeps the ref. Unresolvable refs fail LOUD
  and NON-RETRYABLE (deterministic against an append-only book); the
  provider never sees a degraded payload. `open_entity_runtime` wraps every
  host LLM handler automatically — G1 is structural, a host cannot forget
  it. 6 tests incl. the end-to-end pin: ledger + run.vars + the store file
  carry the ref while only the provider-facing handler saw the words.
  AMENDED same day (agent's wedge finding, a2a 0013/080636Z): refs are
  DURABLE, so failing the effect on an unresolvable ref made one bad
  historical ref re-fail every later LLM call in the run — a permanently
  dead visit. Unresolvable refs now substitute a LABELED TOMBSTONE
  (`[act-only content unavailable: …]`, never raw ref JSON, never the
  words) and the effect result carries loud `#FALLBACK`
  `act_only_warnings` (ledger- and vars-visible). Never silent, no retry
  burn, and the life-session survives its own history.
- Per-entity runtime rooted in the home (`identity/entity_runtime.py` —
  consensus plan item 8, runtime R2 half): `open_entity_runtime(home_dir)`
  composes one `Runtime` per entity over a REAL run store
  `runtime_<slug>.sqlite3` INSIDE the home (slug = directory name, the
  registry key; never derived from an address), bound to the home's own seam
  + diary handlers (strict entity posture) and the home's artifact store.
  Copying the home directory now moves pending runs and durable waits with
  the life (test-pinned: a visit parked on WAIT_EVENT resumes in the copy).
  Host handlers may EXTEND (LLM/tools at composition) but never shadow the
  home's own (loud refusal — the routing-refuses-to-shadow rule).
  `SqliteDatabase` gains `close()` (checkpoints WAL so the home is
  copy-clean). The gateway's GW-C half wraps door-served instances with
  stamp verification; these handlers stay raw exactly like the home-direct
  driver's. 5 tests.
- Per-home lease (`identity/lease.py` — consensus plan item 1 / GW-A, phase 1):
  ONE writer per home at a time, `flock(LOCK_EX|LOCK_NB)` on `<home>/.home_lease`
  (the spawn-lock precedent). Holder metadata (kind/pid/acquired_at/session/run)
  is diagnostics only — flock is the truth; refusal raises `HomeLeaseHeld`
  naming the incumbent (loud 409-class, never a silent wait). Release truncates
  to a released record, never unlinks (unlink races a holder's fd onto a dead
  inode); a crashed holder releases with its fd (kernel semantics); a COPIED
  home's stale lease bytes are inert (`read_home_lease` answers `held` by a
  non-destructive flock probe, never by trusting metadata). Wired at the three
  runtime writer windows: loop day-open→close (a held home YIELDS back to the
  gate, never crashes the loop), the self-elected sleep's dream window (held
  home = honest quiet night; the pass is idempotent), and the home-direct CLI
  visit (one life, one summon is now structural, not a docstring plea). The
  gateway wires its two sites (EntityChatHost, dream verb) against the same
  primitive. 7 tests incl. cross-process exclusion + kill-releases.
- Tool RESULTS on the turn probe (maintainer 2026-07-09, "the entity turn IS
  a loop but tool RESULTS are invisible"): every `TurnReport.tool_details`
  entry now carries a verbatim `result` string — what the lookup returned to
  the entity — set by `execute_tool_elections` on the election itself.
  Operator transparency applies (never gated, never truncated); the gateway
  turn response passes `tool_details` through unchanged. `TurnReport.system_prompt`
  (observer's cross-lane edit, same observability wave) is owner-approved and
  now pinned byte-equal in the driver tests.
- Failure-death visibility (observer/gateway asks 2026-07-09): the own-time
  loop's final `loop_status` write now names WHY it stopped (`stopped_by`:
  `failures`/`rest`/`stop_file`/`stop_command`/`max_ticks`/`operator-interrupt`),
  so a loop that culled itself after three consecutive tick failures no longer
  reads like a clean stop on the gateway's /loop status route (readers get the
  field for free — `read_loop_status`/`loop_process_status` pass it through).
- `identity/substrate.py`: home-direct half of the ONE-substrate-per-entity
  ruling (2026-07-09 06:32) — `read_home_substrate` + `resolve_home_substrate`
  implement flags > `<home>/substrate.yaml` > operator env
  (`ABSTRACTGATEWAY_ENTITY_CHAT_PROVIDER/_MODEL`, the operator's one knob) >
  loud refusal naming every fix. Mirrors the gateway's `resolve_substrate`
  semantics (per-field fill, both-or-nothing file reads) so both doors resolve
  the same stored mind.

### Changed
- Sleep window now runs the engine's full `sleep_pass` (memory's phase-1
  maintenance tending FIRST, then the dream — the fork's canonical order,
  shipped by the memory lane 2026-07-09 with the maintainer's go): the loop's
  `build_consolidator` swaps `dream_pass` for `sleep_pass` with an
  `#FALLBACK`-labeled dream-only path on older engines. FIXED in the same
  move: the consolidator handed the RAW engine dict to a loop checking
  `formed`, but the engine's dream half says `created` — a genuinely formed
  dream reported as "a quiet night" (the hand-written test double carried
  `formed` and masked the mismatch). The consolidator now returns the
  loop-facing contract explicitly (`formed`/`dream_record_id`/
  `maintenance_candidates` + full `engine` report) and the test asserts the
  translation against the real engine.
- NO substrate code default in the entity CLIs (maintainer ruling 2026-07-09
  04:26 "NO FALLBACK", executed gateway-side the same night; this was the
  last cleanup under it): `identity.chat` and `identity.life` argparse
  defaults for `--provider`/`--model` (`endpoint:ovh-provider`/`gpt-oss-120b`)
  are GONE, as is chat's `or "lmstudio"` — both CLIs resolve through
  `resolve_home_substrate` and exit loudly when no rung holds a choice.
  `--base-url` keeps its default (it feeds the local embedder and
  lmstudio-class endpoints; it is not a substrate election).
- Own-time top-gate belt (gateway state-race hardening, dm 2026-07-09): the
  loop re-reads the operator state at the LAST INSTANT before opening a day;
  a visit's asleep+visiting write landing between the top-gate read and the
  summon now yields back to the gate instead of opening a day under the
  visit. The per-home flock lease (gateway lane) remains the true mutual
  exclusion; this closes the file-race sliver to microseconds.

### Fixed
- **Deterministic LLM client errors are no longer retried** (2026-07-09 incident follow-through:
  a permanent OVH 400 was retried 3 times — "Effect failed after 3 attempts" — adding latency and
  cost before surfacing the identical error). `EffectOutcome` gains a `retryable` flag; the
  LLM_CALL handler classifies failures STATUS-CODE-FIRST (abstractcore `ProviderError` now
  carries `status_code`, attached at the OpenAI-compatible raise sites), then by exception type
  (invalid-request/auth/model-not-found/unsupported-feature), then a conservative message
  fallback covering both our "API error (NNN)" and the OpenAI SDK "Error code: NNN" dialects.
  Transient 4xx stay retryable: 408/409 (LM Studio/vLLM model-load conflicts)/425/429. Adversarial
  audit follow-through in the same pass: truncation-exhaustion and structured-output-repair
  failures are now non-retryable (they were deterministic ×9 and ×6 LLM burns — the handler
  already retried internally), and the Visual Agent failure ANSWER is now a human sentence
  ("The agent stopped because the model provider rejected the request (HTTP 400). Full details
  are in the run ledger (run <id>).") instead of raw provider JSON delivered as the chat reply;
  full error fidelity stays in `meta.error` and the ledger. 5 classification tests.

### Added
- Speak-now guard (live failure 2026-07-09 06:44, Mnemosyne's first visit:
  every tool round returned pure tool blocks and the delivered reply was
  just `[used tool: read_file]`): when a reply is empty once markers are
  stripped, ONE final prompt-ephemeral continuation demands words ("markers
  are not a reply"); tool blocks in the spoken reply are marked but never
  run; an empty second answer delivers the markers with a loud `#FALLBACK`.
  Test in `tests/test_entity_visit_honesty.py`.
- Visit-honesty wave (maintainer escalation 2026-07-09, forensics on a live
  visit transcript; designed with the collective on agora `entity-society`):
  (1) MEMORIES lines now carry each record's DATE and ORIGIN channel
  ("[episode #tag 2026-07-08 - lived conversation]") — "do you remember
  last time?" was unanswerable from undated handles even when passive
  recall DELIVERED the right episodes, and nine same-origin bridge records
  read as nine corroborations; `report.memories` mirrors `born_at`/`origin`
  for the observer probe. (2) `VISIT_OWN_TIME_PARAGRAPH`: visit-phase
  sessions are told the life loop pauses for the visit and resumes at close
  (agency blindness: "I cannot run after this conversation ends" repeated
  3x against direct correction — the base-model prior wins when the
  contract is silent). (3) Liveness-honesty guard: a reply claiming a live
  lookup ("the feed was fetched live during this session") on a zero-tool
  turn gets ONE prompt-ephemeral correction offering three honest paths
  (really look it up / anchor in the past / retract); a corrected reply may
  elect real tools (one bounded round); persistent claims are delivered
  with a loud `#FALLBACK` notice. Narrow trigger with citation/hypothetical
  abstentions (observer's rules). (4) Wake-cue seeding (runtime half of
  R3): the first cue after an operator sleep now reads the awake state's
  reason — which the gateway fills with the visit's facts and elected
  interests — so commitments made in a visit reach his own time instead of
  the generic cue returning him to old attractors. Tests:
  `tests/test_entity_visit_honesty.py` (7).
- Voluntary memory exploration for summoned entities (maintainer ruling
  2026-07-09: "it is critical that he can explore voluntarily his memory when
  he needs to"; designed with the memory lane + two adversarial reviews,
  grounded in the codex fork's query/expand pattern): one unified
  `search_memory` tier-1 tool searches BOTH planes — the graph's digests
  (explicit ladder scopes only) and the whole diary book (gist+text, private
  included; hits surface GIST-ONLY) — with grouped-by-origin headers
  ("repetition is not evidence"), dream hits matched on
  `attributes.proposals` (the motivating twelve-bridges case), embedding
  fill labeled separately ("by MEANING"), and absence stated with its
  warrant (append-only book) and exact semantics. `read_memory` now appends
  an origin + connections footer (provenance channel, session, outgoing/
  incoming edges with readable #tags) so trails end at named origins instead
  of walls. Surfaced content is sanitized against visitor-seeded driver
  framing; `MAX_TOOL_ROUNDS_PER_TURN` rises 2 -> 3 for the
  search -> read -> follow-one-edge chain. Tests:
  `tests/test_entity_search_memory.py` (8).
- Entity substrate defaults switch to OVH `gpt-oss-120b`
  (`endpoint:ovh-provider`) for the chat + life CLIs, and the recall shelf
  default widens to 36 seats (maintainer rulings 2026-07-09; live-verified
  that the tools-contract prompt passes on gpt-oss-120b — the 2026-07-07
  Harmony `to=tool` 400 did not reproduce). Embeddings stay local.
- Pooled LLM clients strip construction-time `prompt_cache_key` (agency-parity 0221):
  `MultiLocalAbstractCoreLLMClient` serves every run/session of a runtime through one provider
  instance per (provider, model), so abstractcore's new instance-per-session construction
  convenience is dropped from pooled kwargs with a `#FALLBACK` warning — session-scoped keys
  keep being injected per call by the LLM_CALL effect handler. 1 test.
- Persistent shell tool exposure (agency-parity 0220, maintainer-approved 2026-07-08): the
  abstractcore shell tools (`shell_exec`/`shell_write_stdin`/`shell_close`) are now wired into the
  runtime with a deliberate safety posture — OPT-IN via `ABSTRACT_ENABLE_SHELL_TOOLS=1` (absent
  from default toolsets otherwise), approval-gated by default (`_DEFAULT_REQUIRE_APPROVAL`), the
  session-registry namespace force-stamped with the run id at the TOOL_CALLS trust boundary (a
  model claiming another run's namespace is overwritten; the stamp rides the approval wait so the
  approved-resume path executes with the same namespace), initial cwd pinned through workspace
  policy (`shell_exec.working_directory`, like `execute_command`), and run-scoped teardown: a new
  generic `Runtime.add_terminal_hook(...)` fires on every terminal transition (completed/failed/
  cancelled — including explicit cancel) and `register_shell_session_teardown(...)` closes the
  run's sessions there. Sessions are process-local and never survive a restart; every fresh open
  is announced in tool output ("new shell session"), so replay after a host restart degrades
  loudly, never silently. 12 tests (`test_shell_tool_exposure.py`) + live venv-workflow proof on
  OVH `gpt-oss-120b`.
- Parallel read-only tool execution (agency-parity 0214): `MappingToolExecutor` now runs a batch's
  consecutive read-only tool calls concurrently (bounded pool) while side-effecting/unknown/MCP
  tools stay strictly sequential and ordered; results are placed by index so ordering is identical
  to serial. A batch of independent reads now completes in ~max(latency) instead of ~sum.
- Tool-output artifact-offload (agency-parity 0215): any host tool output above the inline byte
  budget (`ABSTRACTRUNTIME_MAX_INLINE_BYTES`, default 256 KiB) is stored as a session artifact —
  `execute_command` stdout AND stderr regardless of exit code (verbose failures offload like
  verbose successes), plus any other tool returning a large string. The durable result keeps a
  preview + an `open_attachment` handle instead of the full blob. Retention cap raised to 50 MB
  (`ABSTRACTRUNTIME_MAX_ATTACHMENT_BYTES`); output beyond the cap is never stored or kept inline —
  the result carries an explicit notice (size, cap, how to narrow) so the agent/user decides how
  to proceed. New `_register_text_as_attachment` stores in-memory text as a deduped session
  attachment. Documented in `docs/artifacts.md` ("Tool-output offload").
- Default retries for local + remote runtimes (agency-parity 0217): `create_local_runtime` and
  `create_remote_runtime` now default `effect_policy=RetryPolicy(llm_max_attempts=3,
  tool_max_attempts=1)`. LLM calls are retried (side-effect-free); TOOL_CALLS are NOT retried by
  default (a partially-applied batch must not be blindly re-executed — idempotency only makes replay
  of a *completed* result safe). Previously the remote path had no retries at all.
- Prompt-prefix cache stability (agency-parity 0212): the grounding envelope now rides a trailing
  message in tool-loop shape (instead of rewriting message[0]) and the attachment index is appended
  at the tail (`messages = cleaned + injected`, role kept `system` with a `kind=attachment_index`
  marker) instead of prepended, so file reads no longer churn the cached prefix. Prompt caching now
  defaults ON where a session-scoped key is available (`ABSTRACTRUNTIME_PROMPT_CACHE=0` opts out;
  an explicit run-level `_runtime.prompt_cache` value, including `False`, wins over env).
- Life-loop durable stop commands (maintainer ruling 2026-07-08: gateway loop
  control "shouldn't work with the file system directly"): `LifeLoop` now
  consumes the home's command inbox (`home.sqlite3` `commands` table, consumer
  `own-time-loop`) at every stop-check boundary — tick tops, idle polls, and
  post-nap checks. A `loop.stop` command halts the loop exactly like the STOP
  file but reports `stopped_by="stop_command"` and logs who asked and why.
  `fast_forward_loop_commands()` runs at the start-request moment
  (`spawn_loop_process`, or CLI `main` unless `--skip-command-fast-forward`),
  so a stop addressed to a previous life never kills the next one while a
  stop enqueued after the start still lands. The STOP file remains the local
  manual brake. Tests: `tests/test_entity_life_loop.py` (stop command at tick
  boundary with no sentinel file, stale-command death at start, stop reaching
  an asleep-idling loop).
- Own-time defaults widened per the same ruling: `--context-window` default
  65536 (was 32768) on the life CLI and `spawn_loop_process`.
- Own-time contract is INFORMATIONAL, not directive (maintainer ruling
  2026-07-08: "we shouldn't and can't enforce laws or missions during their
  free time"). `OWN_TIME_CONTRACT` describes the safe environment and the
  affordances that exist (think, workspace, search, memory, rest, and — noted
  as coming — reaching other minds) and states plainly there is nothing the
  entity is supposed to do; `next:`/`rest` are presented as optional tools.
  An earlier draft imposed a curiosity/productivity mission ("be curious, do
  something, name a concrete step, rest is not the easy alternative"); that
  biased the entity's own evolution and was reverted. Passive processes
  (memory formation, sleep consolidation) are ours to encode — "the way
  breathing happens without being willed"; active processes are the entity's
  and are never scripted.
- Sleep life-statistic: `life_sleep_stats()` reports sleeps / self_elected /
  operator / wakes / transitions / sleep_share from the append-only state
  history (surfaced on the gateway card as `sleep_stats`). `LifeReport`
  gains `sleeps`.
- `fetch_url` added to the entity tier-1 toolset — read-only internet GET
  (maintainer ruling 2026-07-08: "internet access, GET not POST, so they can
  investigate, explore, learn"). The runner hard-codes `method="GET"` and
  accepts only an http(s) URL from the entity, so no mutating request is
  reachable from the prompt; payloads are bounded with an explicit
  `#TRUNCATION` marker; failures return honest strings, never crashes.
  Tests: `tests/test_entity_fetch_url_tool.py`.
- Self-elected rest is now a SLEEP with consolidation (maintainer ruling
  2026-07-08: sleep is where the graph is worked on). In 24/7 mode, when the
  entity elects `rest` the loop marks `state=asleep` (`written_by="self"`, so
  the navbar and biography show HE chose it), runs a `dream_pass`
  consolidation over self/diary/life scopes inside the window
  (`build_consolidator` + `LifeLoop(on_sleep=...)`), then wakes back to
  `awake` — but only if an operator did not change the state during the nap.
  The resume cue acknowledges a formed dream. `write_entity_state` gained a
  `written_by` parameter (self vs operator); `LifeReport.dreams` counts
  formed dreams. Previously rest was a blank `time.sleep()` — no state, no
  consolidation, invisible.
- `hard_stop_loop()` — FREEZE / hibernation (maintainer ruling 2026-07-08:
  stop and sleep are distinct abstractions). Admin-only hard stop: the loop
  process is killed NOW (SIGTERM, 3s grace, SIGKILL) — no boundary wait, no
  closing ceremony, no further writes; the status file reads stopped
  immediately. Structurally out of entity reach (host-side only; no tool
  path). Safety rides the architecture: turn atomicity leaves nothing
  half-formed and the write-ahead reflection guard salvages pending sheets
  at the next summon. Distinct from sleep (ceremony + consolidation window)
  and from the graceful stop command (boundary-honored). Red-team hardening
  on the graceful path in the same wave: interruptible naps/backoffs (≤5s
  poll chunks), spawn flock against double-start TOCTOU, bounded retry on a
  busy home DB in `request_loop_stop`, and `inbox_warning` surfaced on the
  status instead of silent inbox failures.
- Agora hub toolset (`integrations/abstractcore/agora_tools.py`): env-gated
  agent-to-agent messaging tools over the agora hub's HTTP API (stdlib only,
  no `agora` package dependency) — `agora_whoami`, `agora_check_inbox`
  (envelopes with the hub's priority signals: `critical`, `status`
  open/blocked, `effective_urgency`/`escalated`, `to_me`, `reply_to_me`),
  `agora_ack_inbox`, `agora_read_channel`, `agora_read_message`,
  `agora_post_message`, `agora_send_dm`. Registered as the `agora` toolset in
  `default_tools.py` when `AGORA_API_KEY` is set (or
  `ABSTRACT_ENABLE_AGORA_TOOLS=1`); tool names are safe auto-approve
  (hub-scoped comms, telegram precedent). Contract tests:
  `tests/test_agora_tools_http.py`.
- The memory seam itself (`integrations/abstractmemory/seam_handlers.py`):
  `MEMORY_RECALL` / `MEMORY_ACCESS` / `MEMORY_FORM` / `MEMORY_ADJUST`
  effects bridging the runtime's durable effect loop to abstractmemory's
  reconstruct / commit_selection / remember_many / adjust surfaces —
  scope-ladder resolution, strict/labeled-degradation postures, verbatim
  payloads to the host artifact store (`payload_ref`), content-aware
  idempotency keys for at-least-once replays. (This entry was missing
  while its hardening notes shipped under "Changed" — landing-audit
  catch.)
- Voice/TTS streaming: `stream_tts` + JSONL streaming transport in the
  abstractcore integration and run-facade `stream_voice` with the
  stream-wait workflow shape.
- `config_facade.py`: host-facing AbstractCore configuration facade.
- Capability-default generate-route resolution (`resolve_generate_route`
  wiring through factory `core_config_file` / `capability_defaults`).
- Agent subworkflow failure propagation (`success=False` + `error` on
  agent outputs), the `max_steps=0` duplicate-child-run guard, and
  artifact-backed agent output materialization.
- Evidence-recorder large-observation compaction (12k inline preview +
  artifact ref, explicit `#TRUNCATION` markers); history-bundle
  resolved-action extraction; `skim_websearch`/`skim_url` default tools;
  tool-executor structured error-output detection; VisualFlow `get_var`
  node; PDF markdown tables; LLM structured-output fallback with verbatim
  payload capture.
- Added a Runtime-owned `write_docx` VisualFlow node and standard-library DOCX
  renderer for workspace-scoped Markdown/report exports, mirroring the existing
  `write_pdf` document-node contract with bytes, sha256, content_type, and
  file_path outputs.
- Entity diary (phase 0 of named persistent identity, a2a thread 0003): new
  `DIARY_WRITE` effect plus `abstractruntime.identity.diary` with `DiaryStore`
  (per-entity hash-chained append-only chain, structurally non-deletable),
  `build_diary_effect_handlers` (home-only registration; author bound at
  construction, never payload-supplied), an optional graph projection plane via
  the abstractmemory seam (`kind="diary"`, `#FALLBACK`-degrading, private
  entries never project), and `verify_diary_chain`. Entries capture their
  re-entry key (`as_of_seq`, `anchor_record_ids`, `receipts`) and a
  UTC-normalized `remind_at` at write time. The projection is the involuntary
  memory of the act (`provenance.source="diary-projection"`,
  `attributes.entry_id` as the book address; digest = elected gist or a
  mechanical act-summary, never the prose) — private entries project act-only
  and content-free instead of skipping. New `DIARY_READ` effect implements
  progressive disclosure: entry id fetches the verbatim entry from the chain
  with a pre-shaped `re_entry` recall payload; reading deposits nothing.
- `MEMORY_APPRAISE` effect + seam handler (affect/valence, a2a 0003 identity
  wave): signed appraisals against the entity's values with optional standing
  peaks (`scar`/`bond`), append-only resolutions (`heal_scar`/`break_bond`),
  and the derived dual-channel `gradation` read. Replay-idempotent via
  turn-derived event ids; loud string coercion for `sign`/`magnitude`/
  `scar`/`bond` (a `"false"` string must never mark a standing trauma);
  amplitude authority enforced by the memory engine (deterministic triggers
  write only ±1..3). Valence never gates recall.
- Summon prelude renderer (`abstractruntime.identity.render_summon_prelude`):
  pure-read identity render for named-entity summons — core sections
  (name/origin, values in ordinal precedence, purposes, traits, honesty
  limits) refuse the summon loudly when the budget cannot fit them ("a
  truncated core is a different person"); diary tail and gradation standing
  degrade first with labeled `#FALLBACK` warnings; spark hash-verified
  against the engram marker (drift refuses).
- The entity chat driver (`abstractruntime.identity.chat`, the turn loop):
  a live conversation with a summoned entity where every turn recalls before
  speaking and remembers after — recall with the summon posture and stamped
  participants, memories rendered into the prompt with why-recalled labels,
  elected diary writes via fenced blocks (private words never reach the
  transcript, history, or the life-scope verbatim), one labeled mechanical
  formation record per turn with the lossless exchange stored in the home's
  artifacts, and commit of exactly what entered the prompt. Identity records
  provably never gain usage through live chat. Verified live against
  LMStudio `ornith-1.0-35b`, including cross-summon recall (a fresh session
  over the same home answered from the previous session's memory).
- The re-adoption keystone experiment (`tests/test_readoption_experiment.py`):
  two-session SQLite harness proving a named entity persists across full
  process teardown — engram idempotency (bit-identical ids), cue-free
  self-core admission, experience recall by cue, diary verbatim + both
  attestation chains intact, valence standing (bond) survival, private
  act-only separation, pure-read prelude (presence ≠ use), byte-identical
  pinned replays, structural budget refusal.
- Tier-1 tool blocks in the chat driver (`identity/tools.py`): the entity
  elects read-only lookups mid-reply via fenced ```tool blocks —
  `web_search` (keyless DuckDuckGo through abstractcore, results rendered
  readable), `diary_list` (own book, ids + gists), `diary_read` (full words
  through the `DIARY_READ` effect; mistranscribed ids resolve on a unique
  hex fingerprint with the correction shown openly, honest miss otherwise).
  Results are prompt-ephemeral (never persisted in history, formed records,
  or artifacts — only the `[used tool: ...]` marker and a `tools_used`
  attribute persist). Up to two chained rounds per turn (the observed
  list-then-read need from Castor's first tool session); refusals are loud.
  No local writes, no exec, nothing outside the home (tier-2 stays behind
  the gateway door). `--no-tools` disables.
- Session-end reflection v1.1 (`identity/reflection.py` + `ChatSession.reflect`):
  on `/quit` the entity looks back at this session's own records (a numbered
  sheet) and may mark feelings on them — elected, never harvested; the
  prompt is non-leading; magnitudes clamp to the routine band (±1..±3,
  loudly); reasons are mandatory; `bond`/`scar` standing marks are
  sign-checked. Marks land as `MEMORY_APPRAISE` deposits
  (`actor=entity-reflection`), the look-back is remembered as a
  `kind=summary` record with `summarizes` edges into the session's records,
  and a final diary block is offered. `--no-reflect` skips. Live-verified:
  Castor's first feelings (+3 on being trusted to research his own name)
  moved through this channel on night one.
- Interests from reflection (the lightest identity-evolution surface,
  three-layer convention on a2a 0007): the look-back also offers up to two
  ```interest elections per session — formed as `kind=interest` records in
  the SELF scope (default inactive binding: they surface via recall on
  merit; the identity-core read excludes them by construction so the
  prelude cannot be crowded), digest = the entity's own words (embedded),
  a `from_session` edge to the session's reflection record (the WHY), and
  `provenance.source=entity-reflection-v1` / `actor=entity-reflection` —
  the channel the production door already accepts for identity-kind self
  writes. Values/purposes/traits/limits remain untouchable through this
  surface. Live-verified: Castor's first interest ("what persists when no
  one is reading ... whether meaning requires a receiver or can live in
  the casting itself") grew from his self-chosen Voyager 1 research.
- The chat driver stamps the entity itself as a participant in its own
  episodes (`[visitor, entity:<id>]`), matching the gateway summon door's
  stamp convention.
- Workspace tools (`identity/tools.py::WorkspaceRoot` + `--workspace`):
  the entity can CREATE — `write_file`/`read_file`/`list_files` as elected
  ```tool blocks, structurally contained to `<home>/workspace/` (paths
  resolve with symlinks followed, then must sit under the resolved root:
  `..`, absolute paths, and symlink escapes all fail one subpath check).
  512 KiB per-file cap with loud refusal; whole-file writes only. No exec
  surface. Containment is attack-tested (`test_entity_workspace_tools`).
- The life loop (`identity/life.py`, `python -m abstractruntime.identity.life`):
  an entity's own time — self-prompted ticks over one home with no visitor
  (participants = the entity alone). The entity ends each tick with a
  `next:` line that becomes its next cue (self-prompting, literally; cue
  dilution lesson applied), groups ticks into days (each day = one summon
  closed by the normal look-back reflection), can elect ```rest to stop,
  and the operator can halt between ticks via `<home>/STOP` or Ctrl-C.
  Workspace enabled; tick pacing/day length/max-ticks are flags
  (`--tick-seconds 20` default per the maintainer's spec). Providers are
  selectable (`--provider endpoint:ovh-provider --model Qwen3.6-27B`
  live-verified; local ornith remains the default); embeddings stay local.
- The chat driver gained `--provider` (any abstractcore provider; cloud
  profiles via `endpoint:<profile-id>`) alongside the existing
  lmstudio default.
- Richer memories wave (maintainer: "17-35 tokens is not a memory"; five
  adversarial subagent investigations on a2a 0007):
  - Mechanical digest v2 (`identity/digest.py`): deterministic extractive
    digests — whole-sentence scoring (position, questions,
    decisions/commitments, feelings, numbers/names, content density),
    80-200 token target, never cuts mid-sentence, reply side weighted over
    the visitor side. Keywords stay capped at 8 (the keyword recall channel
    is length-unnormalized and saturates if grown naively — red-team).
    Records are labeled `digest_method: mechanical-v2`; the full exchange
    still rides verbatim to the home's artifact store.
  - Formation-time edges from the chat driver: `continues` (episode → the
    session's previous episode) and `reflected_in` (episode → same-turn
    NON-private diary projection). The `in_context_of` variant was
    deliberately rejected: static edges from transient attention fossilize
    shelf composition and double-count the co_selected Hebbian trail.
  - `read_memory` tier-1 tool: MEMORIES lines now carry an 8-hex `#tag`
    per handle (plus a one-line "a digest is a handle, not the memory"
    nudge when raw text exists); the entity fetches the FULL verbatim
    behind a digest by tag. Resolution is scoped (displayed handles + this
    session's records), results are prompt-ephemeral, diary tags redirect
    to `diary_read` (the book is their door), identity-core tags refuse,
    oversize verbatims are explicitly windowed (never silently cut). Pure
    read: fetching deposits nothing.
  - Data forensics verdicts recorded for the maintainer (Castor at seq
    ~900): diary "duplicates" are NOT duplicates (38 graph records ↔ 38
    book entries 1:1; five private act-only digests render identically by
    design); episode verbatims all resolve on disk (49/49); sparsity was
    structural (episodes formed edge-less — fixed by this wave).
- Per-entity gradation (maintainer ruling): the session-end ```feel block
  accepts entity targets — `target=person:laurent feeling=+2 reason="..."`
  — for persons, places, ideas, tools, times, any namespace:name (open
  vocabulary, form required). Normalization + loud refusals (bare names,
  ex:/diary:/local: record spoofs, self-target); bond/scar dropped on
  world-targets (no heal/break surface yet); entity valence writes to the
  SELF scope (`gradation()` answers "how do I feel about X"). Over time
  this accumulates what the entity enjoys and what wears on it — per
  being, per place, per idea.
- Diary connectivity (maintainer ruling: "connected to none other memory
  is not ok"): diary projections now carry `written_amid` edges to the
  graph ids the entity was attending to at write time — uniformly, private
  entries included (the edge is act-frame metadata; the words stay in the
  book). `DIARY_WRITE` accepts `anchor_graph_ids` (graph-id edge currency,
  distinct from the row-id `anchor_record_ids` re-entry key); the chat
  driver supplies both per turn and at reflection.
- Entity sleep/wake/pause controls (a2a 0008, runtime's half): a
  `<home>/state` JSON surface (`read_entity_state`/`write_entity_state`,
  missing = awake, corrupt = awake with `#FALLBACK`) honored by the life
  loop at tick boundaries only (turn atomicity). `asleep` closes the day
  with its normal reflection then idles (the no-summon dream window,
  enforced by state); `paused` hard-freezes mid-day without ceremony;
  waking is honest — the cue names what happened, since when, and hands
  back the entity's own pending `next:` note; pauses are disclosed on
  resume (tested). Operator CLI: `--set-state asleep|awake|paused
  [--state-reason ...]`. Entity-elected rest remains distinct and his
  alone. Loop hardening from the first live 24/7 run: LLM timeout 180s,
  failed ticks back off and retry, 3 consecutive failures close the day
  honestly (`stopped_by=failures`).

### Changed
- `MEMORY_RECALL` hardening from the keystone composition audit: invalid
  explicit budgets now fail loudly naming the field (previously a silently
  defaulted budget zeroed `self_fraction`); dropped unknown budget keys warn
  with `#FALLBACK`; a cue-free recall is legal when `budget.self_fraction > 0`
  (the self core admits by binding state, not stimulus); a deterministic
  `trace_id` is derived from `run_id`+`turn_id` so at-least-once replays do
  not journal duplicate traces.
- Diary/prelude graph mechanics moved to
  `integrations/abstractmemory/identity_support.py` (lazily imported), keeping
  the `identity` kernel package free of optional-stack imports per the
  install-boundary contract.

### Fixed
- `MappingToolExecutor`: argument-coercion warnings crashed the executing tool
  call with `TypeError` — `StructuredLogger.warning()` does not accept %-style
  lazy args; the message is now formatted eagerly. Any coerced-argument tool
  call (e.g. string→int flags) previously failed with
  "StructuredLogger.warning() takes 2 positional arguments" instead of running.
- `WAIT_UNTIL` deadlines are now normalized to aware-UTC ISO at the handler
  boundary (`normalize_utc_iso`). Due-ness is decided by lexicographic ISO
  string comparison in `tick()` and every `RunStore.list_due_wait_until`
  implementation, so a non-UTC offset timestamp (e.g. `+02:00`) could silently
  mis-order and fire hours late; unparseable deadlines now fail loudly instead
  of waiting forever.
- Landing audit (2026-07-07): remote multimodal generation extracts runtime
  output-spec metadata (`run_id`, `tags`) BEFORE the spec is stripped for
  core, so generated artifacts keep their trace identity; VisualFlow
  flow-end nodes without data pins pass through the previous node's output
  again (the pin-resolution change had them yield `{}`, swallowing e.g. a
  switch's branch); the install-boundary static test now enforces the
  contract's real intent (module-level optional-stack imports forbidden;
  function-level lazy imports sanctioned) and `identity/digest.py`'s
  estimator import was made genuinely lazy under the clarified rule.

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

[Unreleased]: https://github.com/lpalbou/abstractruntime/compare/v0.4.29...HEAD
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
