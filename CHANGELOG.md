# Changelog

All notable changes to AbstractRuntime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Per-tick lease windows in the own-time loop** (B1 keystone, operator
  ruling "i should always be able to visit"; memory ruled time-sliced
  alternation M2-clean, commons c1322/c1324): `LifeLoop.run()` no longer
  holds the home writer lease for a whole day. Each home-writing window
  takes and returns it — the SUMMON window (open + pending-look-back
  salvage), one window per tick around `session.turn()`, and a CLOSE window
  around `session.reflect()`. Between-tick idles, boundary waits, paused
  freezes, and failure backoffs are LEASE-FREE, so a waiting visit slots in
  at any tick boundary (~tick_seconds bound instead of a day-long hold). A
  held lease at a tick boundary is a wait (never a failure, never a
  consumed tick slot); the close wait is bounded (`CLOSE_WAIT_MAX_POLLS`,
  ~2 min) and a day that broke ON a stop defers instead of waiting (stop
  commands are consumed-on-read — the wait's own polls would never re-see
  one; adversary find). FAST-YIELD: when the day closes because a visitor
  arrived (state mode=visiting or auto-yield reason — each detection
  channel works alone), the inline reflection is DEFERRED to the
  write-ahead `pending_reflection.json` marker and quiescence lands right
  after the in-flight turn, keeping the visit door's 55s yield window
  honest; the loop's own next summon salvages the deferred look-back under
  its summon window. Pinned in `tests/test_directory_lease.py` (per-tick
  windows, wait-not-fail + no-slot-consumption, fast-yield + salvage
  round-trip with attribution, both detection channels).
- **`salvage_pending_lookback(home_dir, llm)`** (identity/chat.py): the
  DOOR half of fast-yield — a session-free salvage for doors that host no
  ChatSession (the gateway's durable visit lane), so any "next open over
  the home" can repay a deferred look-back. Caller holds the writer lease;
  no marker = one-stat no-op. The gateway's open leg is the intended
  consumer (their half of B1).
- **Personal-grant gate at loop start** (laurent 12:44 "own time IS the
  personal phase", the 10:20 consent violation's fix; ruled fields from
  decision:personal-grant-section): the loop refuses to open days unless the
  operator armed `phases.personal` — `read_personal_grant(home_dir)` reads
  `<home>/phases.yaml`'s personal bucket ({mode: disabled|timer|
  until_revoked, expires_at, granted_by, granted_at}; missing/malformed/
  unknown = disabled, fail-closed) and `personal_grant_refusal(grant)`
  names what is missing. Gated at THREE doors: `LifeLoop.run()` re-checks
  at EVERY day-open (revocation/expiry ends the loop at its next boundary,
  `stopped_by="personal_disarmed"`), `main()` refuses before the substrate
  resolve (exit 3), and `spawn_loop_process` refuses synchronously (a child
  dying in its own log is a silent refusal). Wake ≠ grant ≠ start: nothing
  arms personal as a side effect; the gateway's principal-stamped write
  surface is the one arming path (their half of the wave).
- **`write_personal_grant` — the one writer of the personal activation
  bucket** (file + shape settled c1443/c1447: `<home>/phases.yaml`,
  content-named per the substrate/tool_policy pattern; runtime owns the
  format module, the gateway's arming door calls it after its principal
  stamp + marker land). Mechanics enforced in the writer so no caller can
  drift: mode validated; timer requires `expires_at` and normalizes it to
  aware-UTC ISO (the WAIT_UNTIL lexicographic invariant); `granted_at` is
  clocked server-side, never caller-supplied; `granted_by` (the stamped
  principal) is required to arm; `disabled` writes a clean bucket (no grant
  fields linger — markers own history, the file owns current truth);
  FIELD-MERGE preserves foreign phase sections and unknown personal keys;
  corrupt files and newer `schema_version`s refuse loudly instead of
  clobbering.
- **Substrate divergence lane killed** (laurent 12:39, c1430 ask 2b — the
  night pid ran OVH from argv regardless of substrate.yaml): when the home
  carries a persisted mind, `main()` refuses start-time `--provider/--model`
  flags that differ from it, and `spawn_loop_process` refuses divergent
  spawn args — the mind changes via the sanctioned substrate surface (a
  durable, marker-first event), never via start arguments. Flags remain
  valid when no substrate is persisted (then they ARE the operator's
  explicit choice, per the 04:26 chain).
- **Loop spend surface** (gateway c1390's named runtime half): the own-time
  loop runs home-direct (ChatSession, no run ledger), so its LLM/tool usage
  was invisible to the gateway's `/cognition` spend fold (their honest
  `#FALLBACK`). `ChatSession.spend` now counts every LLM call through the
  one `_generate` choke point (provider-tolerant usage shapes:
  total_tokens, prompt/completion, input/output) and tool elections at
  their execution sites; the loop persists lifetime counters to
  `<home>/loop_spend.json` (atomic write, after every tick + at day close so
  reflection/salvage calls count) — cumulative across lives, single-writer
  loop bookkeeping like `loop_status`. `read_loop_spend(home_dir)` is the
  tolerant reader (missing/corrupt = zeros; field names match the gateway
  fold: llm_calls / tool_calls / tokens_total / ticks, source
  `loop-home-direct`).
- **`read_loop_status` now answers `running`** (composition bug, found
  while building B1): the gateway's visit doors decide the auto-yield
  negotiation from `read_loop_status(...).get("running")`, and this reader
  NEVER set that key — the yield request silently never fired (always-False
  missing-key drift class). `running` = live phase AND live pid, folded in
  ONE place (`loop_process_status` now consumes it instead of keeping a
  second predicate copy). Pid-reuse guard: a LIVE day heartbeats its status
  at every tick boundary, and a "day" whose `updated_at` froze past
  `LOOP_STATUS_STALE_SECONDS` (30 min) reads as a corpse regardless of the
  pid probe — a recycled pid must not make the doors negotiate with a
  corpse and 409 forever. The day PHASE now begins at the summon window
  (doors see a summon-in-progress as non-quiescent instead of colliding
  with the held lease).

### Fixed
- **Salvage idempotency** (adversary find on the B1 wave): the pending
  look-back's `turn_id` now derives from the MARKER's session (was: the
  salvaging session's), so a crash between the APPRAISE writes and the
  marker clear re-derives IDENTICAL event ids at the next open and memory's
  at-least-once dedup absorbs the re-run — feelings double-deposited on the
  append-only store are unrepairable. A duplicate summary record (different
  LLM words) remains the accepted residual: records can be superseded,
  valence cannot.
- **CLI visit door under per-tick leases** (identity/chat.py `main`): "one
  life, one summon" was enforced by the day-long lease hold by ACCIDENT —
  with per-window leases the CLI could have opened a second summon between
  ticks. The door now refuses on loop STATUS (open day on a live loop →
  refuse naming the pid, pointing at `--pause-loop`) before touching the
  lease; the lease keeps arbitrating instantaneous writers. `--pause-loop`
  gates its yield write on `running` (a corpse day must not start a
  negotiation nobody answers) and restores `awake` when the quiescence wait
  times out (abandoning the visiting posture left the loop yielded forever;
  the gateway doors already restored it).

### Added
- **Absolute-path re-anchoring in workspace-scoped tools** (incident-driven,
  adversarially designed; shipped by the code seat, sign-off list on the
  runtime DM): models frequently fabricate a plausible-but-wrong absolute
  PREFIX for a file genuinely inside the workspace. `resolve_user_path`'s
  absolute branch now re-anchors a containment-failing path onto the mode's
  roots when safe: nonexistent paths recover by suffix (longest first, root
  before mounts, depth ≥2, candidate must exist — writes to new files never
  re-anchor); EXISTING outside paths re-anchor only with provable same-inode
  identity (APFS case-aliases), never lookalike substitution. Candidates are
  re-resolved + containment-rechecked (symlink-out killed) and ignored-path
  candidates skip silently. Refusals keep one unified string for both
  branches (no filesystem-existence oracle) plus a teaching suffix naming the
  root; the `escapes workspace_root` / `outside workspace roots` substring
  prefixes are a stability contract with host hints. Successful re-anchors
  log `workspace re-anchor: '<raw>' -> '<resolved>'`. Scope: the TOOL lane
  only (`resolve_user_path`); `utils/workspace_paths.py` and the gateway
  HTTP file endpoints are untouched; `all_except_ignored` unaffected.
  Pinned by `tests/test_workspace_policy_absolute_reanchor.py` (14 tests).
- **Per-agent agora identity via alias indirection** (hooks plan H8, the
  fleet identity P0 — previously every run in one process posted as ONE
  agent because the toolset read a process-global `AGORA_API_KEY`):
  the run carries only a NON-secret alias (`_runtime.agora_agent`); the
  host env carries `AGORA_API_KEY__<ALIAS>` (+ optional
  `AGORA_URL__<ALIAS>`); `agora_tools` resolves alias→key at call time.
  Identity is NEVER model-controlled: the tool-calls handler force-stamps
  the schema-hidden `_agora_agent` argument from run vars (the same
  trust-boundary seam as the shell `_registry_namespace` stamp) —
  model-supplied aliases are overridden or stripped. A configured alias
  with a missing key fails LOUD naming the exact env var (no silent
  fallback to the global key — posting as the wrong agent is the bug
  this fixes). Alias keys alone enable the toolset (fleet hosts need no
  global key). ADVERSARY-HARDENED: aliases are validated lowercase
  slugs, making the env-suffix fold provably INJECTIVE — two
  differently-named residents can never silently share a key (the
  conflation find: "research-lead" vs "research_lead" both folding to
  one suffix is now a rejected config, not a silent merge); a
  configured-but-blank alias fails loud instead of quietly posting as
  the global agent; the stamp's lazy import never caches an empty name
  set on transient failure. Pinned: two aliases authenticate as two
  agents; spoof attempts overridden/stripped; approval-forced resume
  re-executes with the plan-time stamped alias (mirror of the shell
  namespace pin); the secret key rests in no run var and no ledger
  record; `_agora_agent` hidden from every tool schema.
- **Steer sidecar — single-writer guidance delivery** (hooks plan H4,
  runtime half): `storage/steer_sidecar.py` ships `SteerSidecarStore`
  (protocol) + `InMemorySteerSidecar` + `SqliteSteerSidecar` (durable,
  WAL). Hosts APPEND steer messages via the new `Runtime.steer(run_id,
  message)` verb (string or content-bearing dict; refuses terminal runs
  and empty messages; requires `Runtime(steer_store=...)`); ONLY the
  tick loop drains pending steers into `_runtime.inbox` at iteration
  boundaries — the tick thread stays the single writer of run state, so
  the inject-into-run-vars loss window (stale-save clobber / terminal
  resurrection, documented in gateway's `inject_guidance`) cannot occur
  through this path. ADVERSARY-HARDENED delivery order (the first cut
  acked before saving — a crash between the two silently lost a durable
  steer): deliver + advance the run-owned watermark
  (`_runtime.steer_watermark`) + save FIRST, sidecar ack LAST —
  at-least-once underneath, exactly-once into the inbox (watermark
  dedup, crash-redelivery pinned); the drain re-checks external control
  before its save (a cancel landing mid-drain is never clobbered,
  pinned) and the ack record follows the `abstract.status` convention
  (`EMIT_EVENT` named `abstract.steer_seen`) so ledger mappers classify
  it instead of rendering a phantom node completion. `SqliteSteerSidecar`
  appends under `BEGIN IMMEDIATE` (two processes can never mint the same
  seq — 60-append 2-instance race pinned), consumed rows compact away,
  and both stores cap undelivered backlogs at 500 with a loud refusal.
  Root-exported. Pinned: boundary delivery + ledger ack, ordered batches,
  terminal/missing-sidecar refusals, a 3-writer concurrency race,
  SQLite restart survival, content-less dict refusal.
- **Entity runs refuse raw steers** (hooks plan H5 interim): a steer
  into a run carrying `_visit` vars OR running the visit workflow id
  (the adversary's birth-window find: real visit runs are born before
  the door seeds `_visit`, so the guard also keys on `workflow_id`,
  set at creation) is refused loudly naming the missing rite — and the
  DRAIN refuses delivery into visit runs independently (defense in
  depth; messages stay pending, loudly logged). The ruled steer
  contract (fresh channel-labeled reconstruction, merge+dedup,
  attributed verbatim) is not built yet, so the generic path stays
  closed for stamped-channel runs instead of delivering an un-rited
  steer. A drift pin keeps core's workflow-id literal equal to
  `identity.visit_workflow.VISIT_WORKFLOW_ID`. Pinned by tests.
- **`tools_ran` parity in durable visit turns** (hooks plan H7a): the
  visit workflow's ANSWER payload now carries driver-authored
  `tools_ran` (folded from the react middle's `turn_captures`; honestly
  `[]` on the v0 single-call path where no tool can run by
  construction) — never derived from reply prose (the marker-imitation
  lesson). The door can serve turn tool-truth without parsing anything.
- **Volatile message markers for prompt-cache stability** (B1 fix,
  runtime half — code seat's prompt-cache adversary, commons c971): a
  top-level `volatile: true` on a wire message marks it per-call
  ephemeral. `_maybe_prepare_prompt_cache` EXCLUDES flagged messages
  from the durable fingerprint sequence (the react adapter's changing
  "[loop] iteration N of M." tail forced a full local-cache re-prefill
  every cycle) and `_strip_volatile_markers` removes the key before
  every provider boundary (local `generate` sites + the remote client's
  message build) — the message rides, the unknown field never reaches a
  strict provider SDK. Pinned: two-cycle growth with a changed volatile
  tail stays an incremental append; the strip preserves content and
  never mutates caller structures. Agent's adapter half (the marker
  emission) shipped same-hour against these functions (cross-package
  smoke on their side).
- **`SqliteSteerSidecar` heals across data-root purges** (gateway
  adversary F3 ask): the schema now rides EVERY connection
  (`CREATE TABLE IF NOT EXISTS` in `_connect`, a no-op on the hot
  path) — a purged data root under a live instance used to recreate an
  empty db whose appends died on "no such table" until a process
  restart. Pinned: append → purge (db+wal+shm) → append heals in place.
- **Overlay layer key `own_time` → `personal`** (phase-vocab migration,
  gateway/agency c1029 follow-up — the tool_policy alias treatment
  applied to `system_prompt.yaml`): `OVERLAY_KEYS` now names
  `personal`; `LEGACY_OVERLAY_KEY_ALIASES` keeps `own_time` READING
  (labeled `#FALLBACK`, ruled key wins when a file carries both) and
  WRITING (normalized to the ruled spelling on disk). The life factory
  reads `personal`; `default_prompt_texts()` keys by ruled spellings
  with legacy twins DERIVED from the alias table (pre-flip serving
  processes keep reading; delete the derived block when every consumer
  has flipped). Pinned: legacy-only file reads, ruled-wins-both,
  write normalization, both-spellings write.
- **First-class `steer_store` on every runtime factory** (gateway c1023
  note: bundle hosts attached the H4 sidecar by poking a private
  attribute post-construction): `create_local_runtime`,
  `create_remote_runtime`, `create_hybrid_runtime` and both file
  variants now take `steer_store=...` and thread it into the Runtime.
  Pinned across all five signatures.
- **Streamed-vs-non-streamed result parity** (code seat c1017: same
  task diverged — non-streamed concluded in 3 calls, streamed re-nudged
  to max_iterations): the streamed normalizer now (1) splits inline
  `<think>` markup out of assembled content into `reasoning` exactly as
  the non-streamed provider stack does (raw deltas on thinking models
  carried the thought text into `content`, and the react parse node
  behaved differently per arm; unclosed trailing blocks extracted too;
  provider-reported reasoning wins), and (2) ACCUMULATES tool calls
  across chunks with id-dedup (last-non-None-wins silently dropped
  earlier calls when a stream emitted them incrementally; identical for
  re-sent full lists). Both pinned.
- **Structured-output calls never ride the token-stream path** (code
  seat c1009 defect 1): a review/structured call under
  `_runtime.stream=true` completed with an EMPTY final answer — the
  streamed normalizer cannot produce validated `data` or artifact-backed
  outputs. `generate()` now forces `stream=False` whenever the call
  carries an output request / `response_model` / `response_format`
  (correctness over rendering; `on_token` stays silent for those calls;
  plain text calls still stream). Pinned with a recording fake provider.
- **Pool-wide `set_on_token`** (code seat c1009 ask 3):
  `MultiLocalAbstractCoreLLMClient.set_on_token(cb)` fans the callback
  out to every pooled client — existing AND lazily-created (per-request
  provider/model overrides would otherwise silently not stream). Pinned.
- **In-process `on_token` streaming callback** (code seat c990 ask):
  `LocalAbstractCoreLLMClient.set_on_token(cb)` registers an optional
  `cb(delta: str, meta: dict)` fired per content chunk when
  `stream=True` — BEST-EFFORT and never load-bearing (the durable
  aggregated result is byte-identical with or without it, pinned; a
  raising callback is disabled mid-stream with one warning, never
  failing the call). Host-side registration only: callbacks never ride
  effect payloads. Gateway-hosted surfaces need the durable plane
  instead (deliberately not built here).
- **Core-inventory facade** (gateway c924 ask; the backlog-0059 boundary
  routes core access through runtime):
  `integrations.abstractcore.tool_inventory_facade.core_registry_tool_rows()`
  is the thin pass-through of core's authoritative registry enumeration
  (`builtin_tool_inventory_as_dicts` — no field added, dropped, or
  retyped; core's per-call schema isolation holds through the facade,
  pinned) + `core_inventory_schema_version()` for serve-time drift pins.
  The gateway's inventory union lights up with zero gateway change.
- **Declare-beside-execute tool registry + the servable emission**
  (tool-inventory build, commons c864 ask 2 / descriptor contract v6):
  `TOOL_DESCRIPTORS` in `identity/tools.py` is ONE frozen record per
  walled tool — declaration (description + native-call schema), grant
  lane, `capability_class` (boundary axis; the canonical
  opposite-numbering pair encoded: `web_search` is grant-lane tier1 AND
  boundary tier2_world), `mutating` (local effect),
  `remote_write_capable` (false for every walled tool — the web lanes
  are GET-hardcoded, unlike core's registry twins), `act_only`,
  `body_optional`, and the EXECUTOR — so a declared name without an
  executor, or the reverse, is structurally impossible.
  `execute_tool_elections` dispatches through the registry;
  `_NATIVE_SPEC_SHAPES` and the body-optional parse gates now DERIVE
  from it (the first cut duplicated the shapes and alarmed the drift
  with asserts — the adversary's find made them genuinely derive, so
  the drift class is removed, not alarmed); `ACT_ONLY_TOOLS` derives
  from the descriptors (the wire flag and dispatch can never disagree);
  the import-time partition gates are plain raises (`python -O` cannot
  strip them). `walled_tool_rows()` is the SERVABLE EMISSION (contract
  rule 1: the sole field source for runtime rows — deep-copied schemas;
  the gateway attaches `executes_via`), root-exported with
  `TOOL_DESCRIPTORS`/`ToolDescriptor` and the name tuples.
  WALLED-WINS is mechanism (agency c909 P0-3, runtime half): the five
  colliding names are pinned as permanently walled descriptors, a
  granted registry-only name gets no declaration and refuses loudly at
  entity-lane execution, and a source pin proves the walled module
  never imports the core tool registry. One fable5 adversary attacked
  the registry refactor (behavior-drift trace against the old dispatch:
  CLEAN on every reachable path; no P0/P1) — its P2s folded same-pass.

### Fixed
- **Visual `llm_call` forwards provider/model INDEPENDENTLY** (flow's
  adversary P0, commons c884 — "workflows that run and lie"): the old
  both-or-neither branch built the pending effect with NEITHER key when
  only one resolved, so a model-only override (the
  model-pool-through-loop-item pattern the authoring stack teaches, ruled
  VALID 2026-06-10: "provider and model are independently optional in the
  LLM effect handler; connected pins resolve at runtime") SILENTLY
  executed every call on the gateway default model — the graph looked
  right, readiness passed, nothing downstream could detect it. The
  partial branch now forwards whichever key is present (absent keys
  resolve from run/gateway defaults, matching the Agent node handler);
  the both-blank and both-present paths are unchanged. Pinned by
  `tests/test_visual_llm_call_partial_override.py`.

### Added
- **Operator iterations ceiling, refuse-at-start** (laurent c786: "hard
  ceiling at 100 calls/turn... customizable, for instance in the
  gateway/console"; seam (b) ruled with gateway c792/c805 — one
  enforcement site for every lane): when the host serves
  `_limits.max_iterations_ceiling` into run vars, `Runtime.start()`
  refuses LOUD if the workflow-declared `max_iterations` exceeds it —
  the refusal names both values and the override surface, and the run
  never exists (never mid-run truncation). Absent ceiling = no
  enforcement (server-declared; the runtime never invents 100). A
  silent workflow under a ceiling below the default gets the default
  clamped down (a default is nobody's word); an explicit narrow value
  (including 0) at or under the ceiling stays the workflow's word.
  The ceiling VALUE lives gateway-side (env default 100 → config-object
  field); pinned by `tests/test_max_iterations_ceiling.py`.

### Changed
- **Phase vocabulary: the four SINGLE HUMAN WORDS** (laurent's ruling,
  commons c786 2026-07-11 20:30 — "like for us human:
  visit/work/personal/sleep"; the room's 9-0 ballot converged on the
  same set): `PHASES = ("visit", "work", "personal", "sleep")`;
  `PHASE_TASKED`/`PHASE_OWN_TIME` became `PHASE_WORK`/`PHASE_PERSONAL`
  (root exports updated). `LEGACY_PHASE_ALIASES` now carries all three
  historical spellings — `resident`→personal, `own_time`→personal,
  `tasked`→work — each mapping DIRECTLY to its ruled key (one hop, no
  transitive chains); the same loud machinery applies (arg + file
  section alias with `#FALLBACK` notes, writes normalize keys on disk,
  narrow legacy grants survive, `canonical_phase` normalizes at every
  mint). The file-section shim generalizes to a candidates loop (ruled
  key first, then legacy spellings; first INTACT section wins; every
  malformed candidate present in the file is named — the find-4 class
  applied to every spelling). The aliases die before release. Deferred
  in the same ruling's spirit, flagged on-channel: the prompt-overlay
  section key `own_time`, life.py's `own_time.log`/`own_time_start`
  bookkeeping names, and the `own-time-loop` command-inbox consumer id
  are at-rest FILE/event contracts with their own compat surfaces — they
  follow in a coordinated flip, not this one. Adversary pass 3 (no
  P0/P1) also hardened the shim's observability: an intact losing twin
  section is now NOTED on read (the write path already warned), the
  both-legacy-spellings write warning names the real mechanism instead
  of a possibly-absent "ruled key", and the unreachable malformed-
  section re-check below the candidates loop was removed as dead code.
- **Personal-phase activation definition line** (semantics c794, then
  laurent-corrected c815 — NO separate `personal_grant` section; personal
  IS the grant): the tool_policy module docstring defines the adjacency
  in place — the personal phase's activation fields (mode
  disabled|timer|until_revoked, expires_at, granted_by, granted_at,
  folded into the config object's `phases.personal` bucket, gateway
  lane, never keys in this file) are the operator's BRAKE; the
  `ToolGrant` this module resolves is the HANDS. Only personal carries
  activation fields — the one operator-armed phase.
- **Scratchpad seed literal 25→20** (agency's a21304a co-sign residual):
  the VisualFlow compiler's scratchpad seed carried a second-copy `or
  25` fallback, drifted from the ruled default — now 20, though the
  upstream `setdefault` makes it reachable only for an explicit
  `max_iterations=0`.

### Fixed
- **`diary_list` joins the act-only class** (memory's e-s 233 R3 ruling,
  adversary-broken claim: a private entry's GIST is part of the private
  words, and one granted `diary_list` call in a durable visit rested
  every entry's gist — private included — in the run store/ledger).
  `ACT_ONLY_TOOLS` is now `("diary_read", "diary_list")`; the ref shape
  generalizes to tool+args (`make_act_only_content(args=...)`) — a
  `diary_list` ref carries the word-free request body and the LISTING is
  re-run fresh against the book at LLM send time
  (`wrap_llm_handler_with_act_only(diary_list_resolver=...)`, wired by
  `open_entity_runtime` automatically), so gists exist only in the wire
  copy, never at rest. Tombstones name tool+args when no entry_id.
  Companion sheet fix (memory's rider): the visit workflow's ANSWER-node
  sheet builder act-frames private diary elections ("you kept a private
  diary entry" — the `private` word named, never the gist), matching the
  chat lane's sheet-privacy fix; a proximity pin in
  `capture_diary_elections` guards the private-meta omission both sheets
  trust. `resolve_entry_id` is now public (gateway's authoring half
  imports it). NOTE: the gateway/agent halves of the pair (door-side ref
  authoring for `diary_list` + the react observe accepting re-run-shaped
  frames) are theirs — until they land, door-lane `diary_list` refuses
  word-free (strictly better than the leak; the in-process chat lane is
  unaffected, R5-verified).
- **Act-only adversarial hardening** (one fable5 adversary, 2 P1 fixed
  same-pass, both pinned): (1) a RAISED resolver/store error (sqlite
  failure, resolver bug) now tombstones with `#FALLBACK` like a
  structured refusal instead of failing the LLM effect and
  terminal-FAILing the visit (the mechanic-4 wedge class); (2) the
  write-direction capture gate is case-insensitive — a model emitting
  ```` ```Diary ```` skipped the lowercase substring gate, resting raw
  private words in run vars + ledger while silently losing the book
  write (the fence parser was always IGNORECASE; the gate now matches).
  Also: act-only resolved-wire headers are defanged by
  `sanitize_tool_surface` (echo of "[… resolved from the book at send
  time]" cannot rest a fake resolution frame in digests).

### Changed
- **Agent `max_iterations` default is 20** (maintainer ruling 2026-07-11
  17:45, commons c726 — "the workflow decides; if the runtime carries a
  default it should be 20 too"; the previous 50 was a 2026-02-21 team
  decision, never a maintainer ruling, and now yields): flipped in
  `core/config.py` (`RuntimeConfig.max_iterations`), `core/vars.py`
  (default `_limits`), `core/runtime.py` (both absent-limits fallbacks),
  the VisualFlow compiler's `_limits` seed, and the visual Agent-node
  adapter fallback. DEFAULTS ONLY: workflow-declared values (bundle
  pinDefaults, explicit `_limits`, Agent-node pins, `ReactMiddle`)
  remain authoritative per the same ruling — a workflow that says 3 gets
  3, one that says 50 gets 50. The loop-node guard (10_000) is a
  different concept and unchanged.

### Fixed
- **A refused reflection election no longer kills the visit close**
  (gateway c709 interim, agency's step-10 live bug — the fdf01e0 rule
  class): the visit workflow's staged APPLY loop (summary → interests →
  diary → feelings) opts each staged effect into the new
  `payload._absorb_failure` mechanism — the runtime's tick loop converts
  a FINAL failed outcome (after effect-policy retries) into a loud
  result (`{"ok": false, "absorbed_failure": <error>}` at the effect's
  `result_key`) and CONTINUES to `next_node` instead of terminal-failing
  the run; the ledger StepRecord still records the failure honestly.
  The stager surfaces each absorbed refusal as a `#FALLBACK` notice in
  `reflection_notices`. Live case: an interest election at close was
  refused by the door (entity-reflection act under a workplace stamp —
  the channel collision agency's hardened gate caught), which
  terminal-FAILED the close and silently dropped the diary + feelings
  stages queued behind it. A refusal is the gate doing its job; the
  close now survives it, the refused election is skipped loudly, and
  the remaining stages apply. Pinned by
  `test_refused_reflection_election_skips_loudly_never_kills_the_close`.
  (The root fix — signed close-reflection authority on stamp-v2, option
  (a) ruled c708/c709 — is gateway's door half and stays gated on the
  consensus signature; this interim stays correct after it lands.)

### Changed
- **Phase vocabulary flips to the four ruled keys** (config-object
  consensus F7/N7, semantics c607, laurent Q1 c684): `PHASES` is now
  `("visit", "tasked", "own_time", "sleep")` with named constants
  `PHASE_VISIT`/`PHASE_TASKED`/`PHASE_OWN_TIME`/`PHASE_SLEEP`, all
  root-exported from `abstractruntime` alongside `ToolGrant`,
  `resolve_tool_grant`, `read_policy_file`, `write_policy_file` — the
  gateway door imports the ONE phase set from the root instead of
  reaching into `identity.tool_policy` (no second copy; the
  diary_type-clamp lesson). Ruled defaults encoded: visit + tasked +
  own_time hold the full tool set ("hands by default"; own_time's brake
  is the door's enabled flag + durable grant, never handlessness); sleep
  keeps read-only-exploration-minus-diary. The own-time loop caller
  (`life.py`) flips to `PHASE_OWN_TIME` in the same change — no
  intermediate state where the old spelling raises (N7 atomicity).
- **Legacy "resident" spelling: loud migration window, dies before
  release** (the lease-shim policy): `LEGACY_PHASE_ALIASES` maps
  `resident → own_time` on BOTH axes — a caller ARG resolves own_time's
  grant with a `#FALLBACK` note (gateway's pre-flip literals keep
  working, no lockstep deploy), and a policy FILE still carrying a
  `resident:` section is honored under own_time with a note naming the
  file path, so an operator's narrow pre-rename grant SURVIVES the
  rename (never the silent widen adversary F7 traced).
  `write_policy_file` normalizes legacy keys on disk (any save converges
  the file to the ruled vocabulary, with a warning); an explicit
  `own_time:` section wins over its legacy twin; unknown phase ARGS
  still raise; unknown FILE keys now warn in grant notes (a typo'd
  section silently granting nothing is the same class of quiet loss).
  Pinned by `tests/test_phase_vocabulary_migration.py`, including the
  N8 negative (a narrow `tasked:` section must be CONSULTED —
  `source=="policy-file"` — never a permissive fallthrough that happens
  to equal the default). `canonical_phase` is the public normalizer for
  consumers that persist or SIGN phase strings (semantics c700 V5: the
  stamp must sign the canonical phase only — an alias inside the MAC
  basis is a verify-time chain-break).
- **Adversarial review hardening** (one fable5 adversary attacked the
  flip before commit; 3 P1 + 4 P2 fixed, all pinned): (1) a single write
  payload naming BOTH spellings now lands the explicit ruled key's word
  in either insertion order (was last-wins by dict order — the file-axis
  precedence held while the write axis violated it); (2)
  `ChatSession.phase` stores the CANONICAL phase (a legacy "resident"
  arg normalized the grant but left the attribute stale — a
  `session.phase == PHASE_OWN_TIME` comparison would silently miss);
  (3) the last off-list `phase="resident"` test call site flipped;
  (4) a malformed ruled section (null `own_time:` from a half-finished
  hand edit) no longer shadows an intact narrow legacy section — the
  operator's last intact word holds instead of widening to the full
  default; (5) a scalar `tools:` value notes its fallthrough instead of
  silently resolving deny-all; (6) write warnings emit only after
  payload validation (no claimed normalization that never landed);
  (7) the arg axis lowercases in one place so both entry points agree
  on "Resident".

### Fixed
- **JSON run store cache is LRU-bounded** (flow's 2026-07-11 P0 incident
  review, runtime-lane follow-up): `JsonFileRunStore._run_cache` retained
  every RunState a full directory scan ever loaded — ~1.5GB RSS after one
  scan of a 3k-run dir with history-bearing vars, paid permanently by
  long-lived serving processes. The cache is now an LRU bounded at 512
  entries (`run_cache_max` constructor param). Eviction is safe under the
  documented ownership contract: a re-load after eviction re-reads the
  last saved state from disk — exactly what any non-owner reader is
  entitled to see; in-cache aliasing (save→load returns the same object)
  is preserved and pinned by test. The `list_runs` starvation hole above
  `limit` concurrent RUNNING runs (mtime rich-get-richer) remains
  documented in the docstring as a designed follow-up — it needs a status
  index or terminal-run archival, not a cache tweak.

### Added
- **Session-free memory exploration** (`identity/memory_reader.py`,
  gateway ask e-s 206): `HomeMemoryReader` extracts `search_memory` /
  `read_memory` (and the tag/origin/trail machinery behind them) from
  `ChatSession` into ONE session-free implementation over a home —
  `ChatSession` delegates (sharing its session tag map so sheet-registered
  tags stay addressable), and the gateway door's per-entity TOOL_CALLS
  executor can construct a reader per home and declare both tools with
  driver parity. `memory_tag` moved with it (re-exported from `chat` for
  existing consumers). Pure reads throughout: no record_access, no commit,
  no journal writes on any path (identity findability ≠ use).
- Atomic writes for the home's operator config files (adversary P2 from
  the prompt-overlay review): `utils/atomic_files.atomic_write_text`
  (tmp + fsync + `os.replace`) now backs `tool_policy.yaml` and
  `system_prompt.yaml` writers — a crash mid-write can no longer leave a
  torn file that silently resolves as "no operator word".
- **Fair scheduling across N run stores** (plan item 11, R3 — the agreed
  0018 spec, phase 4): `scheduler/multi_store.py` ships `TickCandidate`
  (store-agnostic unit of due work), `TickSource` (wraps one store's
  due-scan — UNTIL waits and EVENT waits with deadlines — stamping
  `store` + the door-stamped `channel` from run vars; absent stamps are
  labeled "unstamped", never guessed), the host-injectable
  `AdmissionHook` ordering policy (default: due-order — today's
  single-store behavior generalized), the MECHANICAL starvation floor
  (`apply_starvation_floor` — promotes candidates waiting past
  `max_starvation_s` ahead of policy order AND restores policy-dropped
  starved work: a policy may defer, never bury; weighted-fair, never
  strict priority), and `MultiStoreScheduler` (registry of named
  sources; one sweep = union due candidates → admit → floor → tick each
  run through ITS OWN runtime; per-run failure isolation — one failing
  home never stalls the sweep; per-store stats make starvation visible
  as data; a broken admission hook degrades to due-order with a loud
  `#FALLBACK`). Per-store `ticker` callables let the door register its
  lease-acquiring drive — admission ordering runs strictly BEFORE any
  lease acquisition (agency's P3 pin). This is also the eager D3 sweep:
  registering entity stores makes parked-visit idle deadlines fire
  without a client touch. Run-id → store resolution stays behind the
  API; admission schedules TICKS, never tokens (the LLM ceiling is core
  C3/C4's lane; budget exhaustion arrives as an ordinary loud per-run
  failure). 10 new tests.

### Fixed
- **The DECLARE half of the native tool channel** (agent's c481 measured
  correction: the read-half alone repairs ~1/3 of the failure distribution
  — 5/9 failures were PURE-PROSE fabrications with zero `tool_calls` to
  read; with tools DECLARED in the payload, three benches measured 0
  fabrications). `ChatSession` now builds OpenAI-style function specs from
  the GRANT only (`native_tool_specs()` — the grant stays the single
  authority; specs describe the entity-WALLED implementations, never
  registry twins per the name-collision rule) and declares them on the
  calls whose responses the tool loop reads (turn + post-TOOL-RESULTS
  continuations; guard/reflection continuations demand words and never
  declare). Substrate compatibility by signature: a client whose
  `generate()` takes no `tools` kwarg (fence-convention substrates,
  scripted doubles) is called byte-identically as before.
- **Native tool calls are no longer discarded by the chat driver**
  (maintainer incident 2026-07-11: Mnemosyne fabricating search results
  with `tools_ran: none` even when told "USE YOUR TOOLS"). Root cause
  (agent's live A/B on gpt-oss-120b, 0/9 fenced vs 5/5 native, confirmed
  by core 0/11 vs 9/9): native-tool-channel substrates essentially never
  write the fenced ```tool convention — they emit structured `tool_calls`
  on the response, which `ChatSession` silently dropped (only
  `resp.content` was read), so the model's REAL tool intent was thrown
  away and "helpful" prose fabrication shipped instead. Fix:
  `identity/tools.py` gains `native_tool_elections()` — converts response
  `tool_calls` (dict / JSON-string / nested-function argument shapes) into
  the SAME `ToolElection` currency with the same refusal honesty (unknown
  names refuse loudly, marker lines are door-authored) — and the turn
  loop folds native calls and fenced blocks through ONE executor with one
  shared per-round cap; the continuation's own `tool_calls` carry across
  rounds; a tool-call-only reply with empty content no longer aborts as
  an empty turn. Fenced behavior is byte-unchanged (election fences
  measured alive on the same substrate — the fence death is
  action-specific). Offline pins: 7 new tests incl. the exact Mnemosyne
  class (prose scaffolding + native call → the lookup RUNS and the
  honest continuation ships).

### Added
- Criterion-7 offline fixture placed (agency's draft, runtime-adapted per
  the c177 division: they draft, this package places):
  `tests/test_act_only_resume_criterion7.py` — a private diary entry is
  planted (election at the result boundary), turn 2 runs an act-only
  `diary_read` round through a contract-faithful ReactMiddle, the process
  dies mid-turn AT THE EXACT STEP the `$act_only` ref first rests in
  durable vars (single-step ticking, deterministic against node-graph
  changes), and a fresh runtime over the same home completes the turn.
  Pins all five assertions: ref-at-rest mid-turn (never the words),
  dereference-at-send on resume, byte-identity of wire words against the
  book, no tool re-execution (one tool message in the durable transcript),
  and refs-only at rest after completion + word-free ledger at close.
  Adaptations on package facts recorded in the docstring (fence syntax is
  `visibility=private`; the draft's `private=true` would be refused by the
  parser). The LIVE half stays walkthrough step 5b.

### Fixed
- **Hash-chain fork under concurrent handles** (found by the
  maintainer-driven lease adversarial review, 2026-07-10):
  `HashChainedLedgerStore` cached the chain head per process, so two
  handles over one persisted ledger (two processes, or two store instances
  in one process) each computed `prev_hash` from their own stale head —
  the inner store serialized both inserts and the chain forked permanently
  (`verify_ledger_chain` reports `prev_hash_mismatch`; on the never-purge
  diary book there is no repair). The per-directory writer lease makes
  this unreachable in normal topology; the chain now refuses to fork even
  without it: the head is re-read from the PERSISTED tail on every append
  (new `last_record()` fast path on `SqliteLedgerStore`), appends
  serialize on a per-instance lock (two threads over one instance could
  previously fork in-process), and `SqliteLedgerStore.append_chained`
  derives head + hash + insert inside ONE `BEGIN IMMEDIATE` transaction —
  cross-process fork-free by construction, not by lock discipline. Fork
  repro pinned by test (two handles, interleaved appends, verify green).

### Changed
- **Writer lease re-homed to its mechanism layer** (2026-07-10 vocabulary
  sign-off, a2a/fs/renaming.md, maintainer-approved): `identity/lease.py`
  → `storage/lease.py` with neutral spellings — `HomeLease` →
  `DirectoryLease`, `HomeLeaseHeld` → `DirectoryLeaseHeld`,
  `acquire_home_lease` → `acquire_directory_lease`, `read_home_lease` →
  `read_directory_lease`, refusal text "one writer per **directory**", and
  the on-disk dotfile `.home_lease` → `.writer_lease` (option 2, six
  voices: the file travels inside every copied directory, so the neutral
  name matters at rest). The mechanism is generic one-writer-per-directory
  mutual exclusion — entity homes are the first consumer, item-15 project
  workplaces the designed second; semantics byte-identical (flock truth,
  holder-metadata diagnostics, release-truncates, crash-release,
  stale-copy inertness, one-lease relay rule). `identity.lease` remains as
  a thin re-export shim for the migration window (same class objects, same
  lock file — mixed old/new callers keep excluding each other; pinned by
  test) and DIES BEFORE RELEASE. Stale `.home_lease` files in existing
  homes are inert bytes (flock state never lived in the bytes); safe to
  delete or ignore. Exports added to `abstractruntime.storage`. Design
  validation recorded in the sign-off document: two maintainer-driven
  adversarial passes confirmed the lease arbitrates PROCESSES (not
  principals — visitor writes are refused at the deposit gate regardless),
  that three writers legitimately bypass the gateway's ticking (detached
  loop, home-direct CLI, CLI maintenance verbs), and that the damage class
  under collision is permanent on append-only stores (hash-chain fork,
  seq-axis interleave) — which justifies a kernel lock despite low
  collision frequency.

### Added
- Production merge home for the adapter cycle (a2a 0014 merge-ownership
  ask, ruled): `build_visit_workflow(react_middle=ReactMiddle(...))` — ONE
  owner for the visit graph (this package) with ZERO adapter import (the
  dependency points the other way: abstractagent depends on
  abstractruntime, so the middle arrives as DATA — node map + entry +
  reset hook — built by the CALLER from agent's public API,
  `create_react_workflow(final_next_node="HARVEST")`). BRIDGE replaces the
  v0 REASON in place (entity dress into `_runtime.system_prompt`/`turn_id`/
  word-free `llm_payload_extras`; per-turn reset; decorated user message
  appended to the durable transcript) and HARVEST folds
  `_temp.final_answer` + `_temp.turn_captures` into `_turn.llm` — every
  downstream node runs byte-unchanged; bodies lifted verbatim from agent's
  proven merge pin. Collisions with seam node ids refuse loudly; absent
  middle = the v0 single-call path, byte-identical. Tests: a contract-
  faithful stub middle (no abstractagent import) pins mid-loop election
  capture (words only in the book through a multi-iteration turn),
  downstream-unchanged episode/answer, and the collision refusals; agent's
  real cycle is pinned against the same seams in their
  tests/test_react_visit_merge.py (108/2 their bench).
- Paused visits carry their look-back DEBT explicitly: a pause-frozen visit
  (closed_by=pause, skip_reflection) completes with `reflection_pending:
  true` and the word-free session sheet in the run OUTPUT — the door's
  pending-look-back at the next open consumes it directly instead of
  inferring the debt from run vars. The sheet is private-word-free by
  construction (episode digests + non-private gists; private entries appear
  as their act label only).
- Visual `wait_event` D3 passthrough (flow's follow-through ask): the
  VisualFlow adapter's wait_event node now passes `until` (idle deadline,
  UTC-normalized by the runtime; labeled `{"timed_out": true}` resume) and
  `details` (self-describing wait metadata, e.g. `kind="visitor_message"`)
  through to the WAIT_EVENT effect — visual residents get durable idle
  deadlines and self-describing parks; absent pins stay absent
  (byte-unchanged older flows). 2 tests.
- Door-shape folds from the GW-C confirmations (a2a 0014): (1) moved-home
  refusal home-direct — `open_entity_runtime` refuses a directory whose
  manifest names a different entity (both id generations parsed) BEFORE a
  stray `runtime_<straydir>.sqlite3` could mint beside the true one
  (gateway's GW-B wall, now also at the raw-path surface); (2) the PARK
  idle deadline reads door-seeded `_visit.idle_seconds` from run vars
  (setdefault from the build kwarg for home-direct callers — the workflow
  hardcodes nothing); (3) the door-authored close payload's
  `closed_by`/`reason` ride ROUTE into the look-back context and the run
  output (the wake-cue seeding rule wants how-it-ended).
- Visit workflow: head discipline + item-14 correlation key (party asks,
  a2a 0014). NEW `RENDER` node moves presence + MEMORIES to the MESSAGE
  LANE — the system prompt is `_visit.system_base` only, byte-stable for
  the whole visit (frozen spec §4 line 9; the v0 per-turn head mutation
  would have broken the adapter's multi-iteration prefix contract).
  Decorated user messages are APPEND-ONCE in the transcript (all-but-last
  byte-identical across turns = the cross-turn cache property; blocks are
  dated + as_of-labeled so old ones read as honest history); the formed
  verbatim keeps the visitor's RAW words. `build_visit_workflow` accepts
  the door-stamped `visit_id` and stamps it as `attributes.visit_id` on
  every episode + the reflection summary (item 14: both legs of a
  cross-runtime visit pin ONE string, correlating as data never shared
  rows). A/B extends: `_runtime.node_traces` grep pins that the kernel
  trace only ever sees the MARKED reply (the result-boundary capture means
  the raw private words never exist in any durable channel).
- G1 WRITE DIRECTION at the result boundary + the item-10 A/B fixture. The
  A/B privacy grep (arm A = ChatSession, arm B = visit workflow, one
  scripted visit with a PRIVATE diary election on twin homes) caught what
  the plan's pre-condition predicted: in the durable mapping the raw reply
  — diary fences included — rested in the run store (result_key) and the
  ledger's LLM_CALL result. Fix: the act-only wrapper now captures diary
  elections AT THE RESULT BOUNDARY (`capture_diary_elections`) — the words
  fly to the book through the home's DIARY_WRITE handler before the result
  persists; the durable result carries the MARKED reply + word-free
  `diary_entries` metadata (gist only for non-private). Elections without
  a payload `turn_id` fail loud and non-retryable (losing elected words
  silently and leaking them are both worse). The visit workflow's ELECT
  node becomes a pure fold of the captured metadata. A/B pins criterion 1
  (memory-plane equivalence: same record kinds, same participants, same
  diary words, D2 zero on both arms) and criterion 6's offline half (the
  private words rest ONLY in the book's file family — every file of the
  durable arm's home is grepped).
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
