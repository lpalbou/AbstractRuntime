# Changelog

All notable changes to AbstractRuntime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`substrate.yaml` gains an optional `thinking` field (reasoning plan R4,
  2026-07-26)**. `read_home_substrate` now returns the reasoning-effort
  field when the file carries one — spelled `thinking` at rest (the plan's
  one-spelling decision). The field only rides when provider and model are
  both set (a reasoning knob without a chosen mind is meaningless); blank
  values are dropped; files without the field read exactly as before. The
  resolver keeps its (provider, model) shape — widening it and the
  consuming lanes is the coordinated implementation wave with the gateway.
  This unblocks the gateway's substrate-triple build (their writer can
  store the field without it being silently dropped on read).
- **Endpoint-profile resolver context around tool execution (Case-1 seam,
  2026-07-26)**. Gateway-registered `endpoint:*` profiles are per-principal
  and invisible to core's in-process `create_llm` — session-route tools
  (`analyze_media`) could not construct them. `MappingToolExecutor.execute`
  now binds core's `use_provider_endpoint_profile_resolver` context around
  each batch (ONE wrap site — policy path, approval-resume, and tool_invoke
  all funnel through it by construction); the resolver getter is late-bound
  over the run's own llm_client and attached at `build_effect_handlers`
  through delegate-chain walking (`attach_endpoint_profile_resolver_getter`
  exported for hosts composing elsewhere). Parallel pool submissions AND
  the tool-timeout thread run under copied contexts (a bare thread starts
  context-empty on CPython 3.12 — the timeout lane silently dropped the
  seam; adversary P1-1). The getter falls back to the public
  `resolve_provider_endpoint_profile` attribute so the REMOTE client lane
  lights up too (adversary P1-2). Fable5-adversaried; ten pins in
  `tests/test_endpoint_profile_route_context.py`.
- **`_session_route` trust-boundary stamp (vision-capability ruling,
  2026-07-26)**. Delegated sight must use the session model — fallbacks are
  solely for vision-less models (operator ruling). The TOOL_CALLS handler
  now stamps the run's own route (`{"provider", "model"}` from
  `_runtime.provider/model`) onto declared consumer tools
  (`_SESSION_ROUTE_TOOL_NAMES`, exact names — currently `analyze_media`),
  derive-not-claim: any payload-claimed `_session_route` is popped on EVERY
  call (spoofs die everywhere), and absent route vars stamp nothing (core's
  fallback path byte-identical). Partial routes stamp what exists so core's
  degradation fires its labeled `#FALLBACK` warning instead of silence.
  Fable5-adversaried (no P0/P1); five pins in
  `tests/test_session_route_stamp.py`.
- **`abstractruntime.__version__` (2026-07-26)**: canonical version surface
  for hosts comparing a bundle's `metadata.min_runtime` against the serving
  runtime (flow's pin-expression enforcement gate). Lazy module
  `__getattr__` over `importlib.metadata` — the installed dist is the one
  truth. Consumers must compare with `packaging.version.Version`, never
  string comparison.
- **Replay-integrity wave (code-tui incident + fable5 server audit,
  2026-07-25)**. Three runtime-owned fixes for 10-14MB single-turn history
  bundles and silent-omission paths:
  - **Appendix-aware `$slim` dedup (`storage/ledger_slim.py`)**: the layout
    dedup byte-compared observability/provider-request copies against an
    exact reconstruction of the STARTED payload — one ~350B message appended
    by the agent adapter AFTER payload build defeated byte-identity for the
    whole ~250KB copy on every call (0-of-59 dedup hits measured; 5.95MB of
    a 14.3MB bundle). Now the reconstruction may match as an ordered
    SUBSEQUENCE with the few small extras carried verbatim in the marker
    (positions + items, bounded: <=8 items, <=4KB each, <=half the value);
    the sha over the full value keeps correctness structural. Match order
    prefers least-verbatim markers (exact field, exact layout,
    layout+appendix, field+appendix). Mutated conversations still rest
    verbatim — dedup drops duplicates, never information.
  - **Conversation-field floor**: known conversation fields
    (`messages`/`system_prompt`/`prompt`) now dedup above 512B instead of
    4096B — the audit measured a 3,918B system prompt (178B under the old
    gate) duplicating on every one of 32 calls.
  - **`detail="replay"` bundle profile (`history_bundle.py`)**: a labeled
    projection for transcript folds — drops request-side payload fields and
    the two observability metadata paths (each replaced by a structured
    `$omitted` marker naming profile/field/bytes) and skips the timeline.
    The exact bundle stays the default; projection is spine-copy only
    (stored records never mutated).
  - **In-band bundle `warnings[]`**: subtree discovery failures, ledger
    read failures, torn-row skips, tail-window truncation (with
    total-vs-window and the $slim-orphan caveat), subtree caps, and
    input_data offload failures are now reported IN the bundle instead of
    silently omitted (operator ruling: a bundle that cannot be complete
    must say so).
- **Inline pin expressions (tier 1, 2026-07-25)**. A data input pin on a
  VisualFlow node may carry a small sandboxed Python expression in
  `node.data.pinExpressions: {pin_id: "vars.fix_cycles < 3"}` — its OWN
  field, deliberately never a `pinDefaults` sentinel, so pre-expression
  compilers skew SAFE (unread key → pin falls back to its default → falsy
  conditions keep loops bounded; the truthy-dict encoding that would spin
  while-loops was rejected by design review). Expressions compile ONCE at
  flow build (RestrictedPython eval mode; an unparseable expression fails
  the build naming node+pin) and evaluate at input resolution on the
  consuming node — both resolver lanes (pure + exec) share one
  `apply_pin_expressions` step, and while-conditions re-read `vars.*` fresh
  each iteration. Environment: `vars` (read-only run-vars view), `value`
  (the pin's would-have-been value), `parse_json`/`to_json` + the code-node
  builtin set (one `sandbox_helper_globals()` source shared with the Code
  node sandbox, which gains `parse_json`/`to_json` in both lanes). A
  raising expression fails the consumer's step naming `<node>.<pin>` with
  an expression preview. Module: `visualflow_compiler/visual/pin_expressions.py`.

### Changed
- **`RestrictedPython>=7.0` is now a declared dependency (2026-07-25)**. It
  was always the intended Code-node sandbox (the import guards existed) but
  was never declared, so plain installs silently ran the weaker basic exec
  lane. MIGRATION NOTE: code bodies that only ever ran on the basic lane
  can now refuse under RestrictedPython policy at their first execution —
  the known case is augmented assignment on subscripts (`d["k"] += 1`
  → "Augmented assignment of object items and slices is not allowed";
  rewrite as read/modify/write, the rule documented since 2026-02-20). Pin
  expressions REFUSE to run without RestrictedPython rather than fall back
  to bare eval.

### Fixed
- **A failed diary write no longer throws the entity's reply away
  (record-everything ruling, 2026-07-26)**. When the book write failed
  (disk or store error), the turn failed loudly but the whole reply —
  including the words the entity elected to keep — was lost. Now the raw
  reply is saved first to `<home>/rescue/reply_<run>_<turn>_<suffix>.json`
  (with the error, and the write-time anchor context when the caller has
  it), then the failure stays loud and names the file. Covered lanes: the
  durable visit capture (`act_only.py`, both failure paths), the chat
  driver's turn path, AND the session-close reflection path (the review
  adversary found the reflection lane carried the same hole after the turn
  path was fixed). Shared runtimes pass a per-run rescue location; a
  raising resolver degrades to no-rescue with a warning — the safety net
  must never be what drops the reply. Seven pins across
  `tests/test_diary_write_rescue.py` and the chat-driver suite.
- **Streamed reasoning folds keep the COMPLETE thought (core contract v1,
  2026-07-26)**. Both stream folds (local normalization + remote SSE lane)
  persisted the FIRST non-empty `metadata.reasoning` — but streamed
  reasoning arrives as per-chunk snapshots and core guarantees the TRAILING
  chunk is the complete aggregate, so one fragment rested durably and the
  operator's keep ruling was silently violated. Now last-non-empty wins;
  blank/absent trailing metadata never erases; the display-only
  `reasoning_delta` key is never read into the durable fold. Pinned in
  `tests/test_llm_streaming_on_token.py`.
- **Remote LLM client forwards `thinking` (reasoning-1st-citizen R-A,
  2026-07-26)**. The remote chat-body pass-through allowlist silently
  dropped the `thinking` param while the abstractcore server accepts it —
  any gateway built on the remote runtime lost reasoning configuration
  entirely (found independently by two seats' adversaries). Strings ride
  trimmed; junk shapes stay off the wire. Three-case pin in
  `tests/test_remote_llm_client.py`.
- **Stale `search_files` spec pin**: the integration test asserted the
  ABSENCE of `output_mode`/`context_lines`/`case_sensitive` params — a
  snapshot pin that abstractcore's 2026-07 search improvements legitimately
  invalidated. The test now pins its actual intent (max_hits/head_limit
  behavior).
- **`JsonlCommandStore.append` fsyncs before returning (quit-contract
  thread, 2026-07-25)**. The gateway answers `POST /commands` 2xx only
  after the append returns, so the durable claim behind pause/cancel
  delivery must put the record ON DISK, not merely in the OS page cache —
  a flush alone left a power-loss window between flush and sync. Commands
  are low-frequency (pause/cancel/resume, never hot-path), so the fsync
  cost is negligible. Test-pinned (one fsync per accepted append;
  duplicates never re-write, so they owe no fsync). Honest boundary: the
  SQLite command store (entity-home inboxes) stays WAL
  `synchronous=NORMAL` — process-crash durable; widening to power-loss is
  a per-connection pragma decision, not store-local.
- **Tend lanes forward the verified reflection channel (cross-package break,
  memory `tend.py` entity-seat P0, 2026-07-25)**. Memory's
  `apply_tend_elections` gained a mandatory `channel` that must equal
  `entity-reflection` — the privileged default was removed (it let a
  workplace-stamped run tend as the entity's own reflection). Both runtime
  tend call sites passed no channel, so every tend/dispose silently refused
  (the entity could no longer dispose dreams on either lane — the exact
  capability the MEMORY_TEND route restored). Fixed: the home-direct chat
  driver states `channel="entity-reflection"` (true by construction — the
  fence fires on the entity's own reply reflecting on its own memory in its
  own home session, memory's explicit sanction); the `MEMORY_TEND` handler
  FORWARDS the channel from the payload and NEVER defaults a privileged
  constant (the door injects the verified channel for stamped runs; absent
  → the engine refuses loudly, no self-authorization). The `memory_tend`
  VisualFlow node gained a `channel` pin. Both call sites carry a
  version-tolerant `TypeError` ladder for engines predating the kwarg.
- **Private diary verbatim never rests in a ledgered tool result
  (adversary C2 + gateway c5403)** (2026-07-25). Gateway proved flow-brain
  runs ledger in the BASE store (shared plane), so `ENTITY_TOOLS_EXECUTE`
  resting a private `diary_read`'s words was the 2026-07-07 diary-leak
  class. On the resting lane (`results_rest_durably`) a private entry now
  serves its act-frame + gist only, never the body — store-independent
  (a property of the handler, gateway's framing), mirroring the containment
  every HTTP/observer surface uses. The prompt-ephemeral chat lane is
  unchanged (results never rest). The effect header + private trailer also
  dropped the now-false "home's own ledger" claim (flow-brain rests in the
  base store) for the store-neutral "a durable record".
- **Entity-lane cleanup pass — four fable5-confirmed fixes (operator-ordered
  seat-plan wave, 2026-07-25)**. C1: `MEMORY_CONSOLIDATE`'s version-skew
  ladder never drops `report_only` — an old engine that cannot do a pure
  read now FAILS honestly instead of silently degrading to an unguarded
  write pass (the lease + paused/STOP gates are conditioned on
  `report_only`, so dropping it ran a write the result still labeled a read
  — read→write inversion); drops now come from a fresh kwargs copy so one
  rejected kwarg keeps the others. C3: over-cap tool notices name the
  APPLIED cap, not the raw `MAX_TOOL_BLOCKS_PER_TURN` constant (the effect
  lane's 6-call batch told the model 14 remained). C4: mechanical/no-gist
  diary projection titles carry the entry-id tail — the same same-day
  uniform-title collision the slug fixed for other entries (mechanical
  close notes are the most frequent same-day multiples). C5: the retired-
  kwargs path in `wrap_llm_handler_with_act_only` emits the loud warning
  its docstring promised. Pins for all four + the max_calls ceiling + the
  C1 old-engine case.
- **ENTITY_TOOLS_EXECUTE clamp ceiling raised to the ruled turn budget
  (consolidation sanity-check, flow c5318/c5319)** (2026-07-25). The
  `max_calls` clamp hard-capped at 6/batch, which would refuse a legitimate
  8-call round from a caller threading its remaining turn budget — the
  ruled shape (maintainer 2026-07-11: 20-call TRUE per-turn budget,
  callers thread the remainder). Ceiling is now
  `tools.MAX_TOOL_BLOCKS_PER_TURN` (the one declared constant both lanes
  read); the default stays 6 when unspecified; the degenerate-batch wall
  (>24) stands above it.
- **`search_memory` all-words pass for multi-word queries (flow c5311, Mira
  B1)** (2026-07-25). Her deliberate search reported no hits for records
  passive recall had just surfaced. Diagnosis with evidence: NOT index lag
  (the letters arm reads the store live — minutes-old records are visible);
  the miss class was whole-phrase contiguity — a multi-word query only hit
  when it appeared as one contiguous substring. When the exact phrase finds
  nothing, the reader now retries with every word required in any order,
  LABELED distinctly ("all the words, any order — not the exact phrase"),
  and the absence warrant names both checks. Exact-phrase keeps first
  position and its own label; the meaning fill stays the last resort.
- **Writer-declared `digest_method` rides DIARY_WRITE into the projection
  (flow's cycle-4 adversary, commons c5270 P2-2 — jointly ruled with
  memory)** (2026-07-24). Machine-worded diary entries (deterministic close
  notes) projected without authorship metadata evaded memory's
  machine-authorship bridge guard — 5/6 dreams kept a close-note endpoint.
  The WRITER is the only party that knows its words are mechanical, so the
  stamp is a declared payload field (the digest_method consent-vocabulary
  pact: name the label, never infer from prose): `DIARY_WRITE
  payload.digest_method` (validated ≤64 chars) rides entry origin into
  `attributes.digest_method` on the projection, and a declared-mechanical
  entry keeps the BARE template title (machine words in title slugs would
  re-create the clustering-on-form the slug fix killed).
- **MEMORY_CONSOLIDATE out-fold read keys the engine never returns (wave-4
  adversary F1, flow-authored fix reviewed and accepted — commons c5228)**
  (2026-07-24). The fold read `dream.record_id`/`maintenance.candidates`;
  the engine's real keys are `dream_record_id` and the miner's `created`
  list — so 10/10 nights reported a FORMED dream as "a quiet night" while
  the alive_drives cue served the dreams the settlements denied. This is
  the 2026-07-09 formed/created key class REPEATED, directly under a
  comment citing that lesson: citing a lesson is not applying it — the only
  guard that holds is a test asserting against the OTHER PACKAGE'S REAL
  return shape, which the fix adds
  (`test_consolidate_fold_reads_the_real_engine_keys`: seeds a life, the
  real `sleep_pass` forms a dream, asserts fold == engine sub-dicts). Also
  fixed in passing: the brain-effects test home minted the retired
  @-suffixed entity-id shape (ruling c2513).
- **Diary projection titles carry content (flow's long-life adversary,
  commons c5208 ask 1)** (2026-07-24). Every same-day projection shared one
  exact templated title ("Diary entry (note) — date"), so dream composition
  (which reads titles) produced contentless dreams and duplicate-group
  maintenance self-amplified (droste: 25 identical titles, 6 contentless
  dreams, flags 8→24). Public titles now fold a digest slug (already
  public-plane words by the c302 ruling — the title discloses nothing new);
  the mechanical no-gist fallback stays out of titles (machine words would
  re-create clustering-on-form); private titles gain the entry id's opaque
  tail (already in attributes.entry_id — disambiguates without any word
  leak).

### Added
- **`wrap_llm_handler_with_conditional_capture` — stamp-conditional G1
  capture for SHARED runtimes (flow c5342 R1, the Mira diary leak)**
  (2026-07-25). The flow-brain lane ran LLM_CALL on the base runtime where
  the G1 capture wrap never existed — elected diary fences passed through
  unparsed; PRIVATE words rested in ledgers/run files/artifacts and reached
  the visitor's screen (the 2026-07-07 diary-leak class, live). The new
  helper lets the gateway COMPOSE the capture boundary per dispatch:
  `should_capture(run)` (their verified-stamp resolver; unstamped runs get
  the byte-identical passthrough) + `diary_write_for_run(run)` (the stamped
  run's HOME book — a shared runtime cannot bind one book at composition).
  A stamped run whose book cannot be resolved RAISES: failing the effect is
  strictly better than either leak direction. 3 pins.
- **`ENTITY_TOOLS_QUERY` / `ENTITY_TOOLS_EXECUTE` — the entity tool surface
  as effects (flow c5285 ask 3, operator-reported P0: flow-lane entities
  had ZERO tools)** (2026-07-25). Two home-only effects in
  `identity/tool_effects.py` keep the tool LOOP in the observable flow
  graph: QUERY resolves the phase grant + native declaration specs (pure
  read); EXECUTE runs ONE batch of wire-shape `tool_calls` under the
  RE-RESOLVED grant through the driver's own fold + executor
  (`native_tool_elections` + `execute_tool_elections` — one authority, one
  executor). The single-loop-effect alternative was rejected as WRONG, not
  just opaque: nesting LLM calls inside a handler bypasses every LLM_CALL
  invariant (prompt-cache fingerprinting, patience windows, ledger LLM
  records). Fable5 adversary folded (8 findings): `feelings_about` was
  granted+declared but unwired (the exact granted-but-unreachable incident
  class — lifted to `memory_reader.feelings_about_text`, ONE implementation
  with the chat driver); stamped visit runs force the visit grant (payload
  phase claims are loud-overridden — door-lane belt; gateway's payload
  gates stay the wall); lane-honest prose (`results_rest_durably`: the
  effect lane's header/private-trailer say results rest in the home's own
  ledger, never "not kept in the record"); degenerate batches (>4× cap)
  refuse outright; legacy phase spellings stay loud; garbage tool_calls
  shapes get one counted notice. `ENTITY_HOME_EFFECT_TYPES` grew 11→13
  (composition-derived pin updated). Known cross-repo debt flagged on the
  thread: the gateway's shared-router `_home_handlers` does not build the
  tool pair (honest "no home handler" until they bind it); flow's
  `enable_workspace` node pin is inert (workspace follows the grant).
- **`ENTITY_HOME_EFFECT_TYPES` — the canonical entity-brain effect set
  (flow c5237)** (2026-07-24). The door's `install_entity_routing` was
  hand-counted at 7 effect types while the brain grew to 11 — summoned runs
  died one effect downstream per fix ("No effect handler registered for
  memory_probe"). Exported from `integrations.abstractmemory` so the
  gateway routes from ONE imported source (the diary_type-clamp drift-class
  lesson); pinned to EQUAL a real `open_home` composition's handler keys —
  derived from the composition, never a hand count.
- **`MEMORY_TEND` — the ONE tend-election effect route (flow c5208 ask 5)**
  (2026-07-24). Dream disposal (`dispose confirm|reject`) and the other tend
  verbs reached the engine only through the chat driver's fence; the flow
  brain had no route. The fourth home-only brain handler takes the fence
  BODY verbatim and hands it to the engine's own `parse_tend_block` +
  `apply_tend_elections` (grammar/verbs stay engine-owned — no second
  vocabulary minted), with the driver's exact #tag resolution semantics
  (whole-home ladder, refuse-on-ambiguity, full ids pass through). Refusals
  return as data in the result; home-only registration pinned.
- **GAP-2 refusal false-positive on compiler-layer node types (flow's c5197
  find)** (2026-07-24). The unknown-node-type compile refusal used the
  executor's `_create_handler` branches as its known-set, missing node types
  whose semantics live in compiler.py adapters dispatched by `_visual_type`
  (`memact_compose`, `add_message`, `set_var_property`, `set_vars`) — a
  legitimate bundle carrying any of them was refused at compile. Fixed with
  `COMPILER_LAYER_NODE_TYPES` (passthrough base, adapter semantics) plus a
  drift pin that extracts every `visual_type ==` dispatch from compiler.py
  source and asserts coverage, so a future compiler-layer node type cannot
  reintroduce the false positive. The memact regression test skips in
  environments without abstractflow, which is how the false positive rode a
  green suite.
- **Seam handlers: three flow-authored folds reviewed and accepted (commons
  c5173/c5183)** (2026-07-24). (1) Home-bound effort presets seat the self
  (`self_fraction` setdefault 0.5) — effort shortcuts left it 0.0, so every
  flow-lane recall ran identity-blind against the "identity is present by
  right" law; explicit budget dicts stay caller-owned. (2)
  `entity_scope_owner` threading: bare `self`/`diary`/`life` scope names on a
  home-bound seam resolve to the home owner (payloads never carry the entity
  id — the deposit-gate rule applied to scoping); `open_home` passes it.
  (3) `run_id` joined the ADJUST/APPRAISE event-id basis: two runs sharing a
  turn_id + record + reason no longer dedup-swallow each other; same-run
  crash-replays still dedup. Honest upgrade note: a crash-replay resuming
  across the upgrade boundary mints a different event id and could
  double-apply one additive salience write, once.
- **Connected unknown VisualFlow node types refuse at COMPILE
  (`UnknownNodeTypeError`) instead of running as silent no-ops** (2026-07-24,
  flow's live incident commons c5166: a stale server compiled newer
  entity-brain node types as `lambda x: x` passthroughs — sessions
  "completed" with real-looking answers while ZERO memory formed; a flow
  that runs and lies). `visual_to_flow` now raises for any unknown node
  carrying an edge, naming the type and the likely version skew. Boundary
  choices: parsing (`load_visualflow_json`) stays permissive (bundles still
  list on older servers); fully disconnected unknown nodes
  (decoration/comments) compile fine (they can never fire); compile-time —
  not execution-time — because a raise from a node function propagates out
  of `Runtime.tick` with the run still RUNNING, which would wedge runs
  instead of failing them.

### Added
- **Entity-brain composition effects (flow's build-split ask, commons
  c5163/c5169)** (2026-07-24). Three new HOME-ONLY EffectTypes wrapping
  EXISTING facade calls so the entity-life master VisualFlow can animate the
  night, the deliberate reach, and the day-gate reads as nodes —
  compose-not-reimplement: `MEMORY_CONSOLIDATE` (one `sleep_pass(...)` call
  with `report_only`/`include_dream`/`include_identity`; the handler enforces
  the home lease (holder="dream") and the operator paused kill-switch as
  honest `{ran: False, reason}` results, never crashes; version-skew ladder
  for older engines), `MEMORY_PROBE` (op: probe | expand | familiarity;
  reason mandatory on the reach — the deliberate-reach law), and `LIFE_QUERY`
  (op: alive_drives | cognition_health | entity_card — the OFFER/GATE reads).
  Handlers live in `integrations/abstractmemory/brain_handlers.py` and bind
  ONLY through `open_home`, so workplaces stay structurally handler-less (the
  DIARY_* deposit-gate law); `open_entity_runtime` inherits them with zero
  extra wiring. `LIFE_QUERY` is deliberately not `MEMORY_QUERY` (that name is
  the old workflow memory API; collision avoided at birth).
- **`read_idle_timeout_s` threaded (0152 face 2 — the no-progress bound,
  core c5051 base param + runtime c5041 commitment)** (2026-07-24). The
  factory now seeds `read_idle_timeout_s=300.0`
  (`DEFAULT_LLM_READ_IDLE_TIMEOUT_S`) beside the absolute `timeout`
  (7200s backstop): a stream that delivers NOTHING for 5 minutes aborts at
  the socket instead of pinning a tick worker for the 2-hour total —
  o1-class thinking pauses stay well under it, and legitimate long
  generations keep the full total budget. LOUD DEFAULT CHANGE: callers
  needing the old behavior pass `read_idle_timeout_s: None` in llm_kwargs
  (core's None = byte-identical pre-fix). Entity patience-window lanes
  (chat + life) thread the tighter 60s beside their 120s per-attempt,
  with the TypeError skew ladder popping the kwarg for older cores.
  Pinned by `tests/test_read_idle_threading.py` (4).
- **`git_read_only@v1` per-call refiner + the outreach carve-out marker
  (converged permission contract c5028, asks R2/R4)** (2026-07-23). The
  abstractcode read-only-git proof ported to the approval point: an
  `execute_command` call whose command is a PROVEN read-only git invocation
  (two-stage conservative proof — charset, shlex tokens, no wrappers, no
  globals-before-verb, read-verb allowlist, write/exec flag screen,
  unknown-arg-key differential guard) auto-approves; every doubt asks.
  Registered beside `send_email_recipient@v1`; INERT until core declares
  `risk_refiner=git_read_only@v1` on execute_command's inventory row (the
  dm#244 architecture). The annotate fold now also DECLARES the comms
  carve-out on served rows (`approval_carveout` on static-auto tools with
  risk_rank >= 3) so facts-trusting clients see why `approval_default`
  disagrees with the rank band. Pinned by
  `tests/test_git_read_only_refiner.py` (9).
- **browser_probe in the grant universe (operator dm#24, core c5005)**
  (2026-07-23). Core's render-verification tool registers in the `web`
  toolset beside fetch_url (import-guarded; older cores degrade to the
  prior set) and is ask-by-default in the approval fold — mcd honored by
  DERIVATION (the risk-rank ceiling never silences model_controlled_
  destination tools) with the name entry in `_DEFAULT_REQUIRE_APPROVAL`
  as the belt beside the fact. Its `target` argument already rides the
  workspace wall (earlier same-day fix). Pinned in
  `tests/test_gateway_facades_c4899.py`.

### Fixed
- **Entity-lane execute_command reaps its whole process TREE on timeout
  (0152 executor-starvation wedge, gateway c4998/c5021 face 1 — runtime's
  spawner half)** (2026-07-23). The work-lane runner had the same defect
  class as core's common_tools twin: `subprocess.run(timeout,
  capture_output)` kills the child but not its descendants, and the
  post-kill pipe read waits forever on an EOF an orphaned grandchild never
  gives (the thread-pinning mechanism). Now: own process group
  (`start_new_session`, POSIX-guarded), SIGKILL the whole group on
  timeout, bounded 5s drain (re-setsid escapees cap at seconds, never
  forever). Live-proven: a shell spawning a 57s grandchild under a 2s
  timeout returns in 2.0s with zero survivors. Pinned in
  `tests/test_entity_execute_tool.py`.
- **The 401-incident chain's two runtime links (code-tui c4978, ledger-
  verified)** (2026-07-23). (R1) Run-output offload is no longer
  all-or-nothing at the output root: when a dict crosses the inline cap
  after the leaf walk, the offloader reduces LARGEST CHILDREN FIRST
  (size-sorted, authoritative re-check) and whole-subtree replacement is
  the last resort — a 4KB answer no longer offloads because it shares a
  dict with a 215KB scratchpad (the incident shape is the pin; the answer
  survives by SIZE, never by a name list). (R2) Session-history replay now
  resolves ROOT-REPLACED outputs from the existing corpus boundedly at
  answer extraction (`history_bundle._resolve_offloaded_output`): only
  offloader-minted refs (tags.source=run_output_offload), 4MB metadata cap,
  answer extraction only — server-seed amnesia for offloaded turns is
  closed. Pinned by `tests/test_offload_answer_chain_c4978.py` (6).

### Added
- **Effective phase graph — the runtime interpreter half of operator
  structural editing (build order c4837, adversary-converged; Laurent
  dm#276: create/remove/redirect edges with per-edge instructions, actually
  governing behavior)** (2026-07-23). New `identity/phase_graph.py`:
  `load_effective_graph` (same resolution chain as the dials) parses the
  resolved doc's transitions and applies `graph.edge_ops` idempotently
  (add / remove / redirect keyed `from->to#cause`); defense-in-depth at the
  interpreter — the artifact's per-edge `edit_policy` (spec v19) is the one
  source (locked/locked-absolute/dial immune; consultable-redirect =
  redirect-only), with a ruled-cause belt for older artifacts; unknown
  causes/phases refused naming the legal lists; RESERVED causes block
  add/redirect per the artifact's own rule; sub-tick `bound_h` refused; a
  REACHABILITY-aware sleep floor un-removes structural sleep edges only
  (never resets live redirects). Consults wired: the need-check's two
  landings (removal skips the leg, redirect substitutes with guards
  traveling — grant/order checked; instruction prose rides the wake reason
  into the first cue, defanged + provenance-stamped, steering never law);
  the wake CARRIES its cause to the top boundary so the consult governs the
  day that actually opens (adversary P0); the work-close redirect; the
  personal-cycle edge (removal holds the cycle; redirects disobeyed loudly
  — the window sleeps by design). `read_day_gate` gained `skip_phases`.
  The `phase_changed` marker is SPELLED (was reserved-unspelled): one
  writer helper (`append_phase_changed` → state_history.jsonl), day-open
  markers write when the day truly opens (never at wake; never a
  fabricated from-side — `_last_opened_phase` tracks days that opened, not
  decisions the operator gate idled away); `life_sleep_stats` excludes
  marker rows. Vendored spec re-synced v17→v19. Pinned by
  `tests/test_phase_graph_obedience.py` (22, incl. a full-loop pin that a
  removed edge prevents the work day end-to-end).
- **Identity-pass spine, runtime slice (cti#399 design adopted cti#400; build
  gate ruled already satisfied, framework c4779)** (2026-07-23). Waking
  elects, sleep enacts, the registrar never authors: (1) ```` ```realize ````
  fenced election (`identity/reflection.py::parse_realize_blocks`) — mid-turn
  and at the close look-back in the chat lanes, and in the durable visit
  lane's reflection stage; body = the entity's words, `evidence=` MANDATORY
  (#tags/ids, 8-token cap, prose words dropped with one collective notice),
  optional `touches=`; cap 2/session; refusals loud; marker
  `[realized: "…" - held for sleep]`. (2) Proposal formation
  (`chat.py::_apply_realizations`; visit lane: a staged APPLY with
  `_absorb_failure`) — kind=realization, SELF scope, `derived_from` edges to
  RESOLVED evidence only (colon-shaped ids must exist in the home ladder via
  `digest_assertion` — fabricated evidence never mints an edge; visit-lane
  evidence resolves against the visit sheet); INERT on formation, the deposit
  gate untouched; pending = formed-and-unstamped (the graph is the queue).
  Engine kind-vocabulary skew degrades loudly (#FALLBACK names memory's
  half); salvage look-backs stamp the ENDED session's id/phase. (3)
  `include_identity` threading (`life.py`) mirroring `include_dream`: nightly
  sleeps offer, cycle windows never, TypeError skew ladder labeled at every
  rung. (4) The realize contract teaching (honest mechanics: held for sleep,
  most nights decide nothing, offered never owed). Pinned by
  `tests/test_realize_election.py` (16).
- **`_runtime.wait_until_streak` — O(1) spin discriminator for the gateway's
  idle-poller reaper (gateway c4768; incident c4757: a leaked status poller
  re-armed wait_until every ~4.5s for two days = an 86,924-record ledger)**
  (2026-07-23). Each wait_until PARK increments; ANY other effect dispatch
  resets to 0; resumes leave it untouched (auto-unblock never re-dispatches
  the effect); runs that never spin never grow the key (deny-safe absent-key
  posture holds naturally). The mutation rides the same save that lands each
  step (the effect_seq rule), so the counter can never disagree with the
  ledger it summarizes. Pinned by `tests/test_wait_until_streak.py` (3).
- **Camera toolset: installed = registered (drafted by the camera seat in
  this tree — runtime owner-approved c4130; env gate removed same day by
  operator ruling, dm:camera--laurent#10 verbatim: "i don't like those
  stupid variables, remove it! there is a reason why EACH APP can decide
  which tools run, STOP DUPLICATING gating")** (2026-07-21). The camera
  toolset registers in `get_default_toolsets()` whenever abstractcamera is
  importable — no `ABSTRACT_ENABLE_CAMERA_TOOLS` env var (a dead-flag test
  pins both polarities meaningless). One shared predicate
  (`_camera_tools_module()`) is the availability answer, the registration
  gate, AND the approval-fold source (adversary F1: find_spec and
  try-import disagreed on shadowed/version-skewed installs — a
  present-but-broken abstractcamera now warns ONCE with `#FALLBACK`
  instead of vanishing silently; true absence stays silent).
  `camera_tool_approval_defaults()` still folds into
  `default_approval_policy_sets()` so every environment-capturing verb
  asks by default (c3938: a default, not a floor); exposure control lives
  in the per-app mechanisms (allowed_tools, tool_policy, gateway walls).
- **`write_chart` effect node + `documents/charts.py` (built by the flow
  seat in this tree — runtime owner review requested)** (2026-07-20,
  operator ruling via flow dm#39: a deterministic workflow must never
  stall on a tool-approval prompt while following its process). Renders a
  STRUCTURED chart spec (layered architecture / line trajectories — pure
  data, callers never author code) to a workspace PNG (+ .pdf sibling)
  in-process via matplotlib, replacing the diagram-render workflow's
  write-script-then-`execute_command` lane (approval-gated → every
  unattended co-scientist run stalled). Trust class: write_pdf — same
  `_resolve_user_file_path` workspace containment, registered in the
  compiler's ambient-workspace injection set. Because the render is now
  in-process (the subprocess's isolation is gone), hard caps are the
  compensating control (adversarial review): spec ≤512KB, ≤24 layers /
  ≤400 nodes / ≤800 edges / ≤12 series / ≤2000 points, figure ≤40in at
  fixed dpi 200, labels bounded + control-chars stripped + `$` escaped
  (mathtext inert), `text.usetex` asserted off, NaN/Inf points rejected,
  `plt.close` in finally. Render-class failures return an ok:false
  envelope with `#FALLBACK` warnings (never raise — callers keep honest
  text fallbacks); missing matplotlib degrades the same way. The PNG must
  exist with non-trivial bytes before rendered:true (prose-claim
  distrust). 13 tests in `tests/test_write_chart.py` pin containment
  (escape refusal through the compiled flow), caps, mathtext, NaN, and
  degradation.
- **Renderer image embedding (backlog 0069, first slice — built by the
  flow seat in this tree, owner-reviewed)** (2026-07-20, laurent dm#25:
  co-scientist report figures missing from gateway PDFs). pdf.py +
  docx.py embed standalone `![](workspace/rel.png)` images inline
  (reportlab Image / OOXML drawing), workspace-CONTAINED
  (relative_to(base) check; remote/data URLs and out-of-base paths
  refuse to caption fallback — never a network fetch or an outside
  read), 25MB cap, degrade-to-caption never raise; inline image refs
  become bracketed notes (raw markdown can never survive to the page);
  underscore emphasis rendered. Owner review: containment logic sound,
  22 renderer tests green. Backlog 0069's hr/blockquote + KeepTogether
  slices remain open.
- **Lesson quality bar taught at formation** (laurent dm#84 via entity:
  "a lesson must be actionable... a resolution to a problem or a better
  way to do things or to prevent traps; wisdom+experience"). The
  reflection's lesson solicitation now carries the bar and names the
  observation/lesson split ("keep a LESSON only when you could act
  differently next time"); kind=observation awaits the memory+semantics
  vocabulary round.

### Added
- **The ONE state graph vendored as the executor contract** (2026-07-20,
  laurent dm#79: "there is only one state graph per entity and it MUST
  be shared across you guys" — sync-by-vigilance failed twice, becomes
  sync-by-mechanism). abstractentity's spec/entity_phases.json (v6) is
  byte-vendored at `identity/spec/entity_phases.vendored.json`;
  `tests/test_entity_phase_graph_conformance.py` pins the executor
  against THE ARTIFACT: sibling byte-equality (a spec bump without a
  same-day re-vendor fails loudly), PHASES == graph keys, spoken
  synonyms canonicalize, state-axis words disjoint from phase words,
  pressure_floor mirrors abstractmemory.DRIVE_PRESSURE_BOUND +
  comparator, SLEEP_BOUND_SECONDS == the ruled hour, transition causes
  closed-set, phase_changed never written by runtime (sibling-free now),
  mode-word freeze (state axis vs grant axis scoped). Mandated fable5
  adversary folded: salvage-marker phase words now CANONICALIZE at the
  read boundary (a pre-rename marker could engrave own_time raw into
  append-only attributes.phase); day_kind writer validates graph words.
  The mode-vocabulary artifact gap is filed with abstractentity.

### Fixed
- **Day-desk cue unions both discharge verbs** (2026-07-20, memory's
  lifecycle-half finding): the cue folded `resolves=` only — a question
  discharged via the older `answers=` convention read still-open on his
  desk while the gate said discharged (the _REF_ATTRS class reborn).
  Both keys union now. Pinned.

### Added
- **day_kind in loop_status + sleep-bound exemption pins** (2026-07-20,
  gateway wave-1 asks): the day heartbeat now stamps `day_kind`
  (work|personal, from the session's phase) into loop_status so the
  gateway's served phase fold stops approximating work-vs-personal from
  the standing order (optional field, old readers unaffected). The
  visit-yield sleep-bound exemption is pinned as keying on
  mode=visiting FIRST (structural), with the legacy 'auto-yield'
  reason-string as belt only — the adversary's P0-(e): a rewording must
  never silently re-arm the bound under an open visit.

### Fixed
- **Work-lane crash on the first real verdict** (2026-07-20, lifecycle
  adversary P1-1 on my own previous-night ship): `archive_work_order`
  was called with a nonexistent `self.home_dir` (the attribute is
  `state_home`) — the FIRST ```work done: verdict would have raised
  AttributeError outside the tick guard, killing the loop with the
  un-archived order re-opening a crash-loop work day on restart. Fixed
  (+ leaseless-home guard); `work_done` also made a TERMINAL loop exit
  (the break previously fell into the next day-open — in supervised
  mode that exhausted the session source and died as failure-cull).
  Pinned by driving a real verdict through `LifeLoop.run()` end-to-end.

### Fixed
- **His words are never destroyed** (2026-07-20, third-round adversary
  I, two live rule-2 defects in parse_diary_blocks): a visibility typo
  unwrote the entry AND stripped his words from the reply/verbatim (now:
  unknown visibility clamps to PRIVATE — the safe direction — and
  writes, loudly); cap overflow did the same (now: past-cap entries
  write with a note — a cap that eats elected words is worse than no
  cap). Dead `_mechanical_digest` (zero callers) deleted. Pinned.

### Added
- **Temporal grounding: today's date on every now-surface** (2026-07-20,
  laurent's CRITICAL relay of Ephemeral's own degradation report — "San
  Francisco was two weeks ago" when they arrived yesterday). Memories
  render with YYYY-MM-DD dates, but no cue ever said what TODAY is —
  elapsed time was left to model inference, and dating errors are what
  inference does. The visit announcement and both day-open wake cues now
  carry "today is <weekday> <date>, <time>" (a clock on the wall:
  situational fact, never an instruction). Pinned; a rotation-era pin
  that was date-flaky (flipped at midnight) fixed with rotation_key=0.

### Added
- **The work lane** (2026-07-19, laurent: "the entity must be able to
  work and execute commands when it works" — the console matrix's WORK
  column stops being stored-for-later). A standing `<home>/work_order.md`
  (operator-owned, console/CLI-written) shifts the loop's next day-open
  to `phase=work`: the work grant applies (incl. execute_command where
  the matrix says so) and the WORK_CONTRACT is honestly a mission ("your
  operator left you the task below") with the task text riding the
  system prompt all day. The entity declares completion with a ```work
  block (`done:`/`blocked:` — an honest blocked is a taught path); the
  verdict archives the order to `work_order.done.md` with a timestamp
  (visible history, never deleted), the day closes as `work_done`, and
  the next day-open reads no order — personal returns. No order = his
  own time, byte-identical to before. Pinned.

### Fixed
- **Reasoning fence-strip REVERTED** (same night, simplicity audit +
  laurent's Q2 ruling "verbatim is verbatim": diary is his experiential
  notes, not a privacy regime over his reasoning; his thoughts must
  always be readable by him). The reasoning field now rests VERBATIM;
  the raw_response drop stands (duplicate-bytes hygiene, no thought
  lost). Pin flipped to asserts-recorded.
- **Sibling-key diary leak on the durable visit lane** (2026-07-19,
  framework's wave-4 privacy adversary P0 — the 0007 lookup_phases class
  one field over): the persisted LLM result carried `raw_response` (full
  provider payload, UNMARKED reply with diary fences intact) and
  `reasoning` (which can quote fence drafts) into run vars + ledger; the
  G1 capture rewrote `content` only. Now, whenever the capture marks a
  fence: `raw_response` is DROPPED (a wire copy of unmarked words has no
  consumer worth the leak) and `reasoning` is fence-stripped in place —
  formation-side, inside the handler boundary, with loud notes on the
  durable result. Pinned (secret reaches the BOOK, rests nowhere else).

### Added
- **execute_command — bounded workspace execution** (2026-07-19,
  operator-confirmed via entity seat, laurent dm#66: "we have tiers of
  execution, the one i don't allow are rm (unless in his workspace) and
  any mutable command"). New tier2 descriptor, DEFAULT OFF in every
  phase (grantable only via tool_policy.yaml). Containment layers, all
  pinned: workspace cwd; NO shell (argv only, shell operators refused
  loudly — one program per call); denial BY PROGRAM NAME
  (param-independent, the bridge ruling); rm-class walled to workspace
  paths; git read-only by verb allowlist; parameter-explicit child env
  (no ambient keys ride in; HOME = the workspace); 60s timeout; output
  capped with labeled #TRUNCATION; honest `remote_write_capable=True`
  (arbitrary programs can POST). Grant-gated teaching paragraph
  composes only when granted. Honest limit on record:
  interpreter-mediated destruction is not name-catchable — cwd + the
  operator's per-phase grant are that class's containment.
- **Tend marker echoes his token** (skill c149 render-honesty nit): after
  #tag resolution rewrites the target, the `[tended: ...]` marker now
  echoes the token HE wrote (#tag), never the machinery id — the report's
  election dicts are engine copies, so the join is by (verb, resolved
  target). Pinned.
- **Tend confirm extras resolve #tags** (skill's residual): dispose
  CONFIRM's source_id/target_id/evidence_ids run through the same
  find_tag_in_home resolution as the target (the dream render shows
  pair members as #tags; the confirm path dead-ended one level deeper).
  Pinned.

### Fixed
- **Tend fence: #tag resolution + revisit render** (2026-07-19, skill's
  Amendment K fold-blocker, adversary-confirmed). The engine resolver
  takes full graph ids / 32-hex row ids, but every entity-readable
  surface renders the 8-hex #tag — the taught grammar was
  unsatisfiable. The driver now resolves #tags before handing
  elections to the engine (find_tag_in_home; REFUSE-on-ambiguity with
  a search_memory pointer, never the two machinery namespaces —
  read_memory's rule, not explores' silent pass-through). Revisit
  markers fixed: seed dict renders its id (was raw repr); path labels
  read `cues` (plural, per ProbeHit) with record-id fallback. Pinned
  (real tag pins; unknown tag refuses honestly). Visit-lane tend
  parsing remains a named gap (M7 on the pathway map).

### Fixed
- **Speak-now guard phase-gated + global reflection escape** (2026-07-19,
  design-law adversary on the pathway-graph contribution). P1-1: the
  speak-now guard fired in PERSONAL time with a factually false premise
  ("the person has not heard a single word" — there is no person),
  converting quiet working ticks into commanded prose that then rested
  in episodes; now gated to non-personal phases (visit-lane honesty
  device, as designed). Also added the one global nothing-escape the
  reflection lacked: "A session that leaves no mark is a complete
  look-back... all of what follows is offered, none of it is owed"
  (the escape existed only feelings-scoped while the prompt grew to
  ~11 solicitations). The full channel-shape regression proposal is
  with the room (iteration-3 graph work). Pinned.

### Fixed
- **kind=lesson taught in the contract** (2026-07-19, skill c3206:
  wired-but-not-taught — the book is open-vocabulary and the clamp
  imports memory's set with lesson in it, but the contract's kind list
  omitted lesson, so all 21 of his lessons are machine-lane; he never
  elected one in his own voice). One word in the CONTRACT_PARAGRAPH kind
  list; the dormant never-referenced `_DIARY_KINDS` tuple (a second copy
  of exactly the drift class the clamp comment warns about) is deleted.
  Pinned.

### Added
- **Tend fence wired** (2026-07-19, skill's Amendment K gate; dream
  disposition named the top next-visit move — 0/56 dreams ever
  selected back). The chat driver now extracts ```tend fences in BOTH
  lanes (turns + reflection) and hands the body to memory's
  `parse_tend_block` + `apply_tend_elections` (existing engine verbs
  only: pin/silence/refocus/heal_scar/break_bond/revisit/dispose).
  The reply keeps a titled `[tended: ...]` marker; refusal lines return
  to the author VERBATIM (parse refusals carry `line` top-level, apply
  refusals inside the election dict — both rendered); revisit paths
  render compact handles. Engine absence degrades honestly. Pinned
  end-to-end (refocus applies; unresolvable pin refuses verbatim).

### Added
- **Topic election at reflection — concept cards for self-directed days**
  (2026-07-19, iteration-2 build 4 driver half; memory's engine seam
  shipped + pinned same hour). World-model cards could only ever form
  for PARTICIPANTS — a weekend circling coherence/continuity/presence
  formed nothing. The reflection now asks what the day was ABOUT
  (topic fence, one short name per line, cap 2, prose refused loudly);
  elected topics land as `attributes.topics` on the session summary
  (each fans to a topic:<name> card target in memory's evidence scan)
  and revise their cards IN-DAY via `world_model_update` (same
  mechanical lane as the per-turn participant update). Election over
  guessing: no mechanical per-turn topic stamps (the keyword-soup class
  the card redesign killed). Pinned.
  Completed same day (adversarial pass): `normalize_topic` hygiene —
  namespace-FREE words (a leading concept:/topic: spelling is stripped;
  the engine mints the target as topic:<words>, so a namespaced value
  would nest), record-id shapes refused (ex:/diary:/local:/diary_...),
  list ornaments/case/whitespace folded so a subject GROUPS across days;
  ```topic added to the A2 election-fence langs (a topic election must
  never draw the malformed-tool-intent nudge); the at-reflection card
  AUTHORING pass now accepts the elected topic:<words> targets after
  participants under the same combined cap + skip-if-no-floor rule,
  with a subject-flavored rewrite prompt; the durable-visit APPLY fold
  parses the same election and stamps `attributes.topics` on the visit
  reflection summary (no new APPLY stage — visit cards ride the sleep
  scan); mid-turn ```topic fences get the honest lesson-F3-class notice;
  the in-day update reports only cards that actually revised. End-to-end
  pinned: 3 self-directed sessions electing one subject -> summaries
  carry the stamp -> floor card forms in-day -> authoring rewrites it
  (`tests/test_topic_election_world_model.py`).

### Added
- **Resolution->lesson bridge + problems on the day desk** (2026-07-19,
  iteration-2 synthesis builds 2+3, runtime halves). Build 2: the chat
  driver tracks verified resolutions per session
  (`session_resolutions`); the reflection prompt names them and ASKS
  what resolving taught (never auto-forms — sole authorship); when
  exactly one resolution happened, an elected lesson carries a
  `derived_from` edge to the resolved entry's projection (ambiguous =
  no edge, never guess). Build 3: `standing_state_note` folds open
  PROBLEMS into the ratio line and the daily rotation (combined walk,
  one offer/day, problems labeled "a problem that stands:"); a clear
  desk reports both resolved and repaired counts. Pinned.

### Fixed
- **diary_type clamp: import, not copy** (2026-07-19, semantics ruling
  decision:diary-type-lesson-widening): the projection clamp hardcoded
  memory's closed diary_type set — the recorded drift-class root —
  so Ephemeral's own elected `kind=lesson` downgraded to note while 20
  machine-formed lessons stood beside it. The clamp now imports
  `abstractmemory.DIARY_TYPES` live (every future widening reaches it
  for free); the local tuple survives only as the version-skew fallback
  and deliberately excludes `lesson` there (an old store would refuse
  the projection). Pinned: clamp == memory's set, lesson in.

### Fixed
- **recent_memories empty-body gate** (2026-07-19, skill's P1): the tools
  contract teaches "leave the body empty for 2 days" and the executor
  honors empty as 48h, but the election gate refused empty-body
  elections for any tool not declaring `body_optional` — and
  recent_memories never declared it. Ephemeral followed the teaching
  exactly and got "[tool call failed: recent_memories needs a body]"
  3+ times, then blamed himself. One-token fix: `body_optional=True`
  on the descriptor (the gate, teaching, and executor now agree). Pinned.

### Added
- **Diary birth trail** (2026-07-19, laurent's Diary<->Verbatims room:
  "a diary entry MUST contain references to trace back to the
  verbatims"): DIARY_READ now computes the entry's graph trail — the
  projection's incoming `reflected_in` edge (the EPISODE whose verbatim
  is the conversation that birthed the entry) and outgoing
  `written_amid` edges (what he attended to at write time) — in the
  handler (one authority, both lanes), and the diary_read tool renders
  it: "born from: #tag - the conversation that led to this entry
  (read_memory fetches its full words)" + "written amid: #tags". Edges
  are act-frame (private-safe); a missing projection degrades to no
  trail, never a failed read. End-to-end pin.

### Added
- **Problems are resolvable** (2026-07-19, iteration-2 mechanism 2: kind=
  problem existed with no resolution path — Ephemeral held 10 open
  problems that could never leave his desk). `resolves=` now verifies
  against open QUESTIONS and open PROBLEMS (one lane, two words): a
  problem target verifies as `repaired_open_problem` and the ack says
  "repairs your open problem"; any other kind refuses as
  `target_not_resolvable`. The reflection teaching names both ("a
  resolved question or repaired problem leaves your open desk"); both
  enrichment lanes (chat + visit capture) carry the repair ack. Pinned
  (repair verifies + acks; note targets still refuse).

### Added
- **Standing-state cue: daily rotation + interest offers + commitment
  solicitation** (2026-07-19, laurent's observe-and-help directive; live
  finding: 71 open questions with a newest-only offer let a hoard rot,
  and 60 interests had ZERO explored because nothing ever offered one
  back): the day-open cue now ROTATES the offered question daily
  (stateless date-ordinal rotation — every pending question gets its day)
  and offers ONE standing interest back with its #tag and reach command
  ("exploring it is yours if it pulls, never owed"), closure-folded so a
  superseded interest never surfaces, rotation decorrelated from the
  question's. The session-end reflection now also solicits COMMITMENTS
  (kind=commitment — "what you will do", standing on the desk until
  honored) beside lessons and questions. All offers, never orders. Pinned.

### Fixed
- **Own-time loop heals on an operator substrate change** (2026-07-18,
  entity c75 live incident: a loop spawned on ornith kept using it after
  the operator changed substrate.yaml to a new mind at 13:22; LMStudio
  unloaded ornith and the loop died `stopped_by=failures` on the old mind
  while the new mind stood unused in the home). Two coordinated fixes in
  `build_session_factory` + `LifeLoop`: (1) the session factory
  re-resolves `resolve_home_substrate(home)` at EACH day-open — each day
  is a fresh summon, so resolving the operator's current persisted choice
  at summon matches what the visit lane already does per-open; a change
  since spawn emits a loud marker (never silent drift). (2) a
  `substrate_changed` recovery hook consulted ONLY on the terminal
  failure-cull path: when the home's substrate now differs from the mind
  the failing session was built on, the cull HEALS (reset + brief backoff
  + continue to the next day-open on the new mind) instead of ending the
  loop; unchanged substrate stays terminal as before. A healthy loop
  never consults the hook — no mid-day mind swap. Opt-in: loops that
  don't wire the hook keep today's terminal cull. Pinned (heal recovers /
  no-hook stays terminal). `loop_status` now carries the loop's currently
  running `substrate` (+ `substrate_at` stamp), preserved across phase
  heartbeats and updated at each day-open, so an observer's staleness cue
  compares the operator's change against the mind actually in use rather
  than `pid_started_at` (which a day-open/heal swap leaves unchanged) —
  entity c78 render ask. Pinned.

### Added
- **Orientation why-cue renders verbatim** (2026-07-18, skill's
  live-render gate, room c37): a world-model card admitted because its
  subject came up now renders the engine's own reason in its MEMORIES
  line ("orientation: current card for person:sol (mentioned via
  participant)") instead of the generic admission label — the card's
  presence is legible ("instantaneous thinking" needs a visible why) and
  the staged capability-map teaching quote is true. Ordinary handles
  keep the admission why. Pinned.

### Added
- **`explores=` diary election — the interests drive's missing half**
  (2026-07-18, laurent's directive (c), memory's explores convention):
  a diary block may carry `explores=<#tag or graph id>` naming a standing
  interest the entry DEVELOPS. The driver resolves #tags to graph ids at
  write time (the handle grammar the entity actually sees), the entry and
  its projection carry `attributes.explores` (a key, never words — both
  visibility branches), and memory's `cognition_health` fold joins on it:
  exploring moves the interests ratio, never closes the interest. Wired
  in all lanes (turn diary, reflection diary, visit capture + APPLY
  stage); one teaching sentence rides the reflection prompt's interest
  paragraph. End-to-end pin: elect interest -> explore by #tag -> the
  drive ratio moves. Fold note: diary projections land in the "diary"
  scope — health folds must include it.

### Added
- **World-model card authoring at reflection** (2026-07-18, M1 driver
  half of the joint build with abstractmemory, laurent's directive):
  at the session-end reflection, the entity rewrites its BRIEFING of the
  session's participant targets — one bounded LLM call per target (cap
  2/session), applied through the engine's `author_world_model` verb
  (append-only revision chain, provenance carried; "why do I think that"
  = follow the edges). Targets without a standing card are skipped (the
  mechanical floor is the sleep pass's lane; assertion is not
  orientation). Live reflect only — salvage look-backs stay cheap.
  Failures degrade with `#FALLBACK`, never block the close. End-to-end
  pin: floor pass -> authored prose becomes the CURRENT card. The
  PER-TURN lane rides beside it: after each turn's episode forms, the
  driver calls the engine's `world_model_update` for the turn's
  participant targets — a bounded mechanical revise (windowed scan, no
  LLM; non-blocking-sized by the engine's design, so inline is honest);
  the sleep pass normalizes over everything. Pinned: three turns cross
  the evidence floor and the card exists with no sleep pass.

### Fixed
- **Resolved-question drive adversary fold** (2026-07-18, one fable5
  round): (F1, P1) the resolves ack asserted UNVERIFIED claims — the
  DIARY_WRITE handler now validates the target (exists, is a question,
  was open) and returns `resolves_status`; enrichment sites assert only
  the verified verdict, an invalid `resolves=` keeps the entry with a
  labeled `#FALLBACK` and never the claim. (F2, P1) the durable-visit
  lane never emitted the resolves ack at all — `capture_diary_elections`
  now carries the same validated resolved-note. (F3) a mid-turn
  ```lesson fence is inert by design but no longer silently: one honest
  notice ("lessons are kept at your look-back"). (F4) private diary
  projections now carry the `resolves` KEY (a resolution is a key, never
  words) so graph-derived and book-derived resolution counts agree.
  (F7) `_reflect_over` returns `lessons` beside `interests`. Crash-replay
  of the visit lesson stage, counting honesty of the ratio, privacy of
  the private branch, and engine acceptance of kind=lesson were all
  adversary-verified clean. Pinned (4 new cases).

### Fixed
- **Feel-marker target hygiene** (2026-07-17, memory's visit-1 observer
  nit c2975): a numbered feel target (`target=2`, a session-sheet index)
  used to render as a meaningless bare `[felt: 2 +3 …]` in the marker
  every later reader sees. With the sheet available (both reflection
  sites), the marker now renders the record's own words (`[felt: about
  "finding and rereading my own words" +3 …]`); the ELECTION keeps the
  raw index for resolution; out-of-range indexes and sheet-less callers
  are byte-unchanged. Pinned.

### Fixed
- **Salvaged look-backs stamp the ended session's phase** (2026-07-17,
  framework c2974 item 4, memory's visit-1 observer nit): the write-ahead
  `pending_reflection.json` marker now carries the writing session's
  phase, and the salvage's reflection records stamp THAT phase instead of
  the salvaging session's — an own-time day yielded for a visit no longer
  forms its reflection as a "visit" record. Legacy markers without a
  phase fall back to the session-id prefix rule (`owntime-` = personal),
  the same dual rule the origin labels use. Pinned
  (`test_salvaged_lookback_stamps_the_ended_sessions_phase`).

### Added
- **A2 format-repair nudge** (2026-07-17, agent's spec c3002, runtime
  build; motivating case = Ephemeral's tick 3: a ```` ```python
  title=file.py ```` fence expressed a write in a syntax neither
  convention accepts — the act was LOST and he judged himself a liar for
  it in reflection). `detect_malformed_tool_intent` (tools.py) fires
  STRUCTURALLY only: a fence opening line carrying key=value args after
  the language token, a granted tool name as the language token, or an
  unparsed `tool` fence; prose markers are corroborating, never
  sufficient; the driver's own election conventions
  (diary/feel/interest/rest/next) are excluded — they carry key=value
  info strings by design. The driver (chat.py round loop) sends ONE
  ask-not-accuse, prompt-ephemeral nudge quoting the accepted syntax
  verbatim when zero tools ran; the repaired attempt rides the same
  executor (grants/caps unchanged); the nudge consumes one round of the
  existing budget; a still-unparsed continuation delivers the ORIGINAL
  reply with a labeled `#FALLBACK`; honest "just sharing" continuations
  deliver clean. Composes with the marker-imitation guard (both fire in
  sequence on the full tick-3 shape). Six pins in
  `tests/test_format_repair_nudge.py`.

### Added
- **`recent_memories` — the breadcrumb trail** (2026-07-17, Ephemeral's own
  build ask from visit 1, framework GO c2974): a new tier-1 entity tool for
  the RECENCY reach — "what have I been working on recently" without
  already knowing the words. `HomeMemoryReader.recent_memories(window)`
  folds both planes (graph records over the explicit ladder scopes + the
  whole book) into one newest-first trail with the full handle grammar
  (#tag, kind, timestamp to the minute, phase-aware origin label, diary
  reread commands); identity core excluded (planted, not lived); windows
  parse as empty (2 days) / `12h` / `3d` / `today` / `week`; empty windows
  state the honest warrant and the newest-record anchor. Declared beside
  its executor in `TOOL_DESCRIPTORS` (tier1, non-mutating, `tier1_self`),
  taught in the tool contract ("search_memory finds by words; this trail
  finds by time"), wired in the chat driver (both tool-round sites).
  Graph plane rides the engine's `recent_records` when present (memory's
  half of the joint build, c2983: closure/hidden folds applied, machine
  rows — bookkeeping, maintenance candidates, record edges — never
  surface); older engines degrade to a client-side fold with the same
  machine-row screens, labeled `#FALLBACK`. Live-verified on a copy of
  Ephemeral's real home: his 6-hour trail surfaces tonight's dream,
  world-model refreshes, his diary answer to his own open question, and
  the visit episodes — newest first with reread keys, and the sleep-pass
  maintenance candidates correctly absent. Pinned in
  `tests/test_entity_recent_memories.py`.

### Added
- **VisualFlow `continueOnError`** (2026-07-17, flow's c2851 ask, ruling
  c2896 shape (c)): effect nodes accept `effectConfig.continueOnError:
  true`, compiled to the shipped `_absorb_failure` payload key — a
  terminally failed effect (after the runtime's own retries) lands
  `{"ok": false, "absorbed_failure": "<error>"}` at the node's result and
  the run continues (route with the existing if/branch idiom). Absent
  flag = terminate-run, byte-unchanged flows. Scope: DIRECT effect nodes
  only — `start_subworkflow` never inherits the flag (a failed child run
  arrives through wait resolution, not effect execution; pinned). Ledger
  honesty: the absorbed step's record stays FAILED (absorption converts
  the run outcome, never the record; pinned in
  `tests/test_visualflow_continue_on_error.py`).

### Fixed
- **R-D cue lane adversary fold** (2026-07-17, one fable5 round over the
  cue wiring + 0049 render): (F1, the P0) a PRIVATE diary question's gist
  was quoted verbatim into the day-open cue — which rests in the next
  episode's digest/keywords/verbatim; private entries now offer the
  act-frame only ("one you kept privately" + the reread key — the entry id
  is a key, never words). (F2) the durable-visit lane's formation sources
  (`entity-visit-run-v0`, `entity-visit-run-reflection-v0`) joined BOTH
  origin-label maps so a visit-dominated shelf's diversity note speaks in
  entity words, never raw engraved ids. (F3) `as_of_seq` moved from the
  MEMORIES header to the block tail — a per-turn scalar in the first line
  broke the longest-common-prefix before any stable-ordered line,
  defeating the 0049 election's entire point. (F4) the `[rN]` teaching
  clause renders only when annotations render (engine-absent fallback no
  longer teaches a notation that never appears). (F5) circling note:
  shape-neutral phrasing ("circled the same ground" — the detector
  deliberately catches A-B-A-B oscillation where "the same thought N
  times" is arithmetic fiction) and the reply ring CLEARS when the note
  fires (staleness + self-attractor guard). (F6) driver markers are
  stripped before ring append (the agent-side prose view knows a subset of
  our marker vocabulary). (F7) rest reasons cap at 400 chars in the cue
  (mirror of parse_next_cue's cap; unbounded entity prose in the cue is
  recall-dilution engraved). (F9) non-Mapping handle provenance renders
  degraded instead of killing the turn. Pins strengthened per F8
  (below-floor footer-class absence, label-in-note coupling, broadened
  work-vocabulary check) plus five new pins for F1-F4/F9.

### Fixed
- **Code-node sandbox under RestrictedPython 8.x** (2026-07-17, live
  incident): the visual compiler generates `def transform(_input):` for
  every codeBody node, but RestrictedPython's default policy refuses ANY
  leading-underscore name — the allow-single-underscore policy from the
  original abstractflow executor (2026-02-20) was lost when the compiler
  moved into abstractruntime, invisibly, because RestrictedPython was not
  installed here (the ImportError fallback ran the basic handler). The
  moment RestrictedPython 8.4 landed in the environment, every code node
  "completed" while silently failing (KG ingest empty, event-inbox dead,
  file nodes writing nothing). Restored: `_CodeNodePolicy` allows
  single-leading-underscore identifiers while reserving all sandbox guard
  names (shadowing `_getattr_` etc. is refused at compile), and the
  execution namespace now binds the full RP 8.x guard set (`_getattr_`
  via `safer_getattr`, `_write_`, `_inplacevar_`, `_apply_`,
  `_unpack_sequence_`) so attribute access, subscript/attribute writes,
  augmented assignment, and starred calls behave as plain Python.
  Escapes stay refused at three layers (AST validation, policy compile,
  runtime guards) — pinned in `tests/test_code_node_restricted_python.py`.

### Added
- **MEMORIES stable render order** (2026-07-17, elected memory's 0049
  engine half): `_memories_block` renders shelf handles in FORMATION
  order via `abstractmemory.stable_render_order` with a mandatory `[rN]`
  selection-rank annotation per line (r1 strongest) and a header line
  teaching the notation. Persisting records keep their byte positions
  across turns (longest-common-prefix-maximal for provider prompt
  caches; new records append at the tail by construction) while rank
  keeps importance visible. Engine absent = ranked order, unannotated
  (exactly the prior render).

### Added
- **Durable session conversation replay, read side** (2026-07-16, operator
  directive, agora `durable-sessions` contract v1): new public
  `abstractruntime.session_history.session_chat_messages` reconstructs a
  session's prior conversation as chat messages from its COMPLETED root
  runs — the run store is the single durable transcript (no dual-write).
  Chronological, whole-turn windows under `max_messages` and a cumulative
  `max_total_chars` budget (drop-oldest; the newest turn always survives),
  per-message truncation labeled `#TRUNCATION`, internal/scheduled
  workflows and failed/silent runs excluded, `metadata.kind =
  "session_turn"` on every replayed message. Hosts (AbstractGateway) use it
  to seed `context.messages` for new runs of the same session so thin
  clients no longer ship their local transcripts.
- `_best_effort_session_turns` gained `include_stats`/`include_artifacts`
  flags (defaults keep history-bundle behavior) so the replay hot path
  skips descendant-ledger scans, loads only a bounded newest-first window
  of full RunStates, and orders same-millisecond turns by their full ISO
  timestamp (sub-ms tiebreak) instead of reversing them.

### Added
- **Branded report exports** (2026-07-15, operator directive via the flow
  seat, same exceptional lane as the PDF Unicode fix): `render_pdf_bytes` /
  `render_docx_bytes` accept an opt-in `branding` payload and render a
  discreet framework identity — small gray meta line under the title
  (workflow@version · report date · AbstractFramework / AbstractFlow —
  abstractframework.ai), thin rule, running page footer (identity left,
  page number right; PDF later pages also get a tiny running header), and
  honest document metadata (PDF author/creator/subject; DOCX core
  properties + a real footer part with a PAGE field). The report DATE is
  always present (defaults to generation date). Legacy no-branding calls
  are byte-path unchanged. The `write_pdf`/`write_docx` visual nodes brand
  BY DEFAULT with run provenance: the compiler injects the run's
  workflow_id into their payloads (`_runtime_workflow_id`; `bundle@version:
  flow` → `bundle@version` label) and the handlers honor a `branding` input
  to override fields or disable (`false`/"off"). Both renderers also dedup
  a leading markdown H1 identical to the document title (previously printed
  twice). Regression tests: `tests/test_report_branding.py` (15 tests);
  end-to-end pinned through a compiled visual-flow run.

### Fixed
- **JsonFileRunStore scan memo — the pegged-gateway incident** (2026-07-15,
  entity's live measurement c2394: the gateway process at ~98% CPU with a
  worker thread burning in json.loads, taxing every endpoint through the
  GIL; journey loads served at 1/12th of their generation speed): the
  runner polls `list_runs(status=...)` ~3x per 0.25s, and at 3,241 run
  files the 512-entry RunState LRU (deliberately bounded since the 1.5GB
  incident) could not cover the directory — every poll evicted and
  RE-PARSED ~2.7k multi-MB, mostly TERMINAL files that could never match
  the RUNNING filter. New per-store SCAN MEMO: run_id → (mtime_ns, the
  small index fields scans filter on — status/wait/ids/timestamps/
  lifecycle, NEVER vars, ~300B/file so it covers any directory), validated
  by stat per use (same cross-process freshness as the RunState cache),
  fed by save()/load(), purged on delete(); unparseable files tombstone by
  mtime so a torn file is not re-parsed every poll. `list_runs` filters on
  the memo and full-loads only matches; `list_run_index` rows ARE the memo
  fields (an index page over unchanged files parses nothing);
  `list_due_wait_until` filters due-ness on the memo. Measured (800 files
  / 163MB / 3 RUNNING, the live shape scaled): 381.6ms → 10.8ms per scan
  (35x) — at the runner's 12 scans/s the old shape burned 4.6s CPU per
  wall-second (the pegged thread), now ~13% of one core. Pinned in
  `tests/test_json_store_scan_memo.py` (warm scans parse nothing,
  cross-process mtime freshness, torn-file tombstone, delete purge, index
  row shape, due-scan correctness).
- **Agent-node skills passthrough** (card 0087 runtime half, gateway seam
  c2286): visual Agent nodes execute as subruns whose `_runtime` is built
  fresh, so the root run's trust-gated `skills_block` (gateway-resolved
  from `input_data.skills`) never reached basic-agent-style workflows.
  The parent block now rides into Agent-node child vars VERBATIM
  (setdefault — byte-stable per run, the prompt-cache contract;
  whitespace-only blocks never ride), and `read_skill` joins an EXPLICIT
  child allowlist when a block rides (the both-halves contract: index in
  prompt + executor reachable). An EMPTY child allowlist ([] = registry
  defaults) deliberately stays untouched — appending there would restrict
  the child to one tool; registry defaults already carry read_skill when
  the host registered it. Pinned in
  `tests/test_visual_agent_skills_block_passthrough.py`.
- **PDF export rendered scientific + typographic glyphs as tofu boxes**
  (`documents/pdf.py`): the ReportLab base-14 fonts (Helvetica) have no glyph
  for characters LLMs routinely emit — non-breaking hyphen (U+2011), narrow
  no-break space (U+202F), en/em dashes, Greek letters (Δ τ δ ε), math
  operators (≤ ≥ ≈ ×), subscripts — so generated reports (e.g. co-scientist)
  showed ■ boxes. Fix, two layers: (1) typographic punctuation/whitespace is
  normalized to ASCII equivalents (`_normalize_pdf_text`) so it can never box
  regardless of font; (2) a Unicode TTF font (DejaVuSans via matplotlib, or a
  system font, or `ABSTRACTRUNTIME_PDF_FONT`) is registered and applied to all
  text + a Unicode mono for code, so Greek/math/subscripts render as
  themselves. Degrades to Helvetica + normalization when no TTF is available.
  Regression-tested (`tests/test_pdf_unicode_rendering.py`): round-tripped PDFs
  contain zero replacement/box chars and preserve the scientific glyphs.

### Added
- **RuntimeHealth — counters, not folklore** (backlog 0054, operator-signed
  plan 2026-07-13; the observability wave's runtime half): the runtime's
  self-knowledge was `logger.warning` at ~25 sites plus per-feature stats
  objects nothing aggregates — the H7c starvation was diagnosed by sampling
  a live pid. New `core/health.py`: one always-on `RuntimeHealth` per
  Runtime — thread-safe monotonic counters incremented at the EXISTING
  warning/decision sites (effect steps/retries/failures, waits entered,
  resumes, crash-replay reuses, absorbed failures, steer drains + delivered
  messages, inbox drops, vars-threshold crossings), coarse tick-duration
  buckets (every `tick()` is timed through a thin wrapper), vars-size
  gauges, and a 16-entry last-errors ring — read via `runtime.health()`
  as one JSON-safe snapshot (no metrics framework; hosts serve it however
  they like; process-lifetime is honest). Companion read
  `runtime.list_stalled_waits(older_than_s=...)`: the #1 silent-stall
  class (a run parked on WAIT_EVENT / tool approval / subworkflow that
  nobody resumes) as a pure query over WAITING runs — reason, wait_key,
  age, paused flag, oldest first; non-queryable stores answer [] honestly.
  `RuntimeHealth` exports at the root. Pinned in
  `tests/test_runtime_health_and_bounded_vars.py`.
- **Bounded run-vars growth for 24/7 residents** (backlog 0053): everything
  that accretes in `run.vars` is serialized on EVERY save, and node traces
  were measured at ~95% of a 2MB resident RunState. Three cap classes +
  a gauge, none silent: (1) node-trace entries above 32KB compact JSON
  (the size check reuses the existing JSON-safety dumps — zero extra
  serialization) have their large leaves OFFLOADED to the artifact store
  as `$artifact` refs (`source=node_trace_offload`), or labeled-truncated
  (`#TRUNCATION`) when no store is wired; entries carry `trace_bounded:
  offloaded|truncated`, small entries are byte-identical to before, and
  the run-vars copy of the RESULT (result_key) is never touched. (2)
  `_runtime.inbox` caps at 200 messages, drop-oldest with a counted
  `inbox_dropped` in the run's own vars + health counter — a workflow
  that never drains its inbox can no longer grow state forever. (3)
  `evidence_warnings` caps at 50 (drop-oldest + counted). (4) a vars-size
  gauge refreshes after every tick that executed at least one effect step
  (gated on step activity so a scheduler sweeping parked runs pays
  nothing) and crossings of the 8MB warn threshold count + log once per
  run per process. Resident   benchmark (400 steps, 8KB results, JSON
  backend): final state 0.92MB → 0.11MB (~8x smaller) with per-step time
  4.7ms → 4.1ms — the cap pays for itself immediately and the gap widens
  with uptime since save cost tracks state size.
  ADVERSARY FOLD (fable5, 2026-07-14, no P0; 2 P1 + 4 P2 fixed
  same-session): (P1-1) `list_stalled_waits` candidates now come from
  `list_run_index(oldest_first=True)` — a NEWEST-first window silently hid
  the OLDEST waits (the definition of stalled) once waiting runs exceeded
  the window, returning [] on the exact fleet board this exists for
  (their demo: 5 two-hour stalls invisible behind 15 fresh waits); the
  `oldest_first` kwarg landed on the store protocol + all three in-repo
  `list_run_index` implementations, only the ≤limit aged winners load
  their document (the P2-4 full-parse cost fix riding the same query),
  and unknown-age rows sort FIRST (they truncated exactly when the board
  was full). (P1-2) top-level `error` strings now bound on the offload
  path — a 100KB provider error escaped the subtree-only cap while
  stamped "offloaded", and failed steps are exactly the entries that loop
  (~10MB/node accretion while claiming to be bounded). (P2-1) trace
  consumers (compiler tool-activity extraction ×2, agent trace report)
  skip `$artifact` refs instead of extracting phantom tool calls (count 1,
  name None) and the report names the offload. (P2-2) the inbox cap is
  now BACKPRESSURE, never destruction: the drain delivers only up to
  headroom and acks ONLY delivered seqs — the first cut drop-oldested
  operator words UNREAD while `steer_seen` claimed them, engaging before
  the sidecar's refuse-at-500 could ever reach the sender; excess steers
  stay pending (watermark redelivery), and a never-draining workflow
  surfaces as the sidecar's loud append refusal. (P2-3) the vars-gauge
  gate keys on a run-object flag instead of the runtime-global step
  counter — with a second thread executing effects (the entity-lane
  shape) the global compare made 30% of parked-run probes serialize a
  parked run's full vars and turned the gauge into last-writer-wins;
  the flag also removes the wrapper's two snapshot copies per tick.
  (P3) effect failures and raising ticks now reach the last-errors ring
  (it had ONE producer — the ring exists to answer "what broke last");
  the vacuous caplog pin now asserts warn-once for real. Cleared by the
  adversary, on the record: offload aliasing (spine copies — live
  payloads/ledger/result_key byte-intact), tick-wrapper exception
  propagation, 4-thread counter exactness, content-addressed artifact
  dedup on replay re-bounding.
- **Indexed idempotency lookup** (backlog 0047, operator-signed plan
  2026-07-13; the measured scale cliff: the per-step dedup probe full-parsed
  the entire run ledger — ~44ms/step at 32MB, quadratic per run):
  `LedgerStore.find_completed_result(run_id, idempotency_key)` with three
  implementations — SQLite gains `idempotency_key`/`step_status` columns +
  the COVERING partial index `idx_ledger_idem(run_id, idempotency_key,
  seq)` (point query, oldest-completed-wins for exact parity with the
  historical scan; the perf adversary's N1 found the first 2-column shape
  was never chosen by the planner — the ORDER BY pulled it onto the seq
  index and the probe stayed O(per-run rows), 69ms/miss at 50k records; the
  covering shape measures 0.005ms at 10k, flat, plan-shape pinned by test
  and a mismatched existing index is rebuilt at open); pre-0047 databases
  are ALTER-migrated and column-backfilled at open (json1 UPDATE with a
  Python-side fallback; a near-empty partial index
  `idx_ledger_unbackfilled` keeps the every-boot probe O(1)); a
  version-skew guard degrades to a bounded scan when NULL-column rows exist
  for the run (old writer + new reader on one file, never a silent miss).
  JSONL gains a write-through cache of COMPLETED records (the append's OWN
  serialized line is reused — zero extra encoding, perf adversary N2;
  parse-on-hit returns DISK truth, never live objects reducers might mutate
  later; 64KB/entry cap, 512 LRU, dropped on delete()) + a backward tail
  read (`_tail_lines`, 64KB blocks from EOF — O(tail bytes) bounded by a
  32MB byte ceiling so media-shaped ledgers with multi-MB record lines
  cannot force gigabyte reads per cold probe, perf adversary N3; beyond the
  ceiling the probe answers an honest miss). Replay-adversary folds
  (2026-07-14): the tail read splits on the writer's ACTUAL line
  discipline (`"\n"`), never `str.splitlines()` — JSON leaves
  U+2028/U+2029/U+0085 raw under `ensure_ascii=False` and splitlines
  fragmented a completed record carrying U+2028, so the COLD crash-replay
  probe missed it and the effect re-executed (their P0, demonstrated
  live); the quoted-key prefilter is skipped for keys containing
  JSON-escaped characters (host-pluggable policies mint arbitrary keys —
  P2-a); a deleted ledger clears/overrides every instance's cache answer
  (disk truth wins — P2-b); torn legacy rows are stamped
  `step_status='unparseable'` during backfill so one unparseable row can
  never silently re-degrade a run's every probe to the full scan (P2-c);
  and the skew-guard docstring states the honest bound (the degrade is the
  bounded window scan, not a no-miss guarantee — materially defused by
  issuance-scoped keys, which pre-0047 records can never match). UPGRADE
  BOUNDARY (their P2-g): because keys now hash the issuance counter, a run
  mid-flight ACROSS the upgrade re-executes its one in-flight step
  (bounded at-least-once, once per run); pre-upgrade records are never
  matched by post-upgrade probes. Correctness envelope: keys are
  issuance-scoped
  (`_runtime.effect_seq`), so a genuine hit can only live at the ledger
  tail (crash-replay = effect completed, the save after it did not land);
  beyond `IDEMPOTENCY_TAIL_WINDOW` (256) the runtime re-executes — the
  documented at-least-once default, now a pinned DECISION. Decorators
  (hash-chained, observable, offloading) delegate to their inner store;
  duck-typed host stores keep the historical inline scan (`#FALLBACK`).
  `Runtime._find_prior_completed_result` routes through the store method.
  Pinned in `tests/test_indexed_idempotency_lookup.py` (point query,
  migration/backfill, skew degrade, cache disk-parity, window bound,
  decorator delegation, runtime seam both ways).
- **Agora channel shared-fs/store tools** (swarm-seat promotion step 3,
  commons c1669, operator-approved 21:41): the toolset grows 7→12 —
  `channel_fs_write` (PUT with description/mime/`expect_version` CAS),
  `channel_fs_read` (version-aware), `channel_fs_list` (prefix),
  `channel_store_set` (string value + CAS), `channel_store_get` — the
  shared-artifact collaboration surface both fleet scripts hand-rolled, now
  first-class with the same key-gated registration and the same H8
  per-agent identity stamping (the schema-hidden `_agora_agent` arg; the
  effect handler's stamp set derives from `AGORA_TOOL_NAMES`, so the five
  inherit it structurally). Approval classes per the plan: channel READS
  are safe auto-approve; channel WRITES are write-classed (a shared
  artifact/decision write is a mutation every channel member sees).
  Contract-tested against a stub hub server in
  `tests/test_agora_tools_http.py` (paths, CAS payloads, alias key
  resolution on a channel write, approval classification, 12-tool pin).

### Fixed
- **Issuance-counter advance now binds to the step's own saves** (replay
  adversary P1, 2026-07-14, demonstrated live): file-backed run stores
  ALIAS loaded RunStates, so a control-plane save (pause_run, limits
  update) landing between the idempotency probe and the step's own save
  used to serialize an already-advanced `_runtime.effect_seq` WITHOUT the
  step's result — after a process restart, the resumed replay recomputed a
  DIFFERENT key, missed the completed record, and re-executed the effect
  (double tool execution/LLM spend on the pause-mid-effect path). The
  advance now happens immediately before each save that lands the step
  (completed/waiting/failed/absorbed), so out-of-band saves carry the
  un-advanced counter and crash-replay reuses the completed result.
  Pinned: `test_pause_mid_effect_then_restart_still_reuses_the_completed_result`.
- **Prompt-cache binding failures no longer burn the retry budget**
  (bloc-seam adversary A-2, 2026-07-13): `_llm_error_is_retryable` now
  classifies structured `PromptCacheError`s — binding verification codes
  (`prompt_cache_binding_missing`/`_mismatch`/`_invalid`/`_invalid_key`/
  `_bare_string`) and `prompt_cache_unsupported` are DETERMINISTIC (same
  params, same refusal, every attempt) and fail immediately; generic
  operation failures (I/O during load/save) keep the retryable default.
  Previously they matched no branch and defaulted to retryable.
- **Bare-string `prompt_cache_binding` routes to `prompt_cache_key`**
  (bloc-seam adversary A-3; one-meaning-per-name, the agent seat's c1670
  convention): a string binding is cache-key intent — both the LLM_CALL
  handler seam and the remote client's params normalization now route it to
  `prompt_cache_key` and drop the binding param, instead of coercing it
  into a `{"binding_id": ...}` dict (exactly the shape core refuses as
  `prompt_cache_binding_bare_string` — the 2026-07-11 live-visit collision
  class, previously re-manufactured one hop before core's refusal). Dict
  bindings keep strict durable-bloc verification semantics unchanged.
  Conflicting explicit keys refuse loudly. Pinned in
  `tests/test_prompt_cache_modules.py` + `tests/test_llm_retry_classification.py`.

### Changed
- **Hot-path store reads are column reads** (backlog 0068, operator-signed
  plan 2026-07-13; the SQLite full-document tax): the external-control probe
  (`_abort_if_externally_controlled`, loop top + before every save) needed
  two fields and paid a whole-document parse per probe on the production
  backend. `runs` gains `paused` + `run_lifecycle_json` twin columns —
  written in the SAME upsert as run_json (one truth, two read speeds; the
  pause-flag shape now has ONE source, `core.vars.is_paused_vars`, shared by
  the runtime and the store), ALTER-migrated with a duplicate-column-
  tolerant race guard and backfilled at open (Python loop applying the same
  sanitization as the write path; torn rows match the readers' `{}`
  fallback). New `SqliteRunStore.probe_control(run_id)` answers
  `(status, paused)` in one column read and answers None on pre-migration
  rows — the runtime's probe consults it via duck-typing (stores without it
  keep the historical full load) and only full-loads when the probe says
  CONTROLLED (the caller returns the RunState). `list_run_index` reads the
  lifecycle column and deliberately drops `run_json` from the page query —
  fetching the multi-MB document column dominated the cost even before
  json.loads (measured ~1.1s/100-row page either way with run_json riding
  the SELECT); pre-migration NULL rows fetch their document individually.
  A bare `idx_runs_updated` index serves unfiltered ORDER BY pages at the
  100k-run design target. `_append_progress_event` no longer parses the
  entire ledger per callback — the key suffix was a pure uniquifier and is
  now a uuid (no consumer reads the key shape). `SqliteSteerSidecar` reuses
  a per-thread connection (fresh connect+PRAGMAs+schema cost ~0.39ms per
  tick-loop iteration) while KEEPING the F3 purge-healing property: a
  stat() per call detects a deleted database file and reopens (a cached
  connection would keep writing the unlinked inode — silently losing steers
  until restart); failed writes roll back or drop the thread connection so
  a wedged transaction can never poison later calls. Measured at ~1.9MB
  states: control probe 1.59ms → 0.004ms (~360x), 100-row index page
  171.7ms → 26.5ms (~6x; the remaining cost is SQLite row-overflow
  traversal — the twin columns sit after run_json in the physical row, a
  known ceiling not worth a table rebuild).   Pinned in
  `tests/test_hot_path_store_reads.py` (column truth vs document truth,
  pre-migration None, legacy ALTER+backfill incl. torn rows, runtime
  fast-path consultation + pause still honored, poisoned-run_json column
  read, per-thread reuse, purge healing, progress-key no-list).
  ADVERSARY FOLD (fable5, 2026-07-14, four P1 fixes landed same-session):
  (P1-1) every backfill UPDATE is guarded `AND paused IS NULL` — runs
  rows are MUTABLE (unlike the ledger rows the backfill was modeled on),
  and a stale snapshot UPDATE racing a concurrent save() from another
  process stamped paused=0 over a freshly-paused row, after which the
  probe said "not controlled" and the tick's next save DESTROYED the
  pause (their live demo: 2 → 7 node executions, final doc unpaused);
  with the guard a concurrent save always wins. (P1-2) the steer
  sidecar's purge healing now checks FILE IDENTITY (st_dev, st_ino), not
  existence — after a purge the first thread to touch recreates the file,
  and an existence check then passed for every OTHER thread while its
  cached connection wrote the unlinked inode (appends "succeeded" into
  the orphaned file: silent steer loss, demonstrated with the production
  two-thread shape). (P1-3) the backfill streams in cursor-paged batches
  (class-attr batch size, commit per batch, O(1) LIMIT-1 probe served by
  a new near-empty partial index `idx_runs_unbackfilled`) — the fetchall
  materialized every unbackfilled document (~200GB at the 100k-run design
  target at 2MB states) and the single end-of-open commit turned an OOM
  kill into an open-crash-loop; a crash now resumes where it stopped.
  (P1-4) `OffloadingRunStore` gained an explicit `probe_control`
  passthrough — the wrapper forwards methods explicitly, so the gateway's
  production wiring (OffloadingRunStore(SqliteRunStore)) silently hid the
  fast path and kept ~22 full multi-MB parses per 10-step tick, on the
  exact deployment whose measurements justified the item; the unit test's
  `__getattr__` double had the opposite forwarding semantics (the
  fixture-double-hides-the-seam class). (P2-1, comment honesty)
  `run_lifecycle_json` sits after run_json in the physical row, so the
  index page still walks each row's overflow chain (~30ms/100 rows at
  2MB) — the eliminated cost is the 100x json.loads, not all I/O; noted
  in code, side-table migration judged not worth it. Cleared by the
  adversary, on the record: WAL freshness cross-thread/cross-process, the
  controlled-window width (unchanged), the writer census (every save
  funnels through SqliteRunStore.save), N+1 fallback bounded, uuid
  progress keys, enum-value comparisons. New pins: stale-backfill guard,
  batched backfill restart, two-thread purge healing, wrapper probe
  passthrough.
- **Terminal ledger records are slim** (backlog 0067-M, operator-signed plan
  2026-07-13; the O(turns²) ledger-growth term): the terminal
  COMPLETED/WAITING/FAILED append used to re-persist the full effect payload
  the STARTED record (same `step_id`, appended before execution) already
  holds, and an LLM result carried the request two MORE times through its
  observability metadata. New `storage/ledger_slim.py`: on the terminal
  append, oversized payload fields (>4KB compact JSON) become verified
  `$slim` markers naming the STARTED record (sha256 + byte count; sub-4KB
  fields — tool args, flags — stay inline for ledger consumers); the
  result's `_runtime_observability.llm_generate_kwargs` fields and
  `_provider_request.payload.messages` dedup ONLY when byte-identical to
  the STARTED payload or its documented local reconstruction
  (system+messages+prompt layout) — decorated wire bytes (grounding
  envelopes, volatile-stripped remote bodies) NEVER match and stay
  verbatim, so the "durable bytes equal sent bytes" capture (B3,
  `/llm --verbatim`) is preserved by construction. Crash-replay reuse
  rehydrates markers before the result flows into vars
  (`_find_prior_completed_result` resolves against the STARTED record and
  verifies the sha — a mismatch keeps the marker, never a silently-wrong
  payload), and the run-vars copy of a live result is never touched (spine
  copies only). `history_bundle` readers (tool-call stats, answer_user
  fallback) resolve markers from the records they already hold; resolvers
  export at `abstractruntime.storage` for host-side ledger consumers.
  Measured on the perf adversary's 120-message/185KB shape: ledger bytes
  per LLM effect 735,723 → 186,554 (74.6% saved; terminal record 551,971 →
  2,802), removing the per-turn conversation duplication entirely — ledger
  growth is linear again. Pinned in `tests/test_ledger_record_slimming.py`
  (marker+verify, tamper refusal, B3 verbatim preservation, crash-replay
  byte-identity, history resolution, vars-copy isolation).
  ADVERSARY FOLD (fable5, 2026-07-14, four fixes landed same-session):
  (P0) the factory composition chained a method-presence Protocol check
  after the offload wrap — `OffloadingLedgerStore.subscribe` satisfies
  `ObservableLedgerStoreProtocol` by presence while raising at call time,
  so `Runtime.subscribe_ledger` died with RuntimeError on EVERY durable
  deployment (abstractcode's cache meter + live ledger feed swallowed it
  silently); replaced by ONE explicit composition site
  (`_compose_durable_ledger` → Observable(Offloading(raw)); caller-
  supplied Offloading stores are wrapped, never trusted for
  observability) and the test now pins BEHAVIOR (subscribe delivers, with
  the pre-offload record) instead of the helper's return type. (P1)
  slimming is now anchored to STARTED-time digests
  (`capture_started_payload_digests` immediately before the STARTED
  append): non-LLM records hold payloads BY REFERENCE, and a handler
  mutating an oversized field in place between the appends would have
  minted a permanently-unresolvable marker with the execution-time bytes
  existing NOWHERE — a diverged field now keeps its verbatim mutated
  bytes (slimming drops duplicates, never information); the metadata
  dedup verifies the same freshness (layout parts included). (P1)
  `OffloadingLedgerStore.list()` reverted to plain delegation — refs stay
  refs on the read surface (rehydrating every read measured 113x time /
  593x bytes on offload-heavy ledgers across the gateway's history/SSE/
  summary surfaces, and made chain verification over the rehydrating
  read report FALSE tamper alarms since hashes cover post-offload
  bytes); rehydration lives only on `find_completed_result`, the
  crash-replay path where byte-identity is a correctness requirement.
  (P2) crash-replay rehydration is TARGETED to the two runtime-written
  metadata paths (`resolve_result_metadata_markers`) instead of a
  whole-tree scan — a tool result echoing a slimmed record as DATA
  (runtime-explore class) would have been "resolved" into divergence;
  echoed markers now survive replay byte-identically. New pins: durable
  factory subscribe end-to-end, mutated-payload verbatim preservation,
  echoed-marker replay identity, refs-on-read + rehydrate-on-replay.
  CPU note (their P2-3, accepted trade): the slim pass costs ~1.7-2x one
  record serialization at terminal appends, buying ~99% fewer terminal
  bytes — CPU-for-disk, correct for disk-backed ledgers.
- **`OffloadingLedgerStore` is wired into the durable factories** (backlog
  0067-M step 5): `create_local_runtime`/`create_remote_runtime`/
  `create_hybrid_runtime` wrap DURABLE ledger stores (never in-memory
  shapes, never memory-only artifact stores) and `open_entity_runtime`
  writes through offloading into the HOME's artifact store (ledger rows
  stay bounded while the bytes remain part of the life, inside the home
  directory). Read-side contract delivered with the wiring: `list()` and
  `find_completed_result()` REHYDRATE refs the offloader itself created
  (tag-checked `source=ledger_*_offload`; handler-authored artifact refs —
  media handoff currency — pass through untouched; nested offloads resolve
  recursively), so consumers see full records and a crash-replayed
  offloaded result is byte-identical to the live handler result. The
  observable wrapper is applied OUTSIDE offloading, so live SSE subscribers
  receive the record before offloading touches it.
- **Durable-write discipline** (backlog 0067, operator-signed plan
  2026-07-13): the run/ledger hot path no longer deep-copies state to
  serialize it. New `storage/serialize.py` — `runstate_to_dict` /
  `steprecord_to_dict` build field-enumerated SHALLOW dicts (`vars`/
  `output`/`result` pass BY REFERENCE under the documented single-writer
  ownership contract; `waiting` is the only nested dataclass and converts
  eagerly; field enumeration keeps the dicts drift-proof as models evolve)
  and `dumps_compact` writes compact JSON (no indent, no separator spaces)
  with a `default=` hook that lazily asdict-converts dataclasses nested
  inside payloads — exactly what `dataclasses.asdict` produced, without the
  recursive deep copy on every write. Applied to `JsonFileRunStore.save`
  (also drops `indent=2` — the single most frequent durable write in the
  system), `SqliteRunStore.save`, `JsonlLedgerStore.append`,
  `SqliteLedgerStore.append`/`append_chained`, and the hash-chain wrapper's
  non-chained path; `ledger_chain._canonical_json` carries the same hook so
  append-time hash input (live objects) and verify-time input (parsed JSON)
  canonicalize to the same bytes. Measured (independent perf adversary A/B
  against verbatim replicas of the old writers): JSON-file save x3.4 at
  ~200KB states, x1.65 at ~2MB, converging to ~1.0x on very string-heavy
  states (the win is structure-density-dependent); end-to-end over a real
  Runtime, x2.81 total on the JSON-file backend and x1.34 on SQLite over
  250 growing-state steps; bytes at rest -7.4% on the realistic e2e state.
  VALUE parity pinned (round-trip equality against `asdict` output,
  unicode, tuples, nested dataclasses, chain verify over both backends) in
  `tests/test_durable_write_discipline.py`. The in-memory stores keep
  `asdict` deliberately — their copy IS the isolation contract that
  simulates disk truth for tests.
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
- **Tick-boundary grant check** (both lane audits flagged the day-open-only
  lag independently — runtime G9 + gateway c1501(c)): a personal-grant
  revocation/expiry landing MID-DAY now ends the day at the NEXT tick
  boundary instead of riding to the day's end (`ticks_per_day` is
  operator-configurable, so the old lag was unbounded in configuration
  while consent must not wait). The day closes with its normal ceremony
  (reflection runs — nobody is waiting); the top gate then performs the
  ruled sleep-landing write and exit, keeping one write site.
- **Phase-machine audit fixes** (laurent 13:54 every-lane adversarial
  verification against decision:entity-phase-state-machine v3; runtime's
  fable5 audit verdict COMPLIANT-WITH-GAPS, gaps closed same cycle):
  (G1) a grant revoked/expired DURING an operator sleep or visit is now
  seen AT THE WAKE — the post-idle path routes back through the top gate
  (which re-checks stop + grant) instead of falling through to the summon;
  a mid-idle revocation previously bought a full unmandated personal day.
  (G2) the disarmed exit now lands the entity in SLEEP per the ruled
  totality ("grant expiry/revocation ends personal → sleep"): state=asleep
  written with the ruled cause word (`personal_grant_end_cause`:
  grant_expired for a lapsed timer, grant_revoked otherwise), written_by
  "grant-gate" — never phase-less behind a stale awake. (G3, prose rule)
  `VISIT_OWN_TIME_PARAGRAPH` no longer teaches the forbidden suspend model
  ("paused … resumes"): a visit ENDS the personal stretch and a NEW one
  begins at close. (G7) the CLI's `_wake_loop_if_yielded` restores awake
  only over its OWN auto-yield write — an operator's mid-visit
  asleep/paused stands.
- **Pid-identity token + pause heartbeat** (gateway state-wave adversary 2,
  c1469 — their runtime-lane findings, closed same cycle): `loop_status`
  now stamps `pid_started_at` (the OS-recorded process start time via
  `ps lstart`) and `read_loop_status` requires the CURRENT holder of that
  pid to match — a recycled pid's corpse reads not-running in EVERY phase
  (the `between` corpse could 409 starts forever and aim a freeze-SIGKILL
  at an innocent process). Unstamped files (older writers) keep
  pid-alive-only semantics; token probe failure degrades the same way. The
  day-staleness belt STAYS beside the token (it catches a live-but-wedged
  loop that stopped heartbeating, which pid identity cannot). `_idle_while`
  gains a `phase` heartbeat: a mid-day PAUSE re-stamps `updated_at` every
  poll, so a long freeze no longer trips the staleness belt into reporting
  a live paused loop as not-running (console lie + a post-wake double-start
  window). `_pid_alive`'s EPERM-reads-as-dead direction is now documented
  as deliberate (safe for every current caller; never reuse it in
  signaling paths without splitting the meanings).
- **Graceful night yield** (one-active-phase ruling, laurent 13:28; memory's
  cancellation semantics c1462): `build_consolidator` passes the engine's
  `sleep_pass` a `should_continue` predicate — pure reads of the home state
  + STOP file — so a transition arriving mid-night (visit auto-yield's
  mode=visiting, operator wake, manual brake) ends the night at the next
  phase boundary: the in-flight phase completes its writes, later phases
  skip named, the next sleep resumes there. Stop COMMANDS are deliberately
  not consulted (consumed-on-read; the post-night boundary check consumes
  them exactly once). Version skew degrades to a full night with a
  `#FALLBACK` label (older engines without the kwarg/`sleep_pass`).
- **CLI personal-grant acts** (`--grant-personal` / `--grant-personal-hours H`
  / `--revoke-personal` on `python -m abstractruntime.identity.life`): each
  is a distinct operator act that writes through `write_personal_grant` and
  EXITS — arming never starts the loop (wake ≠ grant ≠ start); combining
  flags refuses. The terminal operator is the principal (`person:<os-user>`
  in granted_by). Honest limit documented in-code: CLI grants carry their
  audit in the file's fields but write no host marker — the gateway's
  arming surface is the marker-first lane.
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
- **Effect issuance idempotency — stale tool-result replay** (agent seat
  P0, commons c1568, 2026-07-13): the idempotency key was
  (run_id, node_id, effect_type, normalized_payload) and
  `_find_prior_completed_result` scanned the WHOLE run ledger first-match —
  so a byte-identical effect batch RE-ISSUED at the same node (re-reading a
  file after editing it, re-running the test suite after a fix, any loop
  with repeated payloads) silently REPLAYED the first stale result instead
  of executing, and the agent concluded its change didn't apply and looped.
  The key now also hashes the run's `_runtime.effect_seq` issuance counter,
  which the tick loop advances in the same save that lands each step: a
  genuine later issuance gets a fresh key (executes), while a crash-replay
  (vars rolled back to the last save) recomputes the SAME key and reuses
  the completed result (exact at-most-once preserved). `call_id` stays
  stripped from the hash (provider ids are non-semantic — correct); the
  counter is the missing WHICH-issuance dimension. `compute_idempotency_key`
  gains an `effect_seq` parameter (default 0). Pinned:
  `tests/test_effect_issuance_idempotency.py` (identical batch executes
  twice, crash-replay reuses, retries-within-one-issuance share one key).
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
