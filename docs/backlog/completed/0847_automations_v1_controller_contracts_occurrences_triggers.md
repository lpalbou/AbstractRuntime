# 0847-abstractruntime: [FEATURE] Automations v1: controller bundle, definition/state contracts, deterministic occurrence creation, trigger-source registry, session-turn selector, index queries

> Created: 2026-09-26
> Status: Completed — UNRELEASED (local commits on `main`; ships in the next minor release after the operator's validation)
> Completed: 2026-09-27
> Type: feature
> Priority: P1 (leads the next minor wave; every other package's Automations work integrates against it)
> Labels: automations, triggers, durability, session-history, run-index
> Design: untracked/design/automations-PLAN.md (2026-09-26)

## Summary

Make an Automation a runtime-native durable workflow. The automation IS a
controller root run (`automation_id == controller root run_id`) executing one
pinned, runtime-packaged VisualFlow bundle
(`abstractframework.automation-controller@1.0.0`, flow `controller`). Each firing
is a serial child run (an *occurrence*) created with a deterministic id, so a
crash at any point around dispatch can never produce a duplicate or a lost
occurrence. Triggers are pluggable `TriggerAdapter`s discovered through the
entry-point group `abstractruntime.trigger_sources`; v1 ships `schedule@1` and
`manual@1`. Commands (revise, pause, resume, run now, stop current, archive) are
applied by one runtime function, `apply_automation_command`, under the same
per-run serialization as `resume()`. One session-turn selector
(`select_session_turns`) replaces the root-only history filters so Growing mode,
occurrence transcripts and Discuss all read the same conversation. Run-index
queries (`list_automations`, `latest_occurrence`, attribution filters on
`list_run_index`) let hosts project automations without loading run payloads.
Everything works from a bundle alone, without the gateway.

## Why

The operator's principles for the feature (design §1, verbatim):

> **Runtime-native.** Automations execute as durable runtime workflows, with ordinary run state, ledgers, artifacts and child runs.
>
> **One system.** An Automation is its root run. Gateway APIs and indexes project runtime truth; they do not independently schedule or execute work.
>
> **Extensible triggers.** "Trigger" is the user concept. Runtime `TriggerSource` adapters register once and become discoverable to hosts and apps.
>
> **Visible and manageable.** Users can inspect results and steps, edit definitions, pause/resume, run manually, stop current work and archive.
>
> **Conversation representation.** Each occurrence contributes a trigger/task turn and answer. Independent mode starts fresh; Growing mode supplies bounded prior conversation. Long-term summaries remain target-owned in v1.
>
> **Isolated discussion.** Discuss creates a separate session seeded at a selected occurrence. It cannot write into automation context or resources.
>
> **Minimal scope.** Reuse commands, stores, effects, bundles and tool ceilings. External admission, automatic summaries and connector infrastructure are deferred.

Operator rulings of 2026-09-26 (verbatim; they amend the design):

> 1. Runtime-root model with child occurrences: **approved**.
>
> 2. Independent mode as the default: **agreed** (matches today's `share_context=false` behaviour for isolated occurrences).
>
> 3. Schedule + manual first; external triggers in the next phase: **agreed**, but the next phase is a PLANNED backlog item, not proposed.
>
> 4. **Discuss is NOT read-only and NOT tool-restricted.** A discussion is a new durable runtime session, forked/seeded from the automation's
>    conversation through the chosen occurrence, replayable like any session, with the target's normal tools. Isolation means only that it
>    never writes back into the automation's session/context (a fork). Contract B's `DISCUSSION_READERS_V1` allowlist is DROPPED. Open detail
>    (question to the operator): workspace of a discussion — own workspace with read access to the occurrence's, or shared read-write.

Today the only scheduling primitive is the `on_schedule` node inside a
user's flow and gateway-side `scheduled:` wrapper runs; neither is an
inspectable, editable, pausable object, their history is filtered out of
conversation replay, and a crash between "child started" and "parent waits"
leaves an orphan child that recovery cannot re-link.

## Scope and non-goals

### In scope (v1)

- **Contract A** — definition (`_meta.automation`), controller state
  (`_runtime.automation`) and the `automation.*` ledger observations, recorded as
  existing `emit_event` StepRecords with stable record identities.
- **Contract B** — occurrence attribution (`_meta.occurrence`, inherited by
  descendants with `role:"descendant"`), deterministic child ids,
  `Runtime.start(..., run_id=)` and `START_SUBWORKFLOW.payload.run_id` with
  atomic create-if-absent + identity validation on every run store, and the
  admission → child create/load → parent wait ordering with recovery that
  reconstructs the wait. Discussion attribution (`_meta.discussion`) and a
  discussion seed stored once and included in the discussion session's later
  history composition.
- **Contract C** — `TriggerSource` / `TriggerBinding` / `TriggerEnvelope`,
  the `TriggerAdapter` protocol, the `abstractruntime.trigger_sources`
  entry-point group and registry, and the two v1 sources `schedule@1` and
  `manual@1`.
- **Contract D** — the controller bundle packaged inside the wheel, its
  `automation.<node_id>` adapters (issuing existing effects only), and
  `apply_automation_command()` sharing the root's per-run lock.
- **Contract E** — `select_session_turns`, `list_run_index` attribution filters,
  `latest_occurrence`, `list_automations`, and the indexed columns
  `automation_id, role, occurrence_index, session_kind, change_cursor` on both
  persistent stores (SQLite and JSON files) plus the in-memory and offloading
  wrappers.
- Context modes: Independent (default; fresh `occurrence` session per
  occurrence) and Growing (controller and occurrences share one `automation`
  session; bounded prior conversation via the selector).
- Discuss at runtime level: a new `discussion` session seeded through a chosen
  occurrence, replayable like any session, running the target with its normal
  tools. It never writes into the automation's session or context.

### Out of scope (v1)

- External event admission (inbox, generic `event` source, `run.finished` /
  `run.failed` sources) — planned as
  [0848](0848_automations_v2_external_event_inbox_and_run_triggers.md).
- Any tool restriction on Discuss: the `DISCUSSION_READERS_V1` allowlist from the
  design is dropped by operator ruling 4. The discussion's workspace model (own
  workspace with read access vs shared read-write) is an open operator question;
  do not guess — implement the fork of session/context and leave the workspace
  policy to the ruling.
- Automatic Growing-mode summaries (`context.growing.summary` returns
  `unsupported_feature`), constrained fetching, calendar scheduling, connectors.
- Gateway routes, attention/seen cursors, legacy-schedule projection, UI — owned
  by the gateway and app packages.
- Parallel execution registry, template database, per-automation generated
  flows, automatic legacy migration, universal broker, arbitrary filter
  language, concurrent occurrences, delivery router, agent-memory transfer.
- Multi-process runners on one store: v1 supports one active runner per store
  (see Risks; 0046 is the general fix).

## Contracts (copied from the design, §3 A–E)

Notation: `?` optional, timestamps UTC RFC3339, `JSON` arbitrary JSON. Other
objects reject unknown fields.

### A — Definition/state/ledger

```text
_meta.automation = {
 schema_version:1, revision:int>=1=1, title:nonempty-string,
 controller:{bundle_ref:"abstractframework.automation-controller@1.0.0",
             flow_id:"controller"},
 target:{workflow_id:string,bundle_ref:string,flow_id:string,input_data:JSON={}},
 trigger:TriggerBinding,
 context:{mode:"independent"|"growing"="independent",
          growing:{summary?:{enabled:bool,every_n:int>=1,max_tokens:int>=1}}={}},
 policy:{serial:true,misfire:"coalesce",failure:"continue"},
 session_id:string,created_at:timestamp,archived_at:timestamp|null=null
}
_runtime.automation = {
 active_revision:int=1,
 pending_occurrence:null|{
   run_id:UUID,index:int>=1,event_id:string,revision:int,
   envelope:TriggerEnvelope,phase:"admitted"|"dispatched"
 }=null,
 anchor:timestamp|null=null,tick:int>=0=0,next_index:int>=1=1,
 scheduled_count:int>=0=0,paused:bool=false,
 last_outcome:null|{run_id:UUID,index:int,status:string,finished_at:timestamp}=null
}
```

Server owns IDs, timestamps and revisions. Any supplied
`context.growing.summary` returns `unsupported_feature` in v1.

Ledger observations use existing StepRecords with `effect.type="emit_event"`,
`effect.payload.name` below and `effect.payload.payload`:

```text
common={schema_version:1,automation_id,revision,at,command_id?}
automation.created    common+{definition}
automation.revised    common+{previous_revision,definition}
automation.admitted   common+{run_id,index,event_id,trigger_envelope}
automation.dispatched common+{run_id,index}
automation.completed  common+{run_id,index,status,finished_at,notify}
automation.coalesced  common+{first_tick,last_tick,missed_count,event_id}
automation.paused     common+{}
automation.resumed    common+{next_fire_at}
automation.archived   common+{active_occurrence_run_id?}
automation.command_result
 common+{command_id,status:"applied"|"rejected",error?}
```

These record facts, not external delivery. Stable record identities prevent
duplicate observations.

Command semantics (design §2): **Pause** suspends scheduled admissions; the
current occurrence finishes. **Run now** is allowed while paused, executes once
and remains paused; rejected when busy/exhausted/archived; no queue. **Resume**
re-arms without firing or catching up paused time. **Revise** activates at the
next controller boundary and re-arms idle waits without firing. **Stop current**
cancels the occurrence tree only. **Archive** prevents future admissions,
finishes current work, retains history. Normal downtime coalesces missed
scheduled ticks into at most one occurrence. Automation pause is
`_runtime.automation.paused`, not runtime execution freeze: the parked command
wait accepts manual admission and the flag remains true.

### B — Attribution/creation/discussion

```text
_meta.occurrence={
 automation_id:UUID,occurrence_index:int>=1,event_id:string,revision:int>=1,
 role:"occurrence"|"descendant",fired_at:timestamp,
 trigger_envelope:TriggerEnvelope
}
_meta.discussion={
 automation_id,occurrence_index,revision,seed_run_id,seed_messages?
}
```

Descendants inherit attribution with `role:"descendant"`; only occurrence roots
become turns. Discussion seed is stored once and included in subsequent local
history composition.

Child IDs:

```python
uuid5(UUID(automation_id), f"{revision}:{occurrence_index}")  # scheduled
uuid5(UUID(automation_id), "manual:" + command_id)            # manual
```

Add `Runtime.start(..., run_id: str|None=None)` and
`START_SUBWORKFLOW.payload.run_id`; neither currently exists. Explicit IDs
require atomic create-if-absent and identity validation, never overwrite.
Persist admission first; create/load child; persist parent wait. Recovery
reconstructs the wait.

> **Amended by operator ruling 4 (2026-09-26):** the design's Discuss ceiling
> (`_runtime.allowed_tools = target_effective_tools ∩ DISCUSSION_READERS_V1`) is
> DROPPED. A discussion runs with the target's normal tools and ceiling; its
> isolation is a fork of session/context only.

### C — Triggers

```text
TriggerSource={
 id:string,version:int>=1,label:string,config_schema:JSONSchema,
 event_schema:JSONSchema,capabilities:{kind:"time"|"manual"|"event"}
}
TriggerBinding={
 binding_id:UUID,source_id:string,source_version:int,config:JSON
}
TriggerEnvelope={
 event_id:string,source_id:string,source_version:int,
 fired_at:timestamp,payload:JSON,binding_id:UUID
}
```

Entry-point group: `abstractruntime.trigger_sources`.

```python
class TriggerAdapter(Protocol):
    descriptor: TriggerSource
    def validate(self, config: dict) -> dict: ...
    def prepare(self, binding: TriggerBinding, *,
                state: dict, now: str) -> TriggerWait: ...
    def normalize(self, binding: TriggerBinding, *,
                  event_id: str, fired_at: str,
                  payload: dict) -> TriggerEnvelope: ...

# TriggerWait:
# {kind:"until",until:timestamp}
# {kind:"event",scope:"run",name:string}
# {kind:"exhausted"}
```

```text
schedule@1:{start_at?:timestamp,every?:string,until?:timestamp,
            count?:int>=1,anchor?:timestamp}
manual@1:{}
```

Persist defaults `start_at=creation_time`, `anchor=start_at`; initially require
equal values. `every` is a positive integer duration `[smhd]`; absent means
one-shot. `count>1` requires `every`. Count scheduled admissions only; one
coalesced firing counts once. `until` is exclusive.

### D — Controller

Bundle `abstractframework.automation-controller@1.0.0`, flow `controller`.

```text
start:on_flow_start
→ read_definition → wait → admit → prepare_context
→ dispatch → record_outcome → next → read_definition
```

Middle nodes have `nodeType:"automation"` and corresponding adapter IDs
`automation.<node_id>`. Exhausted/archive paths reach `end:on_flow_end`.

Adapters issue existing effects. `apply_automation_command()` shares root
mutation serialization; revisions activate at `read_definition`. Manual
admission bypasses only the automation pause gate.

### E — Queries

```python
list_run_index(..., automation_id: str|None=None,
               role: str|None=None, session_kind: str|None=None,
               changed_since: str|None=None) -> list[RunIndexRow]
latest_occurrence(automation_id: str) -> RunIndexRow|None
list_automations(*, status: str|None=None, changed_since: str|None=None,
                 cursor: str|None=None, limit: int=50) -> Page
select_session_turns(store, session_id: str, *,
                     include_occurrences: bool=True,
                     before: str|None=None, automation_id: str|None=None,
                     through_occurrence: int|None=None,
                     limit: int=50) -> list[Turn]
```

Add indexed `automation_id,role,occurrence_index,session_kind,change_cursor`.
No runtime `plane` parameter; gateway selects the store. Session kinds derive
from attribution/controller linkage, including children. Existing indexes omit
vars.

`Page={items,next_cursor,change_cursor}`; opaque restart-stable cursors,
explicit invalidation after incompatible rebuild, archive tombstones retained.

## Current code reality

Verified 2026-09-26 against `2100d1f` (paths under `src/abstractruntime/`):

- **No explicit run id.** `Runtime.start` (`core/runtime.py:1292-1484`) accepts
  `workflow, vars, actor_id, session_id, parent_run_id` only; the id comes from
  `RunState.new` → `str(uuid.uuid4())` (`core/models.py:219-238`), and the run is
  persisted with a plain `self._run_store.save(run)` (`core/runtime.py:1483`).
- **Stores upsert; there is no create-if-absent.** SQLite `save`
  (`storage/sqlite.py:522`) is `INSERT … ON CONFLICT(run_id) DO UPDATE`; the
  JSON store (`storage/json_files.py:513`) writes a temp file and `replace()`s.
  An explicit id therefore needs a new atomic create-or-load primitive on every
  `RunStore` (SQLite, JSON files, in-memory, offloading wrapper).
- **Subworkflow child id is random and saved before the parent waits.**
  `_handle_start_subworkflow` (`core/runtime.py:4164`) calls `self.start(...)`
  at `core/runtime.py:4453` (random child id, child saved immediately), then
  returns `EffectOutcome.waiting(WaitState(wait_key=f"subworkflow:{sub_run_id}"))`
  (`core/runtime.py:4470-4486`). The parent's wait is recorded in the ledger at
  `core/runtime.py:3649-3654` and committed to the run store only at
  `core/runtime.py:2696-2706`. A crash between the two leaves an orphan child and
  a parent that re-dispatches a second child on replay.
- **Per-run lock is held only by resume.** `_PerRunLocks`
  (`core/runtime.py:492-522`, process-wide RLock per run id, instance
  `_RESUME_LOCKS` at `:525`) is acquired only in `resume()`
  (`core/runtime.py:2813`). Commands and controller boundaries must join it; it
  does not serialize separate processes.
- **Terminal state is saved before the terminal event.** Every terminal path
  saves the run and only then calls `_append_terminal_status_event`
  (`core/runtime.py:2640-2641`, `:2690-2691`, `:2722-2723`; method at `:2068`).
  Occurrence completion must therefore be observed by reading the child's state
  on the parent's resume, not by relying on the event alone.
- **Conversation replay is bounded and root-only.** `session_chat_messages`
  (`session_history.py:75`, defaults 40 messages / 8 000 chars per message /
  24 000 total) rides `_best_effort_session_turns`
  (`history_bundle.py:686`), which queries
  `list_run_index(session_id=sid, root_only=True, …)` (`history_bundle.py:816`)
  and also skips runs with a `parent_run_id` on the scan path
  (`history_bundle.py:872-873`). `_classify_turn` (`history_bundle.py:708-727`)
  labels any run with `_meta.schedule` or a `scheduled:` workflow id as
  `scheduled`, and `session_chat_messages` drops `scheduled` turns
  (`session_history.py:155-158`). Occurrence child runs are invisible to every
  history path today; `select_session_turns` must replace these filters, not add
  a parallel one.
- **`on_schedule` is drift-based, not anchored.** The node handler
  (`visualflow_compiler/adapters/event_adapter.py:98-205`) computes
  `until = now + interval` on each pass (`:140-148`) and accepts `ms` and
  decimal amounts (`:134`). `schedule@1` is anchored (`anchor + tick*every`),
  integer `[smhd]` only, with coalescing; it must not reuse this time math.
- **Index filters.** `QueryableRunIndexStore.list_run_index`
  (`storage/base.py:125-141`) filters by `status, workflow_id, session_id,
  root_only` only; the SQLite implementation (`storage/sqlite.py:709`, page query
  at `:747-758`) selects fixed columns and never reads vars. JSON files
  (`storage/json_files.py:719`), in-memory (`storage/in_memory.py:68`) and
  offloading (`storage/offloading.py:419`) mirror the same signature.
- **Bundles load without the gateway.** `open_workflow_bundle`
  (`workflow_bundle/reader.py:75`) opens a directory or `.flow` zip; nothing in
  the package ships a bundle as a resource yet, and `pyproject.toml` declares no
  entry-point groups (wheel = `packages = ["src/abstractruntime"]`).
- **Command store exists and is idempotent per process.** `JsonlCommandStore.append`
  (`storage/commands.py:276`) dedups on `command_id` under an in-process lock and
  returns `CommandAppendResult(accepted, duplicate, seq)`; cursors live in
  `JsonFileCommandCursorStore` (`:183`). The runtime has no automation command
  applier; the gateway runner polls and applies commands itself.
- **Tool ceilings.** `resolve_tool_scope` (`core/tool_scope.py:20`) intersects a
  run's `allowed_tools` with every ancestor. Discussion runs must be started
  without a parent link to the automation tree so this intersection does not
  inherit the controller's or occurrence's ceiling by accident.
- No `automation` or `trigger_source` identifier exists anywhere in `src/`.

## Seams

### What this item must READ before writing (owned by other packages)

- **abstractgateway `src/abstractgateway/runner.py`** — the command lane
  (`_poll_commands` at `:1293`, `_apply_command` at `:1754`) and the
  parent-resume behaviour for `subworkflow:` waits (`:500-550`, `:2531`), so that
  controller children are driven and their parents resumed exactly once by the
  host that already does this, and so `apply_automation_command` is callable from
  `_apply_command` without a second applier.
- **abstractgateway `src/abstractgateway/hosts/bundle_host.py`** —
  `_seed_session_history` (`:2024`), the host-side session history seeding that
  `select_session_turns` must feed (Growing mode, discussion seed), so the
  gateway swaps its call rather than keeping a second history path.

Any reference to these that is not verified in the live code is written to fail
loudly and raised with the gateway owner, never hedged.

### What other packages read from this item (identifiers owned here)

- `apply_automation_command(...)` — the only automation command applier.
- `Runtime.start(..., run_id=...)` and `START_SUBWORKFLOW.payload.run_id`.
- `select_session_turns(store, session_id, *, include_occurrences, before,
  automation_id, through_occurrence, limit)`.
- `list_automations(...)`, `latest_occurrence(automation_id)`, and the new
  `list_run_index` filters (`automation_id`, `role`, `session_kind`,
  `changed_since`).
- `TriggerAdapter`, `TriggerSource`, `TriggerBinding`, `TriggerEnvelope`, and the
  entry-point group `abstractruntime.trigger_sources`.
- Bundle `abstractframework.automation-controller@1.0.0`, flow `controller`.
- The `automation.*` ledger event names and the `_meta.automation`,
  `_runtime.automation`, `_meta.occurrence`, `_meta.discussion` shapes.

These names are frozen once this item lands; the gateway integrates against them
before its own work completes.

## Acceptance criteria

- [ ] `Runtime.start(run_id=...)` and `START_SUBWORKFLOW.payload.run_id` create a
      run atomically if absent, load it when present with a matching identity,
      and raise `identity_conflict` on a mismatch; no path overwrites an existing
      run. Implemented on SQLite, JSON files, in-memory and offloading stores.
- [ ] Occurrence dispatch persists admission → creates/loads the child →
      persists the parent wait; a crash injected at every commit point yields
      exactly one child per `(revision, occurrence_index)` or `manual:command_id`,
      and recovery reconstructs the parent's wait.
- [ ] `apply_automation_command` applies revise/pause/resume/run_now/
      stop_current/archive under the root's per-run lock (shared with
      `resume()`); duplicate `command_id`s are idempotent and every outcome is
      recorded as `automation.command_result`.
- [ ] Pause/run-now/resume/revise/archive semantics match the design §2 rules
      (no catch-up on resume, at most one coalesced occurrence after downtime,
      run-now while paused stays paused, busy/exhausted/archived rejections).
- [ ] `schedule@1` and `manual@1` are discovered through
      `abstractruntime.trigger_sources`; a third-party source registered by entry
      point appears in the registry with no code change here.
- [ ] Independent mode starts each occurrence in a fresh `occurrence` session;
      Growing mode supplies bounded prior conversation from
      `select_session_turns`; only occurrence roots become turns.
- [ ] A discussion session is seeded once through the selected occurrence,
      replays like any session over several turns, runs with the target's normal
      tools, and never writes into the automation's session or context.
- [ ] `list_automations`, `latest_occurrence` and the `list_run_index`
      attribution filters return the same results on SQLite and JSON files;
      cursors survive a restart; archive tombstones remain listable.
- [ ] Replaying an automation's history performs zero provider and tool calls.
- [ ] The controller bundle ships inside the wheel and runs standalone with a
      runtime and a store, without the gateway.
- [ ] Architecture, API and backlog docs describe the contracts; no gateway,
      app or UI change is part of this item.

## Testing

Deliver, and run against both persistent stores:

- `tests/test_automation_occurrence_recovery.py` — crash around every dispatch
  commit; exactly one child, reconstructed wait.
- `tests/test_automation_commands.py` — edit/pause/manual/restart races,
  idempotent replays, revision checks.
- `tests/test_automation_session_turns.py` — multiple occurrences and discussion
  turns through `select_session_turns`, Independent vs Growing.
- `tests/test_automation_discussion_isolation.py` — discussion turns with normal
  tools never write into the automation session, context or controller state.
- `tests/test_automation_index.py` — attribution filters, `list_automations`
  paging, `latest_occurrence`, restart-stable cursors, tombstones.
- `tests/test_automation_replay_read_only.py` — history replay makes zero
  provider/tool calls.

Full suite green; no live provider required (deterministic fixture flows).

## Risks

| Risk | Mitigation | Proof |
|---|---|---|
| Duplicate children | Atomic deterministic create-or-load | `test_automation_occurrence_recovery.py`: crash around every dispatch commit |
| Command races/replay | Shared root lock, idempotent outcomes, optional revision checks | `test_automation_commands.py`: edit/pause/manual/restart races |
| Lost/duplicated context | Shared selector, persistent discussion seed | `test_automation_session_turns.py`: multiple occurrences/discussion turns |
| Discussion writes back into the automation | Discussion is a separate session (fork); no parent link into the automation tree | `test_automation_discussion_isolation.py` |

Additional runtime-owned risks: the per-run lock is process-local, so two
runners on one store can still race (v1 supports one active runner per store;
the general fix is
[0046](runtime_systemic_reliability/0046_per_run_driver_exclusion.md)); a
long-lived controller root accumulates vars and ledger records (keep controller
state bounded; see
[0053](runtime_systemic_reliability/0053_bounded_run_vars_growth.md)).

## Dependencies and ADR status

- Upstream: none. Downstream: the gateway Automations work depends on this item
  and must integrate against it before completing.
- Related runtime items: 0045 (crash-ordering invariant — the dispatch ordering
  here is an instance of it), 0046, 0053, and
  [0848](0848_automations_v2_external_event_inbox_and_run_triggers.md) (v2).
- ADR impact: likely one new ADR — "an automation is its controller root run;
  occurrences are deterministic child runs" — to be written with the
  implementation, not before.
- Effort estimate from the design: 8–12 engineer-days (±50%).

## Related

- abstractframework backlog 0928 (Automations umbrella).

## Contracts pass (2026-09-27)

Final contracts: untracked/design/automations-CONTRACTS.md (root repo; rev 2 with Astra turn-6 amendments 1–11). They supersede the contract text copied above; earlier text is kept as history. Concrete changes for this item:

- Workspace question closed by operator ruling 8: discussions run on the occurrence's workspace READ-ONLY. Implement `WorkspaceScope.read_only` (host key `workspace_read_only`): refuse every `write`/`exec`-classified tool (incl. `write_file`, `edit_file`, `execute_command`, `shell_exec`, `local_helper_start`, abstractagent `execute_python`) and any unclassified tool via one `TOOL_EFFECT_CLASSES` table; VisualFlow writer nodes (`visual/executor.py:810`) refuse; the ambient propagation (`compiler.py:2812`) carries the flag and overrides node inputs; `merge_builtin_workspace_protection` propagates it before its early return (`workspace_paths.py:239`); a missing root is never created under read-only.
- Discussion: root run, `start_discussion(...)`; the persisted discussion root is the session-policy anchor — `Runtime.start` resolves `session_attribution` for every root start with a `session_id` and enforces attribution + read-only workspace, failing closed.
- `create_if_absent` on all four stores + capability preflight; JSON = same-directory temp + `os.link`; identity adds `session_id` and `_meta.creation_digest`; repair rebuildable indexes after recovery; claim process-crash recovery only (power-loss needs file+dir fsync).
- `TriggerAdapter` has six typed methods (`validate`, `initial_state`, `prepare`, `admit`, `rearm`, `normalize`); `TriggerWait` adds `idle`; one-shot = single tick T0 then exhausted; a changed trigger config gets a new `binding_id`; built-in sources missing = error, third-party = `available:false`.
- Controller persists `abstractframework.automation-controller@1.0.0:controller`; directory bundle as package data; loaders `controller_bundle_path`/`controller_workflow_spec`.
- Decision protocol for every controller transition: reconcile by `_runtime.automation.state_version` → exact `LedgerStore.find_by_idempotency_key` (new; never the tail window) → decide (incl. `expected_revision`) → append decision with delta → apply + save. Crash tests before/after decision append, state save, command-cursor advance.
- Retry `policy.retry` (3 attempts, 30s×2 capped 10m); `prepared` inputs frozen at admission; one deterministic child per attempt (`…:a{n}`); output normalized by the target's output contract before reading `notify`; `notify:false` convention removed (quiet by default); one attention record per logical occurrence (`automation.completed`).
- Human attention = interactive USER (not paused) and EVENT waits with prompt/choices; controller and pause waits excluded.
- Strict history mode `session_chat_messages(..., strict=True)` for `prepare_context` and discussion seeding; selector `select_session_turns(..., until_ms, include_drafts)` returns `list[RunState]` and owns `_best_effort_session_turns`' selection block.
- Index columns `automation_id, role, occurrence_index, session_kind` (+ `legacy_schedule` role), SQLite guarded backfill, JSON memo v2; `_meta` identity never offloaded; offloaded seeds/outputs resolved. No change cursor in v1 (`changed_since` unsupported); new `list_attention` (oldest unseen first).
- One controller-writer process per store; lock held for the whole controller tick and command application; async dispatch; pause gates scheduled admission only; admission only from persisted due tick or `manual_pending` (stale wakes re-arm).
- New tests: `test_trigger_schedule.py`, `test_automation_retry.py`, `test_workspace_read_only.py`, `test_controller_bundle_packaged.py` (plus the six listed above).

## Completion report (2026-09-27)

**Status: completed — UNRELEASED.** Local commits on `main`, no version bump (pyproject still `0.5.1`), not pushed. The
release is the first step of the framework's wave (root backlog 0941). Umbrella record: abstractframework backlog 0928
(completed).

**Commits:** `79d9bf6` … `d02578a` (26 commits). Missions R1 (storage, attribution, history, read-only, anchor) and R2
(automations, triggers, controller, commands, attention, D1).
- **Seams and storage:**
  - `79d9bf6`: `run_mutation_lock` held for the whole tick.
  - `e1ee1a7`: `create_if_absent` on every store; `Runtime.start(run_id=)`; `START_SUBWORKFLOW.payload.run_id`.
  - `efb3e44`: index columns, turn-root listing, `list_automations` / `latest_occurrence`.
  - `e8af087`: `latest_occurrence` is one index seek; stale temp files swept on open.
- **History:**
  - `93c540f`: `select_session_turns`.
  - `cebf348`: strict `session_chat_messages`, discussion seed.
- **Read-only:** `ab30729` `workspace_read_only` + `TOOL_EFFECT_CLASSES`.
- **Discussion anchor:** `035cfdf` `session_attribution` + the `Runtime.start` restamp.
- **Automations:**
  - `58ade60`: trigger registry, `schedule@1`, `manual@1`.
  - `3a91de9`: the automation node type.
  - `478441b`: contracts, decision ledger, controller bundle, commands, attention, service.
  - `53e00e8`, `866c13c`, `ab8b9a0`, `f43b708`, `3f58bf8`, `8df97a9`: tests, the crash matrix and the dispatch-from-frozen
    fix.
  - `a7138b0`: docs.
- **Decision D1:**
  - `3cc9900`: `policy.tool_approval` and typed `pending_waits`.
  - `ba2b303`: typed wait details.
- **Review fixes:**
  - `e690b55`: H2, M1, D1-M1, M3 and `actor_id`.
  - `af2de4a`: 44 F1/F2 and **45 H1, the release blocker**. The JSON start cost fell from a 269 ms median to 0.34 ms at 20k
    runs.
  - `b000036`: J51-1, the creation journal, `warm_session_index`.
  - `aa0b1f1`: one `automation_status` rule.
  - `d02578a`: J53-1 / J53-2.
- **Docs:** `a7a1fae` (`docs/automations.md`: controller, triggers, context, discussions, tool approval, storage
  guarantees).

**Tests.** The full suite is **2957 passed / 26 skipped** at `d02578a`. The acceptance criteria above map onto:
`test_automation_{occurrence_recovery, controller_recovery, commands, session_turns, discussion_isolation, index,
replay_read_only, retry, growing_context, definition, controller, status, tool_approval}.py`,
`test_trigger_{registry, schedule_math}.py`, `test_workspace_read_only.py`, `test_controller_bundle_packaged.py`,
`test_discussion_session_anchor.py` and `test_session_start_cost.py`. Every test runs on JSON and SQLite, with SIGKILL
crash points around every controller decision (18/18 held, reviews 45 and 47).

**Reviews** (`untracked/missions-2026-09-25/REVIEW/` in the root repo):
- 43 GO.
- 44 GO for history and read-only, NO-GO for the discussion path (F1, F2) → fixed `af2de4a`.
- 45 GO for the controller core, NO-GO for discussion (H2), and **H1 release blocker** → fixed `e690b55` / `af2de4a`.
- 45 addendum (D1) GO.
- Job 50 (`e690b55`) GO.
- Job 51 (`af2de4a`) GO; history is byte-identical to 0.5.1 for ordinary sessions.
- Job 53 (`b000036`) GO. J53-1/J53-2 were fixed afterwards in `d02578a` (not re-reviewed).

**E2E.** Root `untracked/missions-2026-09-27/E2E/REPORT.md` passed 9/9 with the operator's MLX model. It covered
deterministic child ids (also after a target revise), coalescing, growing context, `kill -9` twice with one child per
tick, and a replay with 0 provider/tool calls and 201 files byte-identical.

**Acceptance criteria:** all met as written, with these differences:
- Discussions run with the target's tools on a read-only workspace (ruling 8).
- The controller is driven by the gateway's ordinary runner. `drive_automation` exists for hosts without one; the gateway
  does not use it.
- One controller-writer process per store (documented).

**Decisions recorded here** (full list in root 0928):
- Discussion sessions are `discussion-session:<uuid5(automation_id, "discuss:"+request_id)>`.
- An independent occurrence's session is its first attempt's run id.
- Tool classes use the spelling `exec`.
- `pending_waits` is live, not stored.
- Under `auto` the grant covers the target's `allowed_tools`, else every classified tool (65). MCP and unclassified tools
  still ask.
- Bounds: `every` ≤ 366 d, `count` ≤ 1,000,000.

**ADR impact:** the ADR this item anticipated ("an automation is its controller root run; occurrences are deterministic
child runs") is **not written yet**. It stays open, recorded in root 0928's residuals.

**Residuals and follow-ups:**
- The store-level exact-key lookup and read cost (review 45 M2) → root 0937, together with runtime 0047 / 0068.
- Duck-typed stores and `Runtime.start` → root 0938.
- JSON order by mtime (review 43 F5) → root 0939.
- Structured failure reason codes on attempts (every failure is `occurrence_failed` at the gateway).
- Review 45 lows:
  - L1: silent cuts at 280 / 2,000 characters (`automations/controller.py:449`);
  - L2: re-anchoring on a resend without `start_at`;
  - L3: a possible second coalesced record;
  - L5: a failed occurrence is absent from its discussion seed.
- Review 44 F5/F6 notes.
- J50-1: a release-notes line.
- abstractagent 0034: the agent-side tool effect declaration is still open. The runtime's central table covers today's
  tools.
- `resolve_discussion_root` is O(turns) per discussion turn.

**Next:** [0848](../planned/0848_automations_v2_external_event_inbox_and_run_triggers.md) (v2) is unchanged.
