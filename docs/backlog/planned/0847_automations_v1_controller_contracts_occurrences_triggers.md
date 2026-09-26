# 0847-abstractruntime: [FEATURE] Automations v1: controller bundle, definition/state contracts, deterministic occurrence creation, trigger-source registry, session-turn selector, index queries

> Created: 2026-09-26
> Status: Planned
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
