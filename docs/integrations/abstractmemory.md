# AbstractMemory integration

AbstractRuntime is the only way a host (AbstractGateway, AbstractNote, an entity home) reaches AbstractMemory
([ADR 0009](../adr/0009_memory_facade_is_the_complete_host_surface_for_abstractmemory.md)). The surface is:

- `abstractruntime.integrations.abstractmemory.memory_facade`: `MemoryFacade`, `is_real_run`, and the AbstractMemory
  names hosts use, re-exported unchanged (`GATEWAY_NAMES`).
- The effect-handler builders of `abstractruntime.integrations.abstractmemory`: `build_memory_kg_effect_handlers`
  (`MEMORY_KG_ASSERT` / `MEMORY_KG_QUERY` / `MEMORY_KG_RESOLVE` / `MEMORY_REVISE`), `build_memory_seam_effect_handlers`
  (`MEMORY_RECALL` / `MEMORY_ACCESS` / `MEMORY_FORM` / `MEMORY_ADJUST` / `MEMORY_APPRAISE`) and
  `build_entity_brain_effect_handlers` (entity homes).

A name or capability a host needs and cannot reach here is a runtime defect: it is added here in a patch release; a
host never imports `abstractmemory` to work around it.

Requires AbstractMemory 0.4.0 or newer (`abstractmemory.graph`). AbstractMemory is imported lazily everywhere:
`import abstractruntime` and `import abstractruntime.integrations.abstractmemory.memory_facade` never import it.

## The ledger row address

Every memory the runtime writes can name the ledger row it came from. A row is addressed by `(run_id, seq)`:

- `LedgerStore.get_record(run_id, seq) -> dict | None` returns one record ready to read: a terminal row's `$slim`
  markers are resolved against the STARTED row of the same `step_id`, and under `OffloadingLedgerStore` offloaded
  leaves are rehydrated. `None` when the row does not exist (deleted ledger, wrong seq).
- `list(run_id)` records carry `"seq"`; `append(record)` returns the seq it wrote.
- SQLite uses its primary key. The JSON-file backend's seq is the record's 1-based position in its file; a per-run
  line-offset index makes `get_record` a seek plus one line parse (see Performance below).
- `seq` is read-side metadata: it is never written inside the record and the hash chain excludes it.

## The step context

While the runtime executes an effect handler it publishes the step to it, out of band
(`abstractruntime.core.step_context`):

```python
from abstractruntime.core.step_context import current_step_context

ctx = current_step_context(run.run_id)  # None outside a runtime step, or inside another run's step
# ctx.run_id, ctx.step_id, ctx.seq (the STARTED row of this attempt), ctx.node_id, ctx.attempt, ctx.effect_type
```

`seq` is what the ledger store reported for this attempt's STARTED row (`None` when a backend does not know it); it is
never guessed. A handler called directly (the home-direct entity driver, a test) sees `None`. A nested runtime driving
a child run inside a handler shadows the context and restores it afterwards.

## Provenance stamping

The runtime stamps every memory write it performs through `MemoryFacade.stamp` / `stamp_current_step`. The stamped
provenance keeps the legacy flat keys (`run_id`, `turn_id`, `writer_run_id`, `span_id`, `entry_id`, ...) so existing
readers keep working, and ADDS a `sources` list of SourceRef dicts (AbstractMemory contract §1):

| Writer | `sources` |
|---|---|
| `MEMORY_FORM` | `step` ref of the forming step; `artifact` ref of the verbatim (stored by the handler, or a caller `payload_ref` the artifact store holds) |
| `MEMORY_KG_ASSERT` | `step` ref of the asserting step; `artifact` ref for `span_id` |
| `DIARY_WRITE` projection | `diary_entry` ref (`run_id` = `diary:<entity_id>`); for a public entry also the `step` ref of the `DIARY_WRITE` step. A private projection names the entry only (it carries no origin by design; the chain entry holds it) |
| `MEMORY_REVISE` | `step` ref of the revising step; one `artifact` ref per `artifact_ids` entry; the caller's own artifact/message/event/diary_entry refs |

Rules (ADR 0009 decision 4):

- A synthetic run id never becomes a `run` or `step` ref: `chat-<session_id>` (home-direct entity driver),
  `diary:<entity_id>`, `session_memory_<sid>`, the global memory owner, or (when the facade has a `run_store`) any run
  the store cannot load. On the synthetic chat path a memory carries its artifact or diary-entry ref only.
- No `step` ref without a seq from the step context. A real run outside any step gets a `run` ref.
- When a `step` ref names a run, the `run` ref of the same run is dropped (the step is the precise address).
- Caller-supplied refs (a `sources` list already in the provenance) are screened: a `run` or `step` ref is dropped
  (the runtime stamps those), a synthetic run id is stripped from any other ref (a diary entry keeps its
  `diary:<entity_id>` chain id), a ref AbstractMemory's `SourceRef` refuses is dropped, and a flat `run_id` that is not
  the run being stamped stays a flat key, never a ref. Each rejection is recorded (`extra.rejected_sources` in the
  graph shape, a `rejected_sources` flat key otherwise); `stamp` never raises for a caller ref.
- `is_real_run` uses AbstractMemory's synthetic prefixes (`chat-`, `diary:`, `session_memory_`, `global_memory`),
  case-insensitively, plus `ABSTRACTRUNTIME_GLOBAL_MEMORY_RUN_ID`. Without a run store only the syntax is checked:
  any other string counts as real, so pass the run store wherever one exists.

## `MemoryFacade`

```python
from abstractruntime.integrations.abstractmemory.memory_facade import MemoryFacade

facade = MemoryFacade.open_mind(home / "memory.sqlite3",            # SQLiteTripleStore + SQLiteJournal + MemoryGraph
                                ledger_store=ledger, run_store=runs, artifact_store=artifacts,
                                diary_store=diary, event_inbox=inbox)
facade.graph, facade.store, facade.journal

facade.stamp(provenance, run_id=..., seq=None, step_id=None, artifact_ids=(), runtime_id=None) -> dict
facade.stamp_current_step(provenance, run_id=..., artifact_ids=()) -> dict
facade.apply_revision(revision_or_dict) -> dict          # RevisionResult as a dict; raises ProvenanceRequired
facade.history(memory_id) / facade.sources_of(memory_id) / facade.derived_from_source(ref)
facade.reconstruct_source(ref, *, limit=200) -> dict     # limit bounds a run's ledger records (max 1000)
```

`MemoryFacade(graph=, ledger_store=, run_store=, artifact_store=, diary_store=, event_inbox=, runtime_id=)` takes only
the stores a host has; every one is optional.

`reconstruct_source(ref)` never raises for a missing source:

| `ref.kind` | Reads | `status: "ok"` content |
|---|---|---|
| `step` | `ledger_store.get_record(run_id, seq)` (a `step_id` in the ref must match) | the record, payload verbatim |
| `run` | `run_store.load` (+ `ledger_store.list`) | `{"run": ..., "ledger": [the first `limit` records]}`; `metadata.ledger_records` / `ledger_returned` / `ledger_left_out` / `truncated` |
| `artifact` | `artifact_store.load` | text (utf-8) or base64 (`metadata.encoding`) |
| `diary_entry` | `diary_store.get_record` | the entry; `metadata.record_hash` / `prev_hash` |
| `event` | `event_inbox.get` | the event payload |
| `message` | none | `unsupported` (a message id is not a ledger key today) |

`{"status": "source_gone", "reason"}` when the run, row, artifact, entry or event no longer exists (or a pruned event);
`{"status": "unsupported", "reason"}` for a kind with no store here, an invalid ref or limit, or a ref of another
runtime: give the facade the `runtime_id` of the ledger it reads (`MemoryFacade(runtime_id=)`,
`open_mind(path, runtime_id=)`, the plane's or the home's) and a ref naming another one answers
`"ref names runtime <x>, this facade resolves <y>"`. Every ref the facade stamps (lifted legacy refs included)
carries that `runtime_id`; the handler builders take `runtime_id=` too (`build_memory_kg_effect_handlers`,
`build_memory_seam_effect_handlers`).

Re-exported AbstractMemory names (`facade.X is abstractmemory.X`, resolved on first use): `memory_facade.GATEWAY_NAMES`
(budgets, replay stream, spark templates, stores and journal, queries, night passes, embeddings pin, drives, disposal,
diary reads `open_questions` / `open_problems` / `open_ideas`,
and the graph names `SourceRef`, `Provenance`, `ProvenanceRequired`, `InvalidSourceRef`, `Revision`, `RevisionOp`,
`RevisionResult`, `MemoryGraph`).

## `MEMORY_REVISE`: memory reconsideration as a workflow step

`EffectType.MEMORY_REVISE` applies ONE revision of a mind (AbstractMemory ADR 0003: assert / close / narrow ops,
atomic, idempotent on `revision_id`) through `MemoryGraph.apply_revision`, with the step that applied it stamped as its
ledger source. Handler: `build_memory_kg_effect_handlers(store=, run_store=, now_iso=, journal=None)`; the
`MemoryGraph` is opened once per builder call. A store that is not a graph home (`LanceDBTripleStore`) answers a failed
outcome naming AbstractMemory backlog 0057.

Payload (a Revision WITHOUT the runtime-owned part of its provenance):

| Key | Required | Meaning |
|---|---|---|
| `ops` | yes | `[{"op": "assert", "payload": TripleAssertion dict}, {"op": "close", "payload": {assertion_id, valid_until, kind: retract\|supersede\|reinterpreted\|merged, replacement_ids, reason}}, {"op": "narrow", "payload": {assertion_id, valid_until, reason}}]` |
| `scope` | no | memory scope, default `"run"` |
| `owner_id` | no | default: resolved from `scope` like the KG lane (`run` / `session` / `global`); required for any other scope |
| `cause` | no | `ingest` (default), `reinterpret`, `merge`, `unmerge`, `ripple`, `person`, `maintenance`, `import`, `migration` |
| `revision_id` | no | default `rev_<sha256(run_id, node, scope, owner, cause, ops, provenance.sources, provenance.derived_from, artifact_ids, occurred_at)[:32]>`: a retried attempt is a no-op; the same ops from a new source are a new revision |
| `occurred_at` | no | observation time (ISO UTC), default now |
| `recorded_at` | no | record time, default now |
| `summary` | no | free text |
| `artifact_ids` | no | artifacts the revision came from (stamped as `artifact` refs) |
| `provenance` | no | `{derived_from, actor (default "runtime"), reader, reason, extra, sources}`; `sources` may name artifact / message / event / diary_entry refs, never `run` / `step` (runtime-owned) |

Unknown payload or provenance keys fail the effect. Result (completed):

```json
{"ok": true, "revision_id": "...", "applied": true, "assertion_ids": ["..."], "closed": [], "narrowed": [],
 "skipped": [], "seq": 1, "scope": "global", "owner_id": "global_memory",
 "sources": [{"kind": "step", "run_id": "...", "seq": 7, "step_id": "..."}]}
```

`applied` is false when the `revision_id` was already applied. An ungrounded revision (no source and no
`derived_from`, e.g. a handler called outside any run) fails with `MEMORY_REVISE refused (ProvenanceRequired): ...`;
no ref is invented to pass it.

### VisualFlow node `memory_revise`

Node type `memory_revise` (compiled by `abstractruntime.visualflow_compiler`) produces `MEMORY_REVISE`. Input pins,
forwarded verbatim (an unset pin is dropped so the handler default applies):
`ops`, `scope`, `owner_id`, `cause`, `revision_id`, `occurred_at`, `recorded_at`, `summary`, `artifact_ids`,
`provenance` (`visualflow_compiler.adapters.effect_adapter.MEMORY_REVISE_PINS`). Output pins: `assertion_ids` (the
node's mapped value), `closed`, `narrowed`, `skipped`, `sources`, `revision_id`, `applied`, `ok`, `raw`. An editor
palette (AbstractFlow) adds the node with exactly these pins.

## Performance

`get_record` / `find_started_record` at 10,000 rows (~10 MB JSON-file ledger; Apple Silicon, warm page cache;
method and script in backlog 0853):

| | JSON files before | JSON files after | SQLite |
|---|---|---|---|
| `get_record` | 58 ms (full parse) | 0.05 ms | 0.01 ms |
| `find_started_record(step, before_seq=terminal)` | 72 ms | 0.08 ms | 0.01 ms (was 3.5 ms) |
| first read in a cold process | 58 ms | 48 ms (builds the index once) | 0.3 ms |

## See also

- [ADR 0009](../adr/0009_memory_facade_is_the_complete_host_surface_for_abstractmemory.md), backlog
  [0853](../backlog/completed/0853_memory_facade_and_ledger_provenance.md) and
  [0854](../backlog/planned/0854_entity_direct_memory_writes_provenance.md).
- [API reference](../api.md#abstractmemory-bridge-kg-effects).
