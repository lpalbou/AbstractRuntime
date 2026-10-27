# ADR 0009: The memory facade is the complete host surface for AbstractMemory

Status: Accepted.
Date: 2026-10-27

## Context

AbstractMemory ADR 0003 (one semantic temporal graph, every memory tied to the ledger, every change a revision)
rules that every memory a mind holds names the ledger sources it came from, and that the verbatim of any source can
be reconstructed. AbstractMemory returns references; it never reads a ledger. The ledger, the artifacts, the diary
chain and the event inbox belong to this package, so two jobs land here:

- **Stamping.** The runtime performs memory writes for hosts (`MEMORY_FORM`, `MEMORY_KG_ASSERT`, the `DIARY_WRITE`
  projection). The audit of 2026-10-27 (`abstractframework/untracked/memory-transition/audit/abstractruntime_entity.md`)
  found that none of them wrote a ledger row address: they wrote `run_id`/`turn_id` (`writer_run_id` on the KG lane),
  and on the home-direct entity path the `run_id` is a synthetic `chat-<session_id>` that no store can load. The
  handlers could not do better: an effect handler received `(run, effect, default_next_node)` and nothing told it
  which ledger row its own step had.
- **Reconstruction.** The only addressable ledger unit is `(run_id, seq)` (`step_id` is shared by a STARTED row and
  its terminal row). `LedgerStore` had no read by address, `list()` did not return `seq`, terminal rows hold `$slim`
  markers instead of payload bytes the STARTED row already holds, and oversized leaves may be offloaded to the
  artifact store. Runs, ledgers, artifacts and inbox payloads are deletable.

And one boundary: ADR 0008 decision 2 already names AbstractMemory as a package a host must not import, yet no
AbstractMemory facade existed, so AbstractGateway imports `abstractmemory` directly (34 lines across 8 files in the
audit). The hole is the same as the one ADR 0008 closed for AbstractCore: a facade that lags what hosts need, with
no rule saying whose job it is to close the gap.

## Decision

1. **One host surface.** `abstractruntime.integrations.abstractmemory.memory_facade` (`MemoryFacade`) and the existing
   effect-handler builders of `abstractruntime.integrations.abstractmemory` (`build_memory_kg_effect_handlers`,
   `build_memory_seam_effect_handlers`, `build_entity_brain_effect_handlers`) are the ONLY surface through which a
   host reaches AbstractMemory. The facade opens a mind (`MemoryFacade.open_mind(path)`: the SQLite triple store,
   journal and `abstractmemory.graph.MemoryGraph` of one home), exposes the graph reads (`history`, `sources_of`,
   `derived_from_source`), stamps provenance (`stamp`) and reconstructs sources (`reconstruct_source`). It is
   complete by contract: a name or capability a host needs and cannot reach through it is a runtime defect, fixed by
   adding it here and cutting a patch release. A host never imports `abstractmemory` to work around a gap (ADR 0008
   decision 2 applies unchanged).
2. **The ledger row address is `(run_id, seq)`.** `LedgerStore.get_record(run_id, seq)` returns one record with its
   `$slim` markers resolved against the STARTED row of the same `step_id` and, under `OffloadingLedgerStore`, its
   offloaded `{"$artifact": id}` leaves rehydrated; `list(run_id)` records carry `"seq"`. `seq` is added on read and
   never persisted inside the record (the hash chain excludes it). SQLite uses its primary key; the JSON-file backend
   assigns `seq` = the 1-based position of the record in its file (the order `list()` returns). `append()` returns
   the seq of the row it wrote when the backend knows it.
3. **The runtime stamps every memory write it performs.** The runtime publishes the current step to the handler it is
   executing (`abstractruntime.core.step_context`: `run_id`, `seq` of the step's STARTED row, `step_id`, `node_id`,
   `attempt`, `effect_type`). The memory writers stamp through `MemoryFacade.stamp`: a `step` ref `(run_id, seq,
   step_id)` for the writing effect, an `artifact` ref for each verbatim artifact (`payload_ref`, `span_id`), a
   `diary_entry` ref for a diary projection. The new `sources` list is additive; the legacy flat keys (`run_id`,
   `turn_id`, `writer_run_id`, ...) stay so existing readers keep working.
4. **A synthetic run id is never a source.** A run id that no store can load (`chat-<sid>`, `diary:<eid>` used as a
   run, `global_memory`) is never stamped as a `run` or `step` ref; a `seq` is never guessed. When the runtime cannot
   name a ledger source, the stamp carries what is real (an artifact, a diary entry) or nothing new, and the graph
   write path (`MemoryGraph.apply_revision`, `ProvenanceRequired`) refuses an ungrounded write. That refusal is the
   intended outcome; fabricating a ref to pass it is a defect.
5. **Reconstruction never raises for a missing source.** `MemoryFacade.reconstruct_source(ref)` answers
   `{"status": "ok", "kind", "content", "metadata"}`, `{"status": "source_gone", "reason"}` when the run, ledger row,
   artifact, diary entry or event no longer exists, or `{"status": "unsupported", "reason"}` for a kind the host did
   not give the facade a store for (or one with no indexed path, `message` today).
6. **The facade is additive within a minor line**, like ADR 0008 decision 4: a name, once exported, is not removed
   without a deprecation note and a major or minor bump.

## Consequences

### Positive
- Every memory the runtime forms in a real run can be traced to the exact ledger row (and artifact) it came from,
  and that row reads back verbatim with one call.
- A host's import-boundary test for AbstractMemory becomes a true gate, as it is for AbstractCore.
- Deleted sources are a typed answer, so a reader can say "the source of this memory is gone" instead of failing.

### Negative
- The facade needs AbstractMemory's graph package (`abstractmemory.graph`, AbstractMemory 0.4.0) for `open_mind`,
  `stamp` and the graph reads; the memory writers stamp through it, so the runtime's AbstractMemory floor moves to
  0.4.0 with the release that carries this ADR. `get_record` and the reconstruction of steps and artifacts need no
  AbstractMemory at all.
- Memory written on the home-direct entity path (synthetic `chat-<sid>` runs) carries artifact refs only until that
  path runs as real runs; derived writes outside any effect (world models, tending, the night passes) carry no
  ledger source until backlog 0854 lands.

### Neutral
- `seq` is read-side metadata: persisted bytes, hashes and the diary chain are unchanged.
- The step context is out of band (a `ContextVar`, like the progress callback): effect payloads stay JSON and a
  handler called directly, outside a runtime step, sees no step.

### Open decisions (operator)
- **Durability of a JSON-file row address.** On the JSON-file ledger (the gateway's default) `seq` is a line
  position: a rewrite of the file (a dropped line, a compaction) shifts every later seq. The facade never answers
  `ok` for a shifted row (a step ref resolves only to a STARTED row of the named step, and a stamped digest is
  verified), but the memory's source then reads `source_gone`. Two ways to make the address durable, not chosen
  here: (a) persist `seq` inside each JSON-file record at append time (the record bytes then carry their address;
  rewrites keep it; the hash chain must keep excluding it, and readers of old files fall back to the position);
  (b) move plane ledgers to the SQLite backend, whose `seq` is a primary key (the JSON-file backend stays for
  development). Until decided, JSON-file ledgers are append-only by contract: nothing rewrites them.
- **Memory ownership.** `owner_id` is a partition string derived from the run (`run`: the run id, `session`:
  `session_memory_<sid>`, `global`: the plane's global owner). The runtime refuses a payload `owner_id` the run does
  not derive, but a session id is still caller-chosen upstream. The full fix (owner = the principal, scope = the
  kind of memory) is an operator decision (AbstractMemory ADR 0003 decisions 2/13 as amended).

## Enforcement
- Code review: a runtime memory write site that does not stamp through `MemoryFacade.stamp`, or that builds a
  `run`/`step` ref from a run id it cannot load or a seq it did not get from the step context, is rejected.
- A host need that the facade does not cover becomes a runtime backlog item or PR before the host ships; the host's
  changelog names the runtime version that closes it.
- The kernel import boundary (`tests/test_runtime_install_boundary.py`) stays green: AbstractMemory is imported
  lazily, inside functions, everywhere in this package, the facade included; the facade imports no AbstractCore.
- AbstractGateway burns its direct `abstractmemory` imports down onto this facade (its boundary test, never an
  exemption for a name the facade could carry).

## Validation
- `tests/test_ledger_get_record.py`: `get_record` on the SQLite and JSON-file backends (and under
  `OffloadingLedgerStore`) returns the terminal row of a slimmed step with its payload rebuilt from the STARTED row;
  `list()` carries `seq`; the diary hash chain still verifies.
- `tests/test_memory_facade.py`: `reconstruct_source` for a step whose terminal row was slimmed, an artifact, a
  deleted artifact (`source_gone`), a run, a diary entry, an event, a message (`unsupported`); `stamp` adds a step
  ref for a real run and none for a synthetic one; facade re-exports keep identity.
- `tests/test_memory_step_provenance.py`: an episode formed by `MEMORY_FORM` in a real run carries a `step` ref whose
  `(run_id, seq)` resolves through `get_record` to that `MEMORY_FORM` step; a diary projection carries a
  `diary_entry` ref; the synthetic chat path carries artifact refs only; `MEMORY_KG_ASSERT` carries the step ref.

## Backlog links
- Implementation: [0853](../backlog/completed/0853_memory_facade_and_ledger_provenance.md) (memory facade, ledger row
  address, step context, stamping).
- Follow-up: [0854](../backlog/planned/0854_entity_direct_memory_writes_provenance.md) (entity writes outside any
  effect: world models, tending, the night passes).
- AbstractMemory backlog 0056 (`abstractmemory.graph`: `SourceRef`, `Provenance`, `MemoryGraph`).

## Related
- ADR 0003 (tamper-evident hash chain: `seq` is excluded from the record hash), ADR 0008 (integration facades are the
  complete host surface; this ADR is its AbstractMemory counterpart).
- AbstractMemory ADR 0002 (AbstractMemory owns the graph), ADR 0003 (one graph, ledger provenance, revisions).
- Root ADR 0007 (active context vs stored memory), ADR 0009 (`span_id` = `artifact_id`).
- AbstractGateway ADR 0001 (lower-level access through runtime facades).
