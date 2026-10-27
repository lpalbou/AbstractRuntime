# 0853 — Memory facade and ledger provenance: the host surface for AbstractMemory, the ledger row address, and a step ref on every memory write

**Status**: in progress (branch `feat/memory-graph-facade`) · **Priority**: P1 · **Created**: 2026-10-27
**Cross-package**: AbstractMemory ADR 0003 and backlog 0056 (`abstractmemory.graph`: `SourceRef`, `Provenance`,
`MemoryGraph`; AbstractMemory 0.4.0), AbstractGateway memory routes (`/api/gateway/memory/*` over this facade) and its
import-boundary burn-down, AbstractNote thin-client slice (step and artifact refs on note revisions). ADR:
[0009](../../adr/0009_memory_facade_is_the_complete_host_surface_for_abstractmemory.md).

## Why
Operator goal (2026-10-27, AbstractMemory ADR 0003): every derived memory is tied to the ledger's messages, events
and artifacts, so the verbatim source of any understanding, or of any change of understanding, can be reconstructed.
AbstractMemory stores references; resolving them is the runtime's job, because the runtime owns the ledger, the
artifacts, the diary chain and the event inbox. Hosts must reach AbstractMemory through the runtime (ADR 0008
decision 2), and no AbstractMemory facade existed.

## Current code reality (2026-10-27, `main` 93c45b4)
- `storage/base.py` `LedgerStore`: `append`, `list(run_id)`, `find_completed_result`; no read by address. SQLite
  (`storage/sqlite.py`) has a `seq` primary key that `list()` does not return; JSONL has positions only.
- A step is a STARTED row and a terminal row sharing one `step_id` (`core/runtime.py`, `_execute_effect_with_retry`);
  the terminal row is slimmed (`storage/ledger_slim.py`: `$slim` markers resolved against the STARTED payload);
  oversized leaves are offloaded (`storage/offloading.py`, rehydrated only by `find_completed_result`).
- Effect handlers receive `(run, effect, default_next_node)`; nothing tells them their own ledger row. The publish
  handler gets `step_id` as an extra argument.
- Memory provenance written today (audit 2026-10-27, `untracked/memory-transition/audit/abstractruntime_entity.md`
  §2): `MEMORY_FORM` `run_id`/`turn_id` + the verbatim artifact as `payload_ref` (`seam_handlers._handle_form`),
  `MEMORY_KG_ASSERT` `writer_run_id`/`writer_workflow_id`/`span_id` (`effect_handlers._handle_assert`), diary
  projections `{source, entry_id}` + `origin` (`identity_support.project_diary_entry`). No ledger row address; on the
  home-direct entity path the `run_id` is the synthetic `chat-<session_id>`.
- AbstractGateway imports `abstractmemory` directly in 34 lines across 8 files.

## Scope
In:
- `LedgerStore.get_record(run_id, seq)` (base default over `list()`, SQLite by primary key, JSONL by position,
  `OffloadingLedgerStore` rehydrating, the decorators delegating); `$slim` markers resolved against the STARTED row of
  the same `step_id`; `list()` records carry `"seq"` (added on read, excluded from the record hash); `append()`
  returns the seq it wrote when the backend knows it.
- `abstractruntime.core.step_context`: the runtime publishes the current step `(run_id, seq, step_id, node_id,
  attempt, effect_type)` to the handler it executes (a `ContextVar`, out of band like the progress callback).
- `integrations/abstractmemory/memory_facade.py`: `MemoryFacade` (`open_mind`, `graph`/`store`/`journal`, `stamp`,
  `reconstruct_source`, `history`/`sources_of`/`derived_from_source`), `is_real_run`.
- Stamping in `MEMORY_FORM`, `MEMORY_KG_ASSERT` and the `DIARY_WRITE` projection.
- ADR 0009, docs (`api.md`, `integrations/abstractmemory.md`), CHANGELOG.

Out:
- Writes outside any effect (world models, tending, the night passes): [0854](../planned/0854_entity_direct_memory_writes_provenance.md).
- Moving the memory writers onto `MemoryGraph.apply_revision` (AbstractMemory's convergence track); the gateway
  routes and the gateway's import burn-down (AbstractGateway).
- A hash chain on conversation ledgers; real runs behind the home-direct entity driver.

## Validation
- Both backends (and the offloading decorator): `get_record` of a slimmed terminal row returns the STARTED payload
  bytes; a mutant that returns the slimmed row unresolved fails; `list()` carries `seq`; the diary chain verifies.
- `reconstruct_source` on real temporary stores (`SqliteRunStore`, `SqliteLedgerStore`, `FileArtifactStore`,
  `DiaryStore`, `JsonFileEventInbox`): ok / `source_gone` after deletion / `unsupported`.
- A real `Runtime` run: the episode formed by `MEMORY_FORM` carries a `step` ref that resolves through `get_record` to
  that step; the KG assertion carries one; a diary projection carries a `diary_entry` ref; the synthetic chat path
  carries artifact refs only. Each test fails with the stamping line removed.
