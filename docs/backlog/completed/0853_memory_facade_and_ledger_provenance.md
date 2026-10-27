# 0853 — Memory facade and ledger provenance: the host surface for AbstractMemory, the ledger row address, and a step ref on every memory write

**Status**: completed 2026-10-27 on branch `feat/memory-graph-facade`, UNRELEASED (ships in 0.14.0, which waits for AbstractMemory 0.4.0 on PyPI) · **Priority**: P1 · **Created**: 2026-10-27
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
- The workflow-native primitive (operator, 2026-10-27: "workflows are the framework's central native primitive; memory
  capture and reconsideration should be expressible as workflow steps/effects with ledger provenance"):
  `EffectType.MEMORY_REVISE` (host handler in `build_memory_kg_effect_handlers` over `MemoryGraph.apply_revision`)
  and the VisualFlow node `memory_revise`.
- Performance of the row address on the JSON-file backend (the gateway's default).
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

## Report (2026-10-27)

Commits on `feat/memory-graph-facade` (after `76b1319` WIP): `de96ede` memory facade, `7b0d031` step context
installed, `dd9e924` stamping, `2574881` MEMORY_REVISE + node, `09db6f5` row-address performance, `76e9988` docs and
version, `9eea551` diary reads in `GATEWAY_NAMES`; this report.

What landed:
- `integrations/abstractmemory/memory_facade.py`: `MemoryFacade` (`open_mind`, `graph`/`store`/`journal`, `stamp`,
  `stamp_current_step`, `apply_revision`, `history`/`sources_of`/`derived_from_source`, `reconstruct_source`),
  `is_real_run`, `GATEWAY_NAMES` (43 names: the gateway's direct imports, including the diary reads `open_questions` /
  `open_problems` / `open_ideas`, + the 8 graph names; every one exists in AbstractMemory, none left out), resolved by module `__getattr__` so the module never imports AbstractMemory at
  import time.
- `core/runtime.py` `_execute_effect_with_retry` installs `effect_step_context(StepContext(...))` around the handler,
  with the seq `append()` returned for the attempt's STARTED row.
- Stamping: `seam_handlers._handle_form`, `effect_handlers._handle_assert`, `identity_support.project_diary_entry`.
  A caller `payload_ref` is an artifact ref only when the artifact store holds it (a payload_ref may name an
  identity-core `.yaml` record). A private diary projection names its entry only (no origin by design).
- `MEMORY_REVISE` / `memory_revise`: contract in `docs/integrations/abstractmemory.md`. Caller `run`/`step` refs are
  refused (runtime-owned); unknown payload keys are refused; the default `revision_id` hashes (run, node, scope,
  owner, cause, ops) so a retried attempt is a no-op.

Performance (10,000 rows, ~10 MB JSON-file ledger, ~1 KB rows, Apple Silicon, warm cache; method:
fill one run's ledger with 5,000 STARTED/COMPLETED pairs through `append`, then the median of 20 calls each):

| | JSONL before | JSONL after | SQLite before | SQLite after |
|---|---|---|---|---|
| `get_record(seq)` | 58 ms | 0.05 ms | 0.02 ms | 0.01 ms |
| `find_started_record(step, before_seq=terminal)` (the slim-resolve path) | 58-72 ms (full parse) | 0.08 ms | 3.5 ms | 0.01 ms |
| `find_started_record(step)` unbounded, row in the middle | 72 ms | 12 ms | 11 ms | 3.4 ms |
| `find_started_record` worst case (first row) / miss | 72 ms | 24 ms / 7 ms | 11 ms | 10 ms |
| first read in a cold process (index build) | 58 ms | 48 ms | 0.3 ms | 0.3 ms |

JSONL: a per-run line-offset index (`_JsonlRowIndex`, 8 bytes per record, 128 runs LRU), written through by
`append`, extended by reading only the bytes another writer appended (keyed by inode + indexed size), rebuilt for a
recreated file; a torn trailing line falls back to the `list()` reading. `append` uses the index (not a full
`list()`) to count after a foreign write. SQLite: `find_started_record` reads newest-first with `LIMIT 1` (Python's
cursor prefetch made the unlimited query scan the rest of the run).

Tests: `test_memory_facade.py` (10), `test_step_context.py` (4), `test_memory_step_provenance.py` (7),
`test_memory_revise.py` (7), `test_jsonl_row_index.py` (6); each docstring lists the mutants checked by hand.
Full suite (`PYTHONPATH=src:<AbstractMemory graph worktree>/src`): `main` 3763 passed / 46 skipped / 0 failed; this
branch 3810 passed / 46 skipped / 0 failed (+47: the get_record, facade, step-context, provenance, MEMORY_REVISE and
row-index tests). `tests/test_packaging_extras.py` pins the new `AbstractMemory>=0.4.0` floor.

Seams and notes:
- `abstractmemory.graph.Provenance.from_dict` turned a STRING `derived_from` into a tuple of characters
  (`tuple("m1")`); reported, fixed on the AbstractMemory branch (`c2f035f`), not worked around here.
- Review N1 (AbstractNote review): a tool approved by a human runs inside `Runtime.resume`, outside
  `_execute_effect_with_retry`, so it saw no step context and its note carried no step ref. Fixed: the approval
  resume publishes the waiting TOOL_CALLS step (`_waiting_step_context`: the WAITING row's `step_id`, the seq of ITS
  STARTED row via `find_started_record`, `seq=None` when that row cannot be found, never guessed). Audit of the other
  paths: WAIT_EVENT / WAIT_UNTIL / ASK_USER / subworkflow resumes only write the payload into vars (no handler or
  tool runs; the next effect goes through `_execute_effect_with_retry`), terminal hooks are not effect handlers, the
  publish / schedule handlers are effect handlers, and the tool executor's worker threads copy the context
  (`contextvars.copy_context`). Test: `test_step_context.py::test_a_tool_run_by_an_approval_resume_sees_the_waiting_steps_started_row`
  (SQLite and JSONL); the reviewer's AbstractNote proof script now records `ledger_seq` / `ledger_step_id` resolving to
  the `note_capture` TOOL_CALLS STARTED row.
- `is_real_run` keeps its own `SYNTHETIC_RUN_PREFIXES`: AbstractMemory's `SYNTHETIC_RUN_ID_PREFIXES` was not on its
  branch when this landed (follow-up: reuse it, with a test that the two agree).
- `llms-full.txt` was not regenerated (coredoc pass at release).

