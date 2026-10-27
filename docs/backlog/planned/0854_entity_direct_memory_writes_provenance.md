# 0854 — Entity memory written outside any effect carries no ledger source: world models, tending, the night passes

**Status**: planned · **Priority**: P2 · **Created**: 2026-10-27
**Cross-package**: AbstractMemory ADR 0003 (every memory is sourced or derived; `MemoryGraph.apply_revision` refuses an
ungrounded write) and its convergence track (moving writers onto revisions). ADR:
[0009](../../adr/0009_memory_facade_is_the_complete_host_surface_for_abstractmemory.md) (Consequences, Negative).
Follows [0853](../completed/0853_memory_facade_and_ledger_provenance.md).

## Why
0853 stamps a ledger source on every memory write the runtime performs INSIDE an effect (`MEMORY_FORM`,
`MEMORY_KG_ASSERT`, the `DIARY_WRITE` projection, `MEMORY_REVISE`). Entity homes also write derived memory by calling
AbstractMemory engines directly, outside any runtime step, so no step context exists and nothing names a source:

- `identity/chat.py` `world_model_update` (two sites) and `author_world_model`;
- `apply_tend_elections` (`identity/chat.py`, `integrations/abstractmemory/brain_handlers.py` `MEMORY_TEND`);
- the night passes `sleep_pass` / `dream_pass` (`identity/life.py` `_consolidate`, `brain_handlers.py`
  `MEMORY_CONSOLIDATE`).

These memories are DERIVED: under ADR 0003 they must name the memories they were derived from (`derived_from`) or a
ledger source. Today they carry `actor` / `channel` / `author` only, so a reader cannot trace them to any verbatim,
and once the writers move onto `MemoryGraph.apply_revision` they would be refused (`ProvenanceRequired`).

## Scope
In:
- Run the derived writes that have an enclosing effect (`MEMORY_TEND`, `MEMORY_CONSOLIDATE`) with that effect's step
  context and stamp its `step` ref through `MemoryFacade.stamp_current_step`.
- For the direct calls on the home-direct chat path and the life loop: either move them behind effects (a real run per
  night pass / tend election, so a step ref exists), or pass `derived_from` = the memory ids the engine read, through
  an AbstractMemory API that accepts it. Never a synthetic `chat-<sid>` run ref.
- Tests on a real home: a world-model update / tend election / night pass record names its sources or the memories it
  was derived from, and `MemoryFacade.sources_of(..., transitive=True)` reaches a ledger source.

Out:
- Real runs behind the home-direct entity driver as a whole (its own item); the gateway's routes.

## Validation
- Each derived writer above leaves provenance that `MemoryFacade.sources_of` resolves to at least one ledger source
  (directly or through `derived_from`); a test fails when the stamping line is removed.
