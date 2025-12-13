# 002_persistence_and_ledger

> Legacy note: preserved for history. See `docs/backlog/completed/002_persistence_and_ledger.md` and `docs/backlog/planned/009_artifact_store.md`.

## Goal
Add durability primitives:
- RunStore (checkpoints)
- LedgerStore (journal d’exécution)
- ArtifactStore (optional, for large payload references)

## Why “ledger” (journal) matters
Without a ledger, you cannot reliably:
- debug what happened across days
- replay the run without re-running side effects
- audit tool/LLM calls

## Deliverables
- `abstractruntime/storage/run_store.py` (interface + in-memory + json-file impl)
- `abstractruntime/storage/ledger_store.py` (interface + in-memory + jsonl impl)
- `abstractruntime/storage/artifact_store.py` (interface + minimal file impl)

## Acceptance criteria
- After restarting the process, a paused run can be loaded and resumed.
- The ledger contains step records with timestamps + status + effect/result.

## Non-goals
- exact-once distributed semantics
- advanced indexing/search (can be added later)


