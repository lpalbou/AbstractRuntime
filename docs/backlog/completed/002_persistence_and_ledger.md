## 002_persistence_and_ledger (completed for v0.1)

### Goal
Add durability primitives:
- RunStore (checkpoints)
- LedgerStore (journal d’exécution)

### What shipped
- Interfaces: `src/abstractruntime/storage/base.py`
- In-memory stores: `src/abstractruntime/storage/in_memory.py`
- File stores:
  - `JsonFileRunStore` (JSON per run)
  - `JsonlLedgerStore` (append-only JSONL)
  - `src/abstractruntime/storage/json_files.py`

### Acceptance criteria (met)
- After restarting the process, a paused run can be loaded and resumed.
- The ledger contains step records with timestamps + status + effect/result.

### Deferred / planned
- ArtifactStore (large payloads by reference)
- Advanced indexing/search for ledgers

