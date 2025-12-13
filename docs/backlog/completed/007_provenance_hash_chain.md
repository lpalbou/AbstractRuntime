## 007_provenance_hash_chain (completed for v0.1)

### Goal
Add tamper-evident provenance to the ledger.

### What shipped
- `src/abstractruntime/storage/ledger_chain.py`
  - `HashChainedLedgerStore`
  - `verify_ledger_chain(records)`
- `StepRecord` includes optional fields:
  - `prev_hash`, `record_hash`, `signature`

### Notes
- This provides tamper-evidence, not non-forgeability.
- Signatures are planned as an optional extra.

