## ADR 0003: Provenance as tamper-evident hash-chained ledger (signatures optional)

### Status
Accepted (2025-12-11)

### Context
We need accountability for long-running agentic workflows:
- durable audit trail of steps
- ability to detect ledger edits/reordering

Full cryptographic signing requires key management and an additional dependency.

### Decision
- Implement **tamper-evidence** first:
  - `prev_hash` + `record_hash` per `StepRecord`
  - `HashChainedLedgerStore` decorator computes hashes on append
  - `verify_ledger_chain(records)` validates the chain
- Keep signatures as a planned optional extra (`abstractruntime[crypto]`).

### Consequences
- We can detect tampering when we have a trusted chain head/checkpoint.
- We avoid premature key-management complexity in v0.1.
- The design is forward-compatible with signed records.

### See Also
- Implementation: [`backlog/completed/007_provenance_hash_chain.md`](../backlog/completed/007_provenance_hash_chain.md)
- Future work: [`backlog/planned/008_signatures_and_keys.md`](../backlog/planned/008_signatures_and_keys.md)
- Documentation: [`provenance.md`](../provenance.md)
- Code: `src/abstractruntime/storage/ledger_chain.py`

