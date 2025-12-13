## Provenance (tamper-evident ledger)

AbstractRuntime’s ledger is an append-only journal of step records.

For accountability and auditability, we add **tamper-evidence**:
- each record includes `prev_hash` + `record_hash`
- modifications/reordering become detectable

This is **tamper-evident**, not tamper-proof.

---

### What is implemented (v0.1)

- `HashChainedLedgerStore(inner)` decorator
- `verify_ledger_chain(records)` verification report

Implementation:
- `src/abstractruntime/storage/ledger_chain.py`

---

### What is not implemented yet

- cryptographic signatures (non-forgeability)
- key management / delegation

Those belong in an optional extra (e.g. `abstractruntime[crypto]`).

---

### Relationship to AI fingerprint

- `RunState.actor_id` and `StepRecord.actor_id` support attribution.
- a future signed mode can bind `actor_id` to a public key.

