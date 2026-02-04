# Provenance (tamper-evident ledger)

AbstractRuntime’s ledger is an append-only journal of `StepRecord` entries. For audit/debug workflows, you can add **tamper-evidence** via a hash chain:
- each record carries `prev_hash` + `record_hash`
- modifications/reordering become detectable when you verify the chain

Implementation pointers:
- model fields: `src/abstractruntime/core/models.py` (`StepRecord.prev_hash`, `StepRecord.record_hash`, `StepRecord.signature`)
- hash-chain decorator + verifier: `src/abstractruntime/storage/ledger_chain.py`

## What is implemented (v0.4.0)

- `HashChainedLedgerStore(inner_store)` — wraps any `LedgerStore` to compute hashes on append
- `verify_ledger_chain(records)` — validates the chain and returns a verification report

Example:

```python
from abstractruntime import Runtime, WorkflowSpec
from abstractruntime.storage import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.ledger_chain import HashChainedLedgerStore, verify_ledger_chain

ledger = HashChainedLedgerStore(InMemoryLedgerStore())
rt = Runtime(run_store=InMemoryRunStore(), ledger_store=ledger)

# ... run workflows ...

records = rt.get_ledger(run_id="...")  # list[dict]
report = verify_ledger_chain(records)
print(report.get("ok"), report.get("errors"))
```

## What is intentionally not implemented (yet)

- cryptographic signatures (non-forgeability)
- key management / delegation / revocation

Those belong in an optional extra (e.g., `abstractruntime[crypto]`) once the design is finalized.

## See also

- `architecture.md` — ledger as the source of truth
- `evidence` capture: `src/abstractruntime/evidence/recorder.py` (stores external-boundary evidence as artifacts + index)

