## Todo 3 — Provenance / AI fingerprint: tamper-evident ledger chain

### Goal
Evolve the current “AI fingerprint” concept into **tamper-evident provenance** for workflow runs:
- Each ledger record becomes part of a **hash chain**.
- Any mutation, deletion, or reordering of records becomes detectable.
- **Signatures are optional** (future extra), so v0.1 provides *tamper-evidence* not non-forgeability.

This directly supports accountability and auditability for stateful agents/workflows.

---

### Key concept distinction (must be explicit)

- **Tamper-evident**: you can detect changes *if you have a trusted copy of the chain head* (or any trusted checkpoint).
- **Tamper-proof / non-forgeable**: requires cryptographic signatures + key management.

We implement **tamper-evident now**, and document signatures as an optional extra.

---

### Where provenance data lives

Current kernel models already include:
- `RunState.actor_id`
- `StepRecord.actor_id`

We add integrity fields at the ledger level:
- `prev_hash: Optional[str]`
- `record_hash: Optional[str]`
- `signature: Optional[str]` (optional extra; leave `None` for v0.1)

Two implementation options:

#### Option A (recommended): extend `StepRecord` + use a `LedgerStore` decorator
- Add optional fields to `StepRecord`.
- Implement `HashChainedLedgerStore(LedgerStore)` that computes hashes on append.
- Underlying store stays unchanged (JSONL backend keeps working).

Pros:
- Minimal kernel impact (additive fields only)
- Works with any backend store
- Hash computation is centralized and consistent

#### Option B: store-level-only hashes (dict-only)
- Keep `StepRecord` unchanged.
- Store dicts with extra fields.

Cons:
- Harder to keep types consistent
- Easier to drift between store implementations

---

### Canonical hashing rules

Critical: all hashing must be **stable** across runs/machines.

Canonical JSON:
- `sort_keys=True`
- `separators=(",", ":")`
- `ensure_ascii=False`

Hash:
- SHA-256 hex digest

Do **not** hash unstable fields unless you mean to (timestamps are fine; they’re part of the record).

---

### Hash chain computation

Pseudo-code:

```python
import hashlib
import json
from dataclasses import asdict
from typing import Any, Dict, Optional


def canonical_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def compute_record_hash(*, record: Dict[str, Any], prev_hash: Optional[str]) -> str:
    # Ensure we don't recursively include our own hash fields.
    clean = dict(record)
    clean.pop("record_hash", None)
    clean.pop("signature", None)

    clean["prev_hash"] = prev_hash
    return sha256_hex(canonical_json(clean))
```

Important:
- `prev_hash` must be included in the hashed payload.
- `signature` is not included in the hash (it signs the hash).

---

### Ledger store decorator design

`HashChainedLedgerStore` wraps an underlying `LedgerStore`.

Responsibilities:
- Maintain per-run chain head in memory for efficient append.
- On append:
  - get current head hash (or `None` if first)
  - set `record.prev_hash = head`
  - set `record.record_hash = compute_record_hash(...)`
  - append to underlying store
  - update head

Minimal interface:

```python
class HashChainedLedgerStore(LedgerStore):
    def __init__(self, inner: LedgerStore):
        self._inner = inner
        self._head_by_run: dict[str, str | None] = {}

    def append(self, record: StepRecord) -> None:
        prev = self._head_by_run.get(record.run_id)
        d = asdict(record)
        d["prev_hash"] = prev
        d["record_hash"] = compute_record_hash(record=d, prev_hash=prev)

        # Mutate record if StepRecord has these fields.
        record.prev_hash = d["prev_hash"]
        record.record_hash = d["record_hash"]

        self._inner.append(record)
        self._head_by_run[record.run_id] = record.record_hash

    def list(self, run_id: str) -> list[dict[str, Any]]:
        return self._inner.list(run_id)
```

Design note:
- If you want the decorator to work even when appending after restart, you can initialize the head on first append by reading the last stored record (optional for v0.1).

---

### Verification utility

Implement:

```python
def verify_ledger_chain(records: list[dict[str, Any]]) -> dict[str, Any]:
    ...
```

Return a structured report (not just boolean):

```json
{
  "ok": true,
  "count": 42,
  "errors": [],
  "first_bad_index": null,
  "head_hash": "...",
  "computed_head_hash": "..."
}
```

Validation steps:
1. For record `i`:
   - expected `prev_hash` is `None` when `i == 0`, else `records[i-1].record_hash`.
   - recompute `computed_hash` from canonical form.
   - compare with stored `record_hash`.
2. If any mismatch, return `ok=false` and populate errors.

---

### Relationship to ActorFingerprint

`ActorFingerprint` currently provides stable `actor_id` from metadata.

Recommended provenance practice:
- At runtime start: create (or look up) an `ActorFingerprint` and set `RunState.actor_id`.
- Every `StepRecord` should copy `actor_id`.
- The hash chain includes `actor_id` values, so tampering with “who did what” is detectable.

---

### Optional signatures (future extra)

Add as `abstractruntime[crypto]`:
- Store `signature` on each record (signature of `record_hash`).
- Store a public key reference in actor metadata.
- Verify signatures in `verify_ledger_chain()` when keys are available.

This is intentionally postponed to avoid key-management complexity in v0.1.

---

### Deliverable checklist

- [ ] `StepRecord` includes optional `prev_hash`, `record_hash`, `signature`
- [ ] `HashChainedLedgerStore` implemented
- [ ] `verify_ledger_chain(records)` implemented with structured report
- [ ] Clear docs stating tamper-evident vs signed provenance


