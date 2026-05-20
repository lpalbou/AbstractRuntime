## 008_signatures_and_keys (planned)

**Status**: Planned
**Priority**: Low
**Depends on**: 007_provenance_hash_chain (completed)
**Related ADRs**: 0003_provenance_tamper_evident_hash_chain

---

## Goal

Extend the current tamper-evident provenance model with optional cryptographic signatures so ledger verification can answer not only "was this chain modified?" but also "which key signed these records?"

---

## Current code reality

- `StepRecord` already has a `signature` field.
- Hash-chained ledgers and verification already ship today.
- Provenance docs explicitly say signatures and key management are deferred.
- Runtime currently relies on actor fingerprints and actor metadata, not public-key-backed non-forgeability.

---

## Why this is still open

The current hash chain is useful for tamper evidence inside a ledger, but it does not provide:

- non-forgeable authorship
- signer delegation or rotation
- trust decisions based on known public keys

This matters only once consumers need stronger audit guarantees than "the stored chain is internally consistent."

---

## Planned scope

1. Define an optional signing interface around `record_hash` signing and verification.
2. Bind `actor_id` / actor metadata to signer metadata in a documented way.
3. Extend verification reports to surface signature validity and signer identity.
4. Keep crypto support as an optional extra dependency such as `abstractruntime[crypto]`.
5. Document acceptable key-loading strategies without making Runtime own a global PKI.

---

## Acceptance criteria

- [ ] A documented signing model exists for ledger records.
- [ ] Signature generation and verification can be enabled without changing the unsigned default path.
- [ ] Verification reports distinguish hash-chain integrity from signature validity.
- [ ] The design documents how signer metadata is attached to records.
- [ ] The feature remains optional and does not force crypto dependencies on all users.

---

## Validation

1. Unit tests for signed append + verification.
2. Negative tests for tampered payloads, wrong keys, and missing signatures.
3. Docs review for key-loading and rotation guidance.

---

## Non-goals

- mandatory signing for all runtimes
- centralized key discovery or global trust distribution
- UI trust policy or certificate management

---

## Priority note

Keep this low priority unless non-forgeable provenance becomes a concrete product or compliance requirement.
