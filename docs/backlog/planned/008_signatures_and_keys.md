## 008_signatures_and_keys (planned)

### Goal
Upgrade provenance from tamper-evident to cryptographically signed:
- bind `actor_id` to a public key
- sign `record_hash` (Ed25519)
- verify signatures in verification reports

### Constraints
- keep as an optional extra dependency (`abstractruntime[crypto]`)
- define key storage approach (file, OS keychain, Vault, etc.)

### Non-goals (v0.1)
- global enforcement or discovery

### Related ADRs
- [ADR 0003: Provenance Hash Chain](../../adr/0003_provenance_tamper_evident_hash_chain.md) — Why signatures are deferred

### Prerequisites
- [007_provenance_hash_chain.md](../completed/007_provenance_hash_chain.md) — Hash chain implementation (completed)

