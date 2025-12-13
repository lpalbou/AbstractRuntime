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

