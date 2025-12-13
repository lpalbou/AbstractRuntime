# 006_ai_fingerprint_and_provenance

> Legacy note: preserved for history. See `docs/backlog/completed/007_provenance_hash_chain.md` and `docs/backlog/planned/008_signatures_and_keys.md`.

## Goal
Introduce an **AI Fingerprint** concept that supports:
- stable identity over time
- accountability across actions
- tamper-evident run histories

This is not “tracking AI across the web” by itself; it is a **local provenance primitive** that can be adopted by systems.

## Key idea
Treat every meaningful action as a ledger entry (journal d’exécution) and make the ledger:

1) **attributable** (who did it)
2) **tamper-evident** (you can detect edits)
3) **portable** (can be exported and verified elsewhere)

## Proposed model

### Actor identity
- `ActorFingerprint` becomes a stable `actor_id` derived from a public key (Ed25519).
- Store `public_key` + optional human/org metadata.

### Signed, hash-chained ledger
Each `StepRecord` gets:
- `prev_hash` (hash of previous record)
- `record_hash` (hash of current record)
- `signature` (sign(record_hash) with actor private key)

This yields:
- a chain that detects record removal/reorder/modification
- optional multi-actor signatures (agent + hosting service)

### Two important modes
- **Development mode**: unsigned hashing only (cheap)
- **Accountability mode**: signing required (cryptographic)

## Minimal API
- `create_actor_keypair()`
- `get_actor_id(public_key) -> str`
- `sign_step_record(record, private_key) -> signed_record`
- `verify_ledger(records) -> VerificationReport`

## Dependencies
Use optional extra:
- `cryptography` (preferred) or `pynacl`

Avoid hard dependency in the kernel.

## Open questions
- Do we want DID compatibility (DID:key) for interoperability?
- Where do keys live (OS keychain, Vault, file)?
- How to represent “delegation” (agent acting on behalf of a user)?

## Non-goals
- global enforcement
- automatic web-wide discovery

## Why this belongs in AbstractRuntime
The runtime is where:
- actions become durable records
- “who did what, when, why” is recorded systematically

AbstractFlow can surface and export provenance, but the primitive must be near the ledger.


