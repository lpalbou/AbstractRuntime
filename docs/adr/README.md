# Architectural Decision Records (ADRs)

ADRs document significant architectural decisions made during AbstractRuntime development. They explain *why* certain approaches were chosen, not *what* was built (that's in the backlog).

## Why ADRs Matter

When you ask "why is it designed this way?", the answer is in an ADR. ADRs are:
- **Immutable**: Once accepted, they are not edited (only superseded by new ADRs)
- **Historical**: They capture the context and constraints at decision time
- **Educational**: They help new contributors understand the architecture

## Index

| ID | Title | Status | Date | Summary |
|----|-------|--------|------|---------|
| 0001 | [Layered Coupling with AbstractCore](0001_layered_coupling_with_abstractcore.md) | Accepted | 2025-12-11 | Kernel stays dependency-light; AbstractCore integration is opt-in |
| 0002 | [Execution Modes](0002_execution_modes_local_remote_hybrid.md) | Accepted | 2025-12-11 | Support local, remote, and hybrid execution topologies |
| 0003 | [Provenance Hash Chain](0003_provenance_tamper_evident_hash_chain.md) | Accepted | 2025-12-11 | Tamper-evident ledger first; cryptographic signatures deferred |

## Relationship to Backlog

ADRs explain *why*. Backlog items explain *what* and *how*.

| ADR | Related Implementation |
|-----|------------------------|
| 0001 | `backlog/completed/005_abstractcore_integration.md` |
| 0002 | `backlog/completed/005_abstractcore_integration.md` |
| 0003 | `backlog/completed/007_provenance_hash_chain.md`, `backlog/planned/008_signatures_and_keys.md` |

## Adding New ADRs

When making a significant architectural decision:
1. Create `docs/adr/NNNN_short_title.md`
2. Use the template: Status, Context, Decision, Consequences
3. Set status to "Accepted" once the decision is final
4. If superseding an old ADR, update the old one's status to "Superseded by NNNN"

