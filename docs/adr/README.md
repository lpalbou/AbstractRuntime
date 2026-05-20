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
| 0004 | [Runtime Owns Run-Scoped Media Execution Truth](0004_runtime_owns_run_scoped_media_execution_truth.md) | Accepted | 2026-05-20 | Hosts must route run-scoped media execution through Runtime |
| 0005 | [Runtime Owns AbstractCore Host Discovery Queries](0005_runtime_owns_abstractcore_host_discovery_queries.md) | Accepted | 2026-05-20 | Hosts should ask Runtime for Core discovery/catalog snapshots |
| 0006 | [Runtime Owns Durable AbstractCore Bloc Prompt-Cache Control](0006_runtime_owns_durable_abstractcore_bloc_prompt_cache.md) | Accepted | 2026-05-20 | Hosts should use Runtime for durable bloc/KV controls and binding-aware execution |

## Relationship to Backlog

ADRs explain *why*. Backlog items explain *what* and *how*.

| ADR | Related Implementation |
|-----|------------------------|
| 0001 | `backlog/completed/005_abstractcore_integration.md` |
| 0002 | `backlog/completed/005_abstractcore_integration.md` |
| 0003 | `backlog/completed/007_provenance_hash_chain.md`, `backlog/planned/008_signatures_and_keys.md` |
| 0004 | `backlog/completed/023_truthful_local_media_residency_boundaries.md`, `backlog/completed/024_runtime_owned_run_scoped_media_execution.md` |
| 0005 | `backlog/completed/026_runtime_host_discovery_facade_for_core_catalogs.md` |
| 0006 | `backlog/completed/027_runtime_durable_bloc_prompt_cache_facade.md` |

## Adding New ADRs

When making a significant architectural decision:
1. Create `docs/adr/NNNN_short_title.md`
2. Use the template: Status, Context, Decision, Consequences
3. Set status to "Accepted" once the decision is final
4. If superseding an old ADR, update the old one's status to "Superseded by NNNN"
