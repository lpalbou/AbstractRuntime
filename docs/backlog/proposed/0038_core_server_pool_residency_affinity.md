# Proposed: Core Server Pool Residency Affinity

## Metadata
- Created: 2026-05-22
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs:
  - `docs/adr/0007_runtime_relays_core_owned_model_residency_truth.md`
- ADR impact: Needs new ADR if promoted. Pool routing and residency affinity would become durable cross-package
  behavior and should not live only in backlog prose.

## Context

Runtime currently treats AbstractCore residency as live, ephemeral state owned by the configured Core backend. That is
enough while a Runtime instance talks to one Core server or one local Core facade at a time. A future deployment may run
a pool of AbstractCore servers where model residency is not globally true: a model can be loaded on one Core server and
not on another.

This proposal preserves that future concern without blocking the current `0036` work. Today, Runtime should simply ask
the configured Core backend which models are loaded now and relay that answer.

## Current code reality

- `RemoteAbstractCoreLLMClient` stores one `server_base_url` and relays residency control to
  `/acore/models/loaded`, `/acore/models/load`, and `/acore/models/unload`.
- `RemoteAbstractCoreLLMClient` also sends media execution requests to the same configured Core server root for
  `/v1/images/*`, `/audio/speech`, `/audio/transcriptions`, and `/audio/music`.
- Runtime has no `core_uuid -> server` registry, Core-server pool abstraction, pool-wide warmup contract, or routing
  affinity token.
- Loaded model state is ephemeral. After an AbstractCore restart, Runtime cannot recover a previous in-memory residency
  state; it must ask Core again.

## Problem or opportunity

If Runtime later supports multiple AbstractCore servers behind one Runtime or Gateway surface, a plain statement like
`model X is loaded` becomes ambiguous. It may mean loaded on one server, loaded on every server in a pool, or loaded on
the server that will handle the next generation request. Those are different contracts.

## Proposed direction

When Runtime grows Core-pool support, make residency target identity explicit:

- Core servers should expose a stable runtime/server identity for the current process lifetime, such as `core_uuid`.
- Residency records should identify the Core backend that supplied the answer.
- Runtime should either route generation to the Core backend that has the resident model, or advertise only pool-level
  residency that a Core-side or pool-side controller guarantees.
- Gateway and higher-level apps should not collapse per-Core loaded state into a global loaded flag unless the pool
  contract explicitly says it is global.

## Why it might matter

Pooling can improve capacity and isolation, but it makes loaded-state truth less obvious. Preserving this as a proposed
item keeps the future design from contaminating the simpler current rule: ask the configured Core backend for live
truth.

## Promotion criteria

Promote this item only when at least one of these becomes true:

- Runtime or Gateway is asked to route requests across multiple AbstractCore servers.
- AbstractCore exposes a server identity or pool membership contract that Runtime should relay.
- Users need warmup/load controls that target a specific Core server or an entire Core pool.
- A real bug shows a model loaded on one Core backend while generation is sent to another.

## Validation ideas

- Contract tests for per-Core residency records including server identity.
- Routing tests proving generation is sent to the Core server that reported the resident model.
- Pool-level tests proving `loaded=true` means either per-server truth or explicitly guaranteed pool-wide truth.
- Restart tests proving Runtime does not treat an old Core identity's loaded state as current after a Core process
  restarts.

## Non-goals

- Do not implement Core-pool routing now.
- Do not block single-Core residency behavior on server UUID support.
- Do not make Runtime query providers directly for loaded-state truth.
- Do not make loaded-state durable; residency remains live and ephemeral.

## Guidance for future agents

Re-check the current Core and Runtime server contracts before promoting this. If the work creates durable rules for
server identity, routing affinity, or pool-wide warmup semantics, write or update an ADR first.
