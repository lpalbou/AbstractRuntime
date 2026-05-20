# Proposed: Runtime local-admin prompt-cache snapshot control

## Metadata
- Created: 2026-05-20
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: `../abstractcore/docs/adr/0007-durable-memory-bloc-cache-binding.md`
- ADR impact: None; this item is intentionally secondary to Core's accepted durable bloc contract.

## Context

Runtime now exposes public host-facing surfaces for:

- prompt-cache control and model residency through `AbstractCoreHostFacade`
- run-scoped durable media execution through `AbstractCoreRunFacade`
- discovery/catalog snapshot queries through `AbstractCoreDiscoveryFacade`

Core `2.13.23` also makes the durable app-facing prompt-cache path clear:

- blocs + KV artifacts + `prompt_cache_binding`

That means live prompt-cache snapshot save/load should no longer be treated as
the main app contract. It remains useful only as a separate local operator/admin
surface.

Today Gateway still implements that surface as a private hack.

## Current code reality

Inspected files and behavior:

- `src/abstractruntime/integrations/abstractcore/host_facade.py`
  - no snapshot admin methods yet
- `../abstractgateway/src/abstractgateway/routes/gateway.py`
  - `/prompt_cache/saved|save|load` still reaches through private Runtime state
    and provider instances directly
  - current implementation hard-codes only MLX and HuggingFace GGUF-style local
    behavior
- `../abstractcore/abstractcore/providers/base.py`
  - exposes public provider methods `prompt_cache_save(...)` and
    `prompt_cache_load(...)`
- `../abstractcore/abstractcore/providers/mlx_provider.py`
  - implements public prompt-cache save/load
- `../abstractcore/abstractcore/providers/huggingface_provider.py`
  - implements public prompt-cache save/load across the provider's supported
    local cache modes
- `docs/backlog/completed/027_runtime_durable_bloc_prompt_cache_facade.md`
  - records the primary app-facing durable bloc track that already shipped

What exists now:

- Runtime owns normal prompt-cache control operations cleanly.
- Gateway owns a working but boundary-breaking local snapshot implementation.
- Core exposes the public provider hooks Runtime would need.

What is still wrong:

- no Runtime-owned public snapshot-admin contract
- Gateway still reads provider-private state and hard-codes two backend paths
- the current item is too narrow because listing/cataloging saved snapshots is
  part of the real operator workflow too

## Problem or opportunity

There is still one operator/admin prompt-cache feature with no clean Runtime
boundary:

- list locally saved prompt-cache snapshots
- save a live local provider cache snapshot
- load a saved local provider cache snapshot

We may still need that feature, but not as a Gateway-local hack and not as the
normal app-facing prompt-cache story.

## Proposed direction

Add a Runtime-owned **local-admin snapshot control** surface as a secondary
prompt-cache track.

### Preferred Runtime boundary

Prefer extending the existing `AbstractCoreHostFacade` with explicit
admin-oriented snapshot methods rather than creating yet another public facade.

Likely methods:

- `list_prompt_cache_snapshots(...)`
- `prompt_cache_save_snapshot(...)`
- `prompt_cache_load_snapshot(...)`

The naming should make the boundary obvious:

- local admin / snapshot control
- not the normal durable bloc app contract

### Capability and backend rules

Do not hard-code provider names in Runtime or Gateway.

Instead, base support on Core's public provider capabilities and methods.
Runtime should stay honest that snapshot support is:

- local only
- model/backend-specific
- available only when Core reports save/load support

The honest target matrix is:

- MLX
- HuggingFace transformers where cache serialization/reload is supported
- HuggingFace GGUF where cache serialization/reload is supported

That is better described as **three local backend families**, not three generic
provider names.

### Path and naming policy

The public surface should not expose arbitrary filesystem paths as the primary
client contract.

Before promotion, define:

- Runtime-owned snapshot root policy
- logical snapshot naming rules
- whether hosts may pass an explicit root/directory
- whether the public contract returns raw paths, opaque names, or both

Gateway's current `saved` listing behavior implies that list/catalog metadata is
part of the required feature, not a separate afterthought.

### Relationship to durable blocs

Keep this separate from the durable bloc contract.

Intended split:

- durable blocs + bindings: app-facing exact reuse across restarts
- snapshot admin: operator/local-admin tooling around live provider cache state

## Why it might matter

If local operator workflows still need snapshot save/load after durable blocs
ship, Runtime should own that boundary too:

- Gateway can stop reaching through private Runtime and provider internals
- support can expand beyond today's MLX + HF GGUF limitation
- clients can distinguish app-facing durable reuse from local-admin snapshot
  tooling

## Promotion criteria

- Completed item `027_runtime_durable_bloc_prompt_cache_facade.md` remains the primary app-facing prompt-cache path.
- Runtime maintainers confirm that a secondary local-admin snapshot surface is
  still needed after that.
- Runtime defines a snapshot root/path policy that does not leak raw filesystem
  contracts casually.
- Gateway confirms that `list/save/load` through Runtime is enough to replace
  the current private-provider implementation.

## Validation ideas

- Host-facade tests covering structured unsupported responses and happy paths
  for:
  - listing snapshots
  - saving snapshots
  - loading snapshots
- Backend coverage across:
  - MLX
  - HuggingFace transformers when Core capabilities allow save/load
  - HuggingFace GGUF when Core capabilities allow save/load
- Negative tests proving unsupported local backends fail honestly instead of
  pretending to support snapshots.
- Gateway adoption tests proving `/prompt_cache/saved|save|load` delegate
  through Runtime only and no longer hard-code only two backend families.

## Non-goals

- Do not treat snapshot save/load as the normal app-facing durable prompt-cache
  contract.
- Do not expose provider-private cache internals.
- Do not promise remote/server snapshot persistence without a real public Core
  contract.
- Do not turn snapshot save/load into a durable workflow effect by default.

## Guidance for future agents

Implement this only after the durable bloc track is shipped and adopted where needed.

This is secondary local-admin work. Keep it honest, capability-driven, and
explicitly separate from the main app contract of blocs + bindings.
