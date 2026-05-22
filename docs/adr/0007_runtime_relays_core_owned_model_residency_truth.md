## ADR 0007: Runtime relays Core-owned model residency truth

### Status
Accepted (2026-05-21)

### Context
Hosts and operator UIs ask Runtime whether a model is loaded. That question can
mean different things:

- a configured default provider/model;
- a cached Runtime or gateway client;
- a provider-owned model actually resident in memory.

Only AbstractCore owns provider implementations and provider-specific loaded
instance knowledge. Runtime can create and cache AbstractCore clients, but that
does not prove provider residency. HTTP-backed local providers, provider
servers, and native in-process providers all have different truth sources.

### Decision
AbstractRuntime relays model-residency truth from AbstractCore. It does not
produce provider-residency truth itself.

Specifically:

- Runtime may expose its own cache/configuration state with fields such as
  `runtime_cached` and `cache_state`.
- Runtime may call public AbstractCore integration contracts such as
  `get_model_residency(...)` on a Core provider object or remote
  `/acore/models/*` responses.
- Runtime must not call provider-native HTTP APIs, provider-specific unload/list
  methods, or private provider internals to establish residency.
- Runtime must not infer loaded state from provider names, provider metadata,
  `local_provider`, `base_url`, ports, model catalogs, default configuration, or
  the existence of a cached client.
- If AbstractCore cannot verify provider residency, Runtime must fail closed:
  `loaded=false`, `resident=false`, `provider_residency_verified=false`, and an
  explicit unknown/provider-unverified state.

### Consequences
- Gateway, Flow, and other hosts can distinguish a cached Runtime client from a
  truly provider-loaded model.
- Positive `loaded=true` in Runtime local mode requires a Core-owned positive
  provider-residency response.
- Provider-specific loaded-instance work belongs in AbstractCore providers or
  capability plugins, not in Runtime.
- Some local rows remain visible as cache/configuration state while correctly
  reporting that provider residency is unknown or not loaded.

### Enforcement
- Reviews should reject Runtime provider-name lists, `base_url` heuristics, and
  direct provider-native residency probes.
- Runtime tests should cover verified loaded, verified not-loaded, and unknown
  provider-residency paths.
- Backlog items that need new residency truth must start with an AbstractCore
  contract or a Core backlog item.

### See Also
- Runtime backlog: [`../backlog/completed/0035_model_residency_provider_truth_for_local_http_clients.md`](../backlog/completed/0035_model_residency_provider_truth_for_local_http_clients.md)
- Runtime backlog: [`../backlog/proposed/0036_local_media_residency_bridge_to_core_residency.md`](../backlog/proposed/0036_local_media_residency_bridge_to_core_residency.md)
- Core companion ADR: [`../../../abstractcore/docs/adr/0008-provider-owned-model-residency-truth.md`](../../../abstractcore/docs/adr/0008-provider-owned-model-residency-truth.md)
