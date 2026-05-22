## ADR 0005: Runtime owns AbstractCore host discovery queries

### Status
Accepted (2026-05-20)

### Context
Gateway and other hosts still need to answer operator and thin-client discovery questions such as:

- which providers are available;
- which models a provider exposes;
- what a model's capabilities are;
- which voice, TTS, STT, and vision catalogs are currently visible;
- which local vision models are cached.

Before this decision, those reads were often handled outside Runtime by:

- raw HTTP proxying from Gateway to AbstractCore server routes;
- direct imports of public `abstractcore.*` registries and helpers from Gateway;
- in at least one case, direct import of a Core-server route implementation.

That creates an inconsistent boundary: execution and some controls go through Runtime, but discovery reads bypass it entirely.

These discovery reads are not durable run truth in the same sense as `LLM_CALL`, `TOOL_CALLS`, or run-scoped media child runs. They are current-environment snapshots that may change between the original query and a later replay.

### Decision
AbstractRuntime owns the host-facing AbstractCore discovery/query boundary.

Specifically:

- Hosts should ask Runtime for AbstractCore discovery and catalog queries rather than importing Core registries or proxying Core server routes directly.
- Runtime provides a public discovery facade for host-oriented snapshot queries.
- These discovery methods are query/snapshot oriented, not durable run effects.
- Replay or audit consumers should treat discovery results as recorded snapshots when they are persisted by hosts, not as work that should be re-executed automatically.
- Gateway-local liveness, auth, install diagnostics, and non-Core helper projections may remain host-local.

### Consequences
- Runtime becomes the single Python integration boundary for AbstractCore execution, control, and discovery surfaces.
- Gateway can shrink into an HTTP projection layer rather than a second Core integration layer.
- Discovery results remain semantically distinct from durable run execution truth.
- Some local discovery implementations may still require Runtime integration code to adapt Core capability helpers or server-backed routes until Core exposes cleaner public helpers.

### Enforcement
- New Gateway or host adoption work should prefer Runtime facades over direct `abstractcore.*` imports for discovery/catalog routes.
- Runtime docs should distinguish durable run work, operator controls, and snapshot discovery queries.
- Direct Gateway reach-through to `runtime._abstractcore_llm_client` remains prohibited when a public facade exists.

### Validation
- Runtime tests should cover local and remote discovery facade behavior.
- Documentation should show the public discovery facade entrypoint and its scope.
- Follow-up backlog work in hosts should remove direct Core discovery bypasses where the facade now covers them.

### See Also
- Implementation: [`../backlog/completed/026_runtime_host_discovery_facade_for_core_catalogs.md`](../backlog/completed/026_runtime_host_discovery_facade_for_core_catalogs.md)
- Existing operator controls: [`0004_runtime_owns_run_scoped_media_execution_truth.md`](0004_runtime_owns_run_scoped_media_execution_truth.md)
- Integration guide: [`../integrations/abstractcore.md`](../integrations/abstractcore.md)
