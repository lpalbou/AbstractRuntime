## ADR 0006: Runtime owns durable AbstractCore bloc prompt-cache control

### Status
Accepted (2026-05-20)

### Context
AbstractCore `2.13.22` added a public durable prompt-cache contract for local
backends:

- text/file blocs as durable source-of-truth content,
- provider/model-specific KV artifacts,
- request-time `prompt_cache_binding` proof for exact reuse.

Before this decision, Runtime only owned:

- best-effort session prompt-cache controls,
- model-residency controls,
- discovery/query snapshots,
- durable run-scoped media execution.

That left a real gap for app-facing durable prompt caching:

- hosts had no public Runtime path for bloc/KV control operations;
- remote Runtime execution dropped `prompt_cache_binding`;
- `LLM_CALL` key injection could derive a competing session key even when a
  durable binding was present;
- local Runtime had no explicit bloc-store root policy and would otherwise
  inherit Core defaults implicitly.

### Decision
AbstractRuntime owns the host-facing durable AbstractCore bloc prompt-cache
boundary.

Specifically:

- Hosts should use the existing `AbstractCoreHostFacade` for durable bloc/KV
  prompt-cache controls rather than adding a fourth public facade.
- The public host surface includes:
  - `upsert_text_bloc(...)`
  - `get_bloc_record(...)`
  - `list_blocs(...)`
  - `get_bloc_kv_manifest(...)`
  - `ensure_bloc_kv_artifact(...)`
  - `load_bloc_kv_artifact(...)`
  - `list_bloc_kv_artifacts(...)`
  - `delete_bloc_kv_artifact(...)`
  - `prune_bloc_kv_artifacts(...)`
  - `delete_bloc(...)`
- Runtime mirrors Core's public bloc request/response model instead of
  inventing a Gateway-local save/load vocabulary.
- `LLM_CALL.payload.params.prompt_cache_binding` is the public Runtime binding
  input. Runtime may also accept the local Python alias
  `expected_prompt_cache_binding`, but it must normalize to
  `prompt_cache_binding`.
- When a binding is present:
  - `binding.key` becomes the effective `prompt_cache_key` when no explicit key
    was supplied;
  - mismatched `prompt_cache_key` and `binding.key` fail before provider or
    server execution;
  - Runtime must not auto-derive a competing session key;
  - local session prompt-cache preparation must not mutate the bound key before
    generation.
- Local Runtime owns an explicit bloc-root policy:
  - default local root: `~/.abstractruntime/blocs`
  - default file-runtime root: `<base_dir>/blocs`
  - explicit `bloc_root_dir=` overrides are allowed for hosts such as Gateway

### Consequences
- Runtime becomes the public Python boundary for three distinct prompt-cache
  tracks:
  - session prompt-cache reuse,
  - durable bloc/KV/binding reuse,
  - local-admin snapshot save/load/list work if promoted later
- Hosts can expose durable bloc caching without reaching into Core or provider
  internals.
- Hosts can also reclaim durable bloc/KV storage through the same Runtime
  boundary instead of manual filesystem deletion or private Core hooks.
- Runtime does not silently inherit Core's default `~/.abstractcore/blocs`
  layout.
- Provider-private prompt-cache snapshot save/load remains a separate local-admin
  concern and is not promoted by this decision.

### Enforcement
- New host adoption work should use `get_abstractcore_host_facade(runtime)` for
  durable bloc/KV operations.
- Runtime docs and AI-readable docs must describe the durable bloc track
  separately from session prompt caching.
- Runtime code review should reject any new auto-derived prompt-cache behavior
  that competes with `prompt_cache_binding`.

### Validation
- Host-facade tests cover delegation and remote route shaping for bloc/KV
  operations.
- `LLM_CALL` tests cover binding-only adoption, alias normalization, and
  mismatch fail-fast behavior.
- Regression coverage includes local/root-policy tests and the broader
  AbstractCore Runtime facade suite.

### See Also
- Implementation: [`../backlog/completed/027_runtime_durable_bloc_prompt_cache_facade.md`](../backlog/completed/027_runtime_durable_bloc_prompt_cache_facade.md)
- Core binding contract: `../abstractcore/docs/adr/0007-durable-memory-bloc-cache-binding.md`
- Related Runtime ADRs:
  - [`0004_runtime_owns_run_scoped_media_execution_truth.md`](0004_runtime_owns_run_scoped_media_execution_truth.md)
  - [`0005_runtime_owns_abstractcore_host_discovery_queries.md`](0005_runtime_owns_abstractcore_host_discovery_queries.md)
