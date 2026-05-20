# 027 Runtime Durable Bloc Prompt-Cache Facade

## Metadata
- Created: 2026-05-20
- Status: Completed
- Completed: 2026-05-20
- Origin: promoted from `proposed/2026-05-20_runtime_durable_bloc_prompt_cache_facade.md`

## Goal

Give hosts a public Runtime-owned path for durable AbstractCore bloc/KV
prompt-cache operations and make normal Runtime `LLM_CALL` execution binding
aware, so apps can use exact prompt-cache reuse across restarts without
reaching into Core or provider internals.

## What Shipped

- Extended `AbstractCoreHostFacade` with durable bloc/KV methods:
  - `upsert_text_bloc(...)`
  - `get_bloc_record(...)`
  - `get_bloc_kv_manifest(...)`
  - `ensure_bloc_kv_artifact(...)`
  - `load_bloc_kv_artifact(...)`
- Wired those methods through local, multi-local, remote, and hybrid
  AbstractCore Runtime clients.
- Forwarded `prompt_cache_binding` through remote Runtime execution.
- Made Runtime `LLM_CALL` key injection binding-aware:
  - normalize `expected_prompt_cache_binding` to `prompt_cache_binding`
  - adopt `binding.key` when no explicit key is supplied
  - fail fast on mismatched explicit key vs binding key
  - skip derived session-key injection when a binding is present
- Added an explicit local Runtime bloc-root policy:
  - default local root: `~/.abstractruntime/blocs`
  - default file-runtime root: `<base_dir>/blocs`
  - optional `bloc_root_dir=...` override
- Raised the AbstractCore dependency floor to `2.13.22`.

## Current Code Pointers

- `src/abstractruntime/integrations/abstractcore/host_facade.py`
- `src/abstractruntime/integrations/abstractcore/llm_client.py`
- `src/abstractruntime/integrations/abstractcore/effect_handlers.py`
- `src/abstractruntime/integrations/abstractcore/factory.py`
- `tests/test_abstractcore_host_facade.py`
- `tests/test_prompt_cache_modules.py`
- `tests/test_remote_llm_client.py`
- `tests/test_packaging_extras.py`
- `docs/adr/0006_runtime_owns_durable_abstractcore_bloc_prompt_cache.md`

## Validation

Focused validation run on 2026-05-20:

- `pytest -q tests/test_abstractcore_host_facade.py tests/test_remote_llm_client.py tests/test_prompt_cache_modules.py tests/test_packaging_extras.py`
- Result: `40 passed in 0.16s`

Broader regression slice:

- `pytest -q tests/test_abstractcore_discovery_facade.py tests/test_abstractcore_host_facade.py tests/test_abstractcore_run_facade.py tests/test_model_residency_control_plane.py tests/test_multimodal_abstractcore_integration.py tests/test_remote_llm_client.py tests/test_prompt_cache_modules.py tests/test_packaging_extras.py tests/test_runtime_install_boundary.py`
- Result: `108 passed in 0.37s`

Additional validation:

- `python -m py_compile src/abstractruntime/integrations/abstractcore/llm_client.py src/abstractruntime/integrations/abstractcore/host_facade.py src/abstractruntime/integrations/abstractcore/factory.py src/abstractruntime/integrations/abstractcore/effect_handlers.py tests/test_abstractcore_host_facade.py tests/test_prompt_cache_modules.py tests/test_remote_llm_client.py tests/test_packaging_extras.py`
- Result: clean

Practical proof points from the shipped tests:

- Remote Runtime now forwards `prompt_cache_binding` in chat-generation
  requests and preserves the exact binding payload.
- Remote Runtime omits local loaded-runtime selectors when bloc/KV calls are
  explicitly proxied with `base_url=...`.
- Local Runtime bloc helpers use the Runtime-owned bloc root and return
  structured artifact metadata including `prompt_cache_binding`.
- Binding-aware execution accepts binding-only requests, preserves matching
  explicit keys, rejects mismatches before provider execution, and does not
  derive a competing session key for keyless bindings.

## Completion report

### Date

2026-05-20

### Summary

Runtime now owns the app-facing durable bloc prompt-cache boundary for
AbstractCore-backed hosts. Hosts can create or read durable blocs, ensure or
load KV artifacts, and then execute ordinary `LLM_CALL`s with a
`prompt_cache_binding` through Runtime rather than relying on provider-private
snapshot hacks.

### Behavior changes

- `get_abstractcore_host_facade(runtime)` is now the public Runtime surface for
  durable bloc/KV prompt-cache operations in addition to normal prompt-cache
  control and model residency.
- Remote Runtime no longer drops `prompt_cache_binding`.
- Local Runtime no longer injects or mutates a competing session cache key when
  a durable binding is present.
- File-backed local runtimes now default bloc storage under `<base_dir>/blocs`
  instead of implicitly inheriting Core defaults.

### Review and refinement cycles

- Cycle 1: design review pushed three improvements before closeout:
  - keep the work inside the existing host facade instead of adding a fourth
    public facade
  - define an explicit Runtime-owned bloc root policy instead of inheriting
    Core defaults silently
  - make `LLM_CALL` execution binding-aware, not just host-control aware
- Refinements shipped:
  - extended `AbstractCoreHostFacade`
  - added explicit local/file-runtime bloc-root defaults
  - normalized and enforced binding precedence in effect handling
- Cycle 2: broader regression and compatibility review found one meaningful
  issue:
  - `MultiLocalAbstractCoreLLMClient` could break monkeypatched or older test
    doubles that do not accept the new `bloc_root_dir` constructor kwarg
- Final refinement shipped:
  - added a narrow compatibility fallback in `_get_client(...)` for local
    client call sites that reject only the new `bloc_root_dir` kwarg
- Cycle 3: independent implementation review found one real remote-path
  blocker plus two coverage gaps:
  - remote bloc/KV proxy calls must not carry local selector fields when
    `base_url` is supplied
  - the direct-client `expected_prompt_cache_binding` alias path needed its own
    regression
  - the primary `MultiLocal` local deployment path and keyless-binding
    precedence needed direct coverage
- Final refinements shipped:
  - `base_url` now wins cleanly for remote bloc/KV proxy calls
  - direct client binding-alias normalization is covered and aligned with
    Runtime semantics
  - tests now cover the factory/local `MultiLocal` durable-bloc path and the
    keyless-binding no-derived-key rule
- Final blocker-only review reported no blockers.

### Tests

- `tests/test_abstractcore_host_facade.py`
- `tests/test_prompt_cache_modules.py`
- `tests/test_remote_llm_client.py`
- `tests/test_packaging_extras.py`
- regression coverage alongside:
  - `tests/test_abstractcore_discovery_facade.py`
  - `tests/test_abstractcore_run_facade.py`
  - `tests/test_model_residency_control_plane.py`
  - `tests/test_multimodal_abstractcore_integration.py`
  - `tests/test_runtime_install_boundary.py`

### Docs

- `README.md`
- `docs/README.md`
- `docs/getting-started.md`
- `docs/integrations/abstractcore.md`
- `docs/api.md`
- `docs/faq.md`
- `llms.txt`
- `llms-full.txt`
- `docs/adr/0006_runtime_owns_durable_abstractcore_bloc_prompt_cache.md`

### Residual risks

- This repo now provides the Runtime-side durable bloc contract, but sibling
  hosts still need to adopt it instead of provider-private snapshot flows where
  those still exist.
- Provider-native snapshot save/load remains a separate local-admin concern and
  should not be confused with the primary durable bloc contract.

### Follow-ups

- Evaluate whether Gateway should expose app-facing durable bloc routes on top
  of this Runtime surface.
- Keep the separate local-admin snapshot item proposed until there is a real
  operator need beyond durable bloc/KV/binding reuse.
