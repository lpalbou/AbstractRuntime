# 028 Runtime Bloc/KV Lifecycle And Pruning

## Metadata
- Created: 2026-05-20
- Status: Completed
- Completed: 2026-05-21
- Origin: promoted from `proposed/2026-05-20_runtime_bloc_kv_lifecycle_and_pruning.md`

## Goal

Close the remaining durable bloc lifecycle gap so Runtime can list, inspect,
delete, and prune AbstractCore-backed bloc/KV storage through the public host
facade instead of leaving hosts to manual filesystem deletion or private Core
hooks.

## What Shipped

- Extended `AbstractCoreHostFacade` with durable bloc lifecycle methods:
  - `list_blocs(...)`
  - `list_bloc_kv_artifacts(...)`
  - `delete_bloc_kv_artifact(...)`
  - `prune_bloc_kv_artifacts(...)`
  - `delete_bloc(...)`
- Wired those methods through:
  - local `LocalAbstractCoreLLMClient`
  - standard local `MultiLocalAbstractCoreLLMClient`
  - remote/hybrid `RemoteAbstractCoreLLMClient`
- Added Runtime-side local lifecycle helpers that normalize Core delete/list
  results into the same JSON-safe host payload shape as the rest of the facade.
- Preserved Core safety semantics on the Runtime boundary:
  - `dry_run=True` preview
  - `clear_loaded=True` live-key clearing when Runtime can see the resident
    provider state
  - `force=True` explicit override
- Added `MultiLocal` orchestration so shared local bloc roots can delete/prune
  across several loaded provider/model clients without inventing raw filesystem
  rules.
- Updated docs, FAQ, API docs, and LLM indexes so the repo no longer claims
  this lifecycle surface is missing.

## Current Code Pointers

- `src/abstractruntime/integrations/abstractcore/host_facade.py`
- `src/abstractruntime/integrations/abstractcore/llm_client.py`
- `tests/test_abstractcore_host_facade.py`
- `tests/test_prompt_cache_modules.py`
- `docs/integrations/abstractcore.md`
- `docs/faq.md`
- `docs/api.md`
- `docs/adr/0006_runtime_owns_durable_abstractcore_bloc_prompt_cache.md`

## Validation

Focused lifecycle and regression run on 2026-05-21:

- `PYTHONPATH=src pytest -q tests/test_abstractcore_host_facade.py tests/test_prompt_cache_modules.py tests/test_remote_llm_client.py tests/test_packaging_extras.py`
- Result: `41 passed in 0.23s`

Additional focused lifecycle slice:

- `PYTHONPATH=src pytest -q tests/test_abstractcore_host_facade.py tests/test_prompt_cache_modules.py`
- Result: `33 passed in 0.26s`

Additional syntax validation:

- `python -m py_compile src/abstractruntime/integrations/abstractcore/llm_client.py src/abstractruntime/integrations/abstractcore/host_facade.py tests/test_abstractcore_host_facade.py tests/test_prompt_cache_modules.py`
- Result: clean

Practical proof points from the shipped tests:

- Remote Runtime now proxies:
  - `GET /acore/blocs`
  - `GET /acore/blocs/kv/list`
  - `POST /acore/blocs/kv/delete`
  - `POST /acore/blocs/kv/prune`
  - `POST /acore/blocs/delete`
- Local Runtime now lists bloc records and per-model artifacts from the
  Runtime-owned bloc root, then deletes one artifact or a whole bloc through
  the same host facade contract.
- `MultiLocal` local runtimes keep the shared bloc root behavior but can still
  route lifecycle operations through the correct loaded provider/model client
  when live prompt-cache state matters.

## Completion Report

### Date

2026-05-21

### Summary

Runtime now owns the durable bloc lifecycle boundary as well as the durable
bloc create/read/load boundary. Hosts can enumerate durable bloc records,
inspect per-model KV artifacts, delete one artifact while keeping the durable
text bloc, prune matching artifacts by filter, or delete the whole bloc and its
derived artifacts through `get_abstractcore_host_facade(runtime)`.

### Behavior Changes

- The host facade is now sufficient for the full durable bloc lifecycle, not
  just durable bloc creation and loading.
- Local and file-backed runtimes no longer require manual bloc-root inspection
  or filesystem deletion for supported operator cleanup flows.
- Remote and hybrid runtimes can drive the same lifecycle operations through
  Core's public `/acore/blocs*` lifecycle routes.

### Review Notes

- The implementation kept lifecycle work on the existing host facade instead of
  adding another public Runtime surface.
- The main tricky point was `MultiLocal` safety: one shared bloc root can back
  several loaded provider/model clients, so Runtime now resolves live-state
  checks through the currently loaded client when possible instead of guessing
  from files alone.
- Runtime still keeps a graceful `dependency_missing` response path at the
  facade boundary, but the published dependency floor now matches the released
  Core lifecycle helpers.

### Tests

- `tests/test_abstractcore_host_facade.py`
- `tests/test_prompt_cache_modules.py`
- regression support in:
  - `tests/test_remote_llm_client.py`
  - `tests/test_packaging_extras.py`

### Docs

- `docs/integrations/abstractcore.md`
- `docs/faq.md`
- `docs/api.md`
- `llms.txt`
- `llms-full.txt`
- `docs/adr/0006_runtime_owns_durable_abstractcore_bloc_prompt_cache.md`

### Residual Risks

- `clear_loaded=True` can only clear live keys that Runtime can actually see in
  the current local process or the remote Core server. External provider state
  outside the public Core contract remains out of scope.

### Follow-Ups

- Keep the separate local-admin snapshot save/load/list proposal independent of
  this durable bloc lifecycle contract.
