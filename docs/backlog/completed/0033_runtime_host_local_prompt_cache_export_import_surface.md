# 0033 Runtime Host-Local Prompt-Cache Export/Import Surface

## Metadata
- Created: 2026-05-20
- Status: Completed
- Completed: 2026-05-21
- Origin: implemented from repository backlog and user request

## Goal

Remove the remaining Gateway prompt-cache export/import bypass by giving Runtime
a public host-facing surface for:

- listing host-local prompt-cache exports
- exporting one live local provider cache to disk
- importing one previously exported local provider cache

without confusing that admin workflow with Runtime's durable bloc/binding
memory model.

## What Shipped

- Extended `AbstractCoreHostFacade` with:
  - `list_prompt_cache_exports(...)`
  - `prompt_cache_export(...)`
  - `prompt_cache_import(...)`
- Implemented the local behavior in Runtime's AbstractCore clients on top of
  Core's existing **public** provider hooks:
  - `prompt_cache_save(...)`
  - `prompt_cache_load(...)`
  - `prompt_cache_artifact_extension()`
  - `prompt_cache_artifact_format()`
  - prompt-cache capability flags
- Kept the surface honest:
  - **local-only** for local/file runtimes
  - remote and hybrid return structured `prompt_cache_local_only` payloads and
    do not issue HTTP requests
- Added a Runtime-owned export root policy:
  - default local root: `~/.abstractruntime/prompt_cache_exports`
  - default file-runtime root: `<base_dir>/prompt_cache_exports`
  - explicit `prompt_cache_export_root_dir=...` overrides are supported
- Added Runtime-owned listing/catalog metadata so hosts no longer need a
  Gateway-local `saved/` directory convention
- Kept per-provider/model separation exact by partitioning export directories
  with reversible encoded provider/model ids instead of lossy slugs
- Preserved the broader prompt-cache boundary:
  - durable app-facing exact reuse is still `bloc + KV artifact + prompt_cache_binding`
  - host-local export/import remains secondary operator/admin tooling around
    live local provider cache state

## Current Code Pointers

- `src/abstractruntime/integrations/abstractcore/host_facade.py`
- `src/abstractruntime/integrations/abstractcore/llm_client.py`
- `src/abstractruntime/integrations/abstractcore/factory.py`
- `tests/test_prompt_cache_export_import.py`
- `tests/test_abstractcore_host_facade.py`
- `tests/test_remote_llm_client.py`
- `docs/integrations/abstractcore.md`
- `docs/api.md`
- `docs/faq.md`
- `README.md`

Related accepted prompt-cache policy still lives in:
- `docs/adr/0006_runtime_owns_durable_abstractcore_bloc_prompt_cache.md`

No new ADR was added. This item extends the existing host-surface boundary
without changing the durable app contract.

## Validation

Focused validation run on 2026-05-21:

- `pytest -q tests/test_prompt_cache_export_import.py tests/test_abstractcore_host_facade.py tests/test_remote_llm_client.py`
- Result: `29 passed in 0.24s`

Broader regression and packaging-adjacent sanity:

- `pytest -q tests/test_prompt_cache_export_import.py tests/test_abstractcore_host_facade.py tests/test_remote_llm_client.py tests/test_prompt_cache_modules.py tests/test_packaging_extras.py`
- Result: `48 passed in 0.21s`

Syntax/import sanity:

- `python -m compileall src/abstractruntime/integrations/abstractcore`
- Result: clean

Documentation build sanity:

- `mkdocs build -q --site-dir /tmp/abstractruntime-0033-docs`
- Result: build passed; upstream Material for MkDocs emitted its current
  non-blocking MkDocs 2.0 warning

Practical proof covered by the shipped tests:

- local/multi-local Runtime can export, list, and import prompt-cache
  artifacts through Runtime's public host facade and local client path
- the same logical export name can coexist for distinct provider/model ids
  without directory collisions
- local listing no longer requires loading the target provider/model client to
  exist on disk
- remote Runtime returns `prompt_cache_local_only` for list/export/import and
  does not emit HTTP traffic
- `create_local_file_runtime(...)` now wires the default export root to
  `<base_dir>/prompt_cache_exports`

## Completion Report

### Date

2026-05-21

### Summary

Runtime now owns the last prompt-cache admin surface that Gateway was still
handling through private provider/runtime internals.

Hosts can keep manual provider-native export/import when they genuinely need
it, but the contract is now:

- public Runtime boundary
- Core public hooks underneath
- local-only filesystem semantics made explicit
- clean separation from durable bloc/binding workflow memory

### Behavior Changes

- `get_abstractcore_host_facade(runtime)` now exposes:
  - `list_prompt_cache_exports(...)`
  - `prompt_cache_export(...)`
  - `prompt_cache_import(...)`
- Local/file runtimes store host-local prompt-cache exports under a Runtime
  root rather than letting Gateway or another host invent its own unmanaged
  directory contract.
- Listing is now Runtime-owned and returns catalog metadata for one
  provider/model partition.
- `prompt_cache_import(clear_existing=True)` now validates the artifact before
  clearing live cache state, then performs the clear/load sequence so a corrupt
  file does not wipe the old cache first.
- When the caller omits `key` on import, Runtime now surfaces the effective
  loaded key returned by the provider instead of dropping it.

### Contract Notes

- This is intentionally a **host-local admin** contract. It is not a durable
  workflow effect and it is not replayed like `LLM_CALL`.
- Runtime currently returns `root_dir`, `artifact_path`, and `meta_path` in the
  public payload because this is a host/operator surface and Gateway still
  needs a real listing/catalog replacement. That path exposure is intentional
  for this local admin surface, not the general durable app contract.
- Remote/hybrid runtimes do not proxy or fake host-local export roots. They
  fail honestly with `prompt_cache_local_only`.

### Tests

- `tests/test_prompt_cache_export_import.py`
- `tests/test_abstractcore_host_facade.py`
- regression support in:
  - `tests/test_remote_llm_client.py`

### Docs

- `README.md`
- `docs/integrations/abstractcore.md`
- `docs/api.md`
- `docs/faq.md`
- `llms.txt`
- `llms-full.txt`

### Residual Risks

- This is still a secondary operator feature. The primary prompt-cache story
  for durable apps remains bloc/KV artifacts plus `prompt_cache_binding`.
- Import/export support still depends on the local AbstractCore backend family
  and the public `supports_save` / `supports_load` capability flags.
- Gateway adoption is still separate work in the sibling package before
  `/prompt_cache/saved|save|load` is fully cleaned up end-to-end.

### Follow-Ups

- Gateway should adopt these Runtime host-facade methods and remove its
  remaining private `/prompt_cache/saved|save|load` implementation.
- Keep AbstractCore item `0797` only as optional follow-up memory. Promote it
  only if Runtime/Gateway later prove they need a narrower cross-provider
  metadata/compatibility contract than the current public hooks provide.
