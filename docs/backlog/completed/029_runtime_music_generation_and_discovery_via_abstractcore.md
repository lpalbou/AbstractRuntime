# 029 Runtime Music Generation And Discovery Via AbstractCore

## Metadata
- Created: 2026-05-21
- Status: Completed
- Completed: 2026-05-21
- Origin: direct implementation from repository backlog/user request

## Goal

Surface AbstractCore-backed music capabilities through Runtime without breaking
the existing boundary rule that Runtime integrates only with AbstractCore, not
directly with `abstractmusic` or other modality packages.

## What Shipped

- Extended the public Runtime-owned discovery facade with music snapshot
  methods:
  - `list_music_providers(...)`
  - `list_music_models(...)`
- Extended the public durable run facade with:
  - `generate_music(...)`
- Extended remote AbstractCore multimodal routing so Runtime can execute music
  generation through AbstractCore Server's `/v1/audio/music` contract and store
  the returned bytes as normal Runtime artifacts.
- Extended local discovery helpers so local runtimes can surface music provider
  and model catalogs through AbstractCore capability registries when the
  configured Core stack exposes them.
- Normalized `music` generated outputs into the same JSON-safe artifact-backed
  result shape used for image and voice outputs.
- Kept the package boundary clean:
  - no direct Runtime import of `abstractmusic`
  - no direct Runtime import of `abstractvoice`
  - no direct Runtime import of `abstractvision`
- Updated the `multimodal` extra to include `abstractcore[music]` so the common
  Runtime media install matches the public surface.
- Updated docs and LLM indexes so the repo no longer describes music support as
  future-only work.

## Current Code Pointers

- `src/abstractruntime/integrations/abstractcore/discovery_queries.py`
- `src/abstractruntime/integrations/abstractcore/discovery_facade.py`
- `src/abstractruntime/integrations/abstractcore/llm_client.py`
- `src/abstractruntime/integrations/abstractcore/run_facade.py`
- `tests/test_abstractcore_discovery_facade.py`
- `tests/test_multimodal_abstractcore_integration.py`
- `tests/test_abstractcore_run_facade.py`
- `tests/test_packaging_extras.py`
- `docs/integrations/abstractcore.md`
- `docs/api.md`
- `docs/faq.md`

Related ADRs already covered the boundary and ownership rules:
- `docs/adr/0004_runtime_owns_run_scoped_media_execution_truth.md`
- `docs/adr/0005_runtime_owns_abstractcore_host_discovery_queries.md`

## Validation

Focused validation run on 2026-05-21:

- `pytest -q tests/test_abstractcore_discovery_facade.py tests/test_multimodal_abstractcore_integration.py tests/test_abstractcore_run_facade.py`
- Result: `58 passed in 0.30s`

Broader regression and packaging validation:

- `PYTHONPATH="/Users/albou/tmp/abstractframework/abstractcore:src" pytest -q tests/test_abstractcore_discovery_facade.py tests/test_multimodal_abstractcore_integration.py tests/test_abstractcore_run_facade.py tests/test_remote_llm_client.py tests/test_packaging_extras.py tests/test_runtime_install_boundary.py tests/test_abstractcore_host_facade.py`
- Result: `84 passed in 0.51s`

Syntax/import sanity:

- `python -m compileall src/abstractruntime/integrations/abstractcore`
- Result: clean

Documentation build sanity:

- `mkdocs build -q --site-dir /tmp/abstractruntime-music-docs`
- Result: build passed; upstream Material for MkDocs emitted a non-blocking
  roadmap warning about future MkDocs 2.0 compatibility

Practical proof points from the shipped tests:

- Remote Runtime now proxies music discovery through:
  - `GET /v1/audio/music/providers`
  - `GET /v1/audio/music/models`
- Remote Runtime now executes music generation through:
  - `POST /v1/audio/music`
- Durable run-scoped music execution now works through
  `get_abstractcore_run_facade(runtime).generate_music(...)`, producing a child
  run with artifact-backed output.
- Live ACE Music smoke proof against AbstractCore `2.13.24` plus
  `abstractmusic>=0.1.4`:
  - direct Runtime local client call returned `audio/mp3` output with
    `media_provider='abstractmusic:acemusic'`
  - full Runtime `LLM_CALL` workflow completed durably with `ledger_entries=4`
    and an artifact-backed MP3 result

## Completion Report

### Date

2026-05-21

### Summary

Runtime now surfaces music in the same three Runtime-owned ways that it already
surfaces other generated-media capabilities:

- discovery snapshots for host/query use
- durable run-scoped execution for existing runs
- artifact-backed normalized outputs for replay-safe state

The boundary remains correct: Runtime still talks only to AbstractCore and
AbstractCore Server contracts. Music capability packages remain behind
AbstractCore.

### Behavior Changes

- Hosts can ask Runtime for music provider and model catalogs through the
  public discovery facade instead of querying Core directly.
- Hosts can ask Runtime to durably execute run-scoped music generation through
  the public run facade instead of doing out-of-band work and synthesizing
  history afterward.
- Remote music bytes are persisted as Runtime artifacts with the same durable
  output shape used by image and voice generation.
- The common `multimodal` install now includes the Core music extra so local
  Runtime installs can expose music when the selected Core profile supports it.
- Runtime is now validated against the `abstractcore>=2.13.24` floor and the
  lightweight remote ACE Music backend path from `abstractmusic>=0.1.4`.

### Review Notes

- No new ADR was added. This change is an extension of existing accepted
  Runtime policy:
  - Runtime owns run-scoped media execution truth.
  - Runtime owns host discovery boundaries.
- The main regression caught during implementation was that remote generated
  media had briefly stopped preserving `output.run_id` and `output.tags` for
  artifact storage. That was fixed before closure.

### Tests

- `tests/test_abstractcore_discovery_facade.py`
- `tests/test_multimodal_abstractcore_integration.py`
- `tests/test_abstractcore_run_facade.py`
- regression support in:
  - `tests/test_remote_llm_client.py`
  - `tests/test_packaging_extras.py`
  - `tests/test_runtime_install_boundary.py`
  - `tests/test_abstractcore_host_facade.py`

### Docs

- `README.md`
- `docs/integrations/abstractcore.md`
- `docs/api.md`
- `docs/faq.md`
- `docs/architecture.md`
- `llms.txt`
- `llms-full.txt`

### Residual Risks

- Runtime is ready for the AbstractCore music boundary, but the exact local
  backend matrix still depends on which AbstractCore capability packages and
  backends a host installs.
- Remote music currently follows the current AbstractCore `/v1/audio/music`
  contract. If Core changes provider/backend routing semantics, Runtime should
  keep mirroring that public contract rather than growing direct modality
  knowledge.

### Follow-Ups

- Gateway/other hosts should adopt the new Runtime discovery and run facades
  for music instead of querying or invoking Core directly.
- If a later Runtime surface needs music-specific operator controls, keep them
  behind the existing Runtime-to-AbstractCore boundary rather than importing
  `abstractmusic` directly.
