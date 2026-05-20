# 026 Runtime Host Discovery Facade for Core Catalogs

## Metadata
- Created: 2026-05-20
- Status: Completed
- Completed: 2026-05-20
- Origin: promoted from planned during Runtime/Core boundary work

## Goal

Give hosts a public Runtime-owned path for AbstractCore discovery/catalog snapshot
queries so Gateway and other hosts can stop rebuilding Core discovery logic or
calling Core directly for provider/media catalogs.

## What Shipped

- Added `AbstractCoreDiscoveryFacade` and
  `get_abstractcore_discovery_facade(runtime)`.
- Added Runtime-owned discovery methods for:
  - provider discovery
  - provider model discovery
  - model capability lookup
  - voice catalog / TTS / STT model catalogs
  - vision provider model catalogs
  - cached vision model snapshots
- Wired the new surface through local, remote, and hybrid AbstractCore runtime
  clients.
- Documented the three distinct public Runtime boundaries:
  - discovery snapshots
  - operator/control prompt-cache + residency
  - durable run-scoped media execution

## Current Code Pointers

- `src/abstractruntime/integrations/abstractcore/discovery_facade.py`
- `src/abstractruntime/integrations/abstractcore/discovery_queries.py`
- `src/abstractruntime/integrations/abstractcore/llm_client.py`
- `src/abstractruntime/integrations/abstractcore/__init__.py`
- `tests/test_abstractcore_discovery_facade.py`
- `docs/integrations/abstractcore.md`
- `docs/api.md`
- `docs/faq.md`
- `docs/adr/0005_runtime_owns_abstractcore_host_discovery_queries.md`

## Validation

Focused validation run on 2026-05-20:

- `pytest -q tests/test_abstractcore_discovery_facade.py tests/test_abstractcore_host_facade.py tests/test_abstractcore_run_facade.py tests/test_model_residency_control_plane.py tests/test_multimodal_abstractcore_integration.py tests/test_remote_llm_client.py tests/test_prompt_cache_modules.py`
- Result: `90 passed, 1 warning in 0.33s`

Additional validation:

- `python -m compileall src/abstractruntime/integrations/abstractcore`
- Result: clean

Practical smoke run with a runtime-bound discovery facade:

- Remote-shaped runtime:
  - providers: `['ollama', 'openai']`
  - voice providers: `['openai']`
  - vision providers: `['mflux']`
  - cached vision models: `1`
  - HTTP calls: `4`
  - elapsed: `0.215 ms`
- Local-shaped runtime:
  - providers: `['mlx']`
  - voice providers: `['openai']`
  - vision providers: `['mflux']`
  - cached vision models: `1`
  - elapsed: `0.453 ms`

## Completion report

### Date

2026-05-20

### Summary

Runtime now exposes a public discovery/query surface for AbstractCore-backed
hosts. Hosts can ask Runtime for provider catalogs, media capability catalogs,
and cached vision model snapshots instead of rebuilding that logic or proxying
Core directly. The surface is intentionally snapshot-oriented rather than
durable-run-oriented: it is the Runtime integration boundary for ephemeral
reads, not a new effect type.

### Behavior changes

- Hosts can bind `get_abstractcore_discovery_facade(runtime)` and use one
  public Runtime surface for provider/media discovery.
- Remote discovery now preserves upstream `status_code` and parsed
  `upstream_error` payloads when proxy calls fail.
- Remote discovery supports per-call `timeout_s` overrides.
- Model capability lookup is normalized through the shared Runtime helper path,
  then wrapped into a stable facade response shape.

### Review and refinement cycles

- Cycle 1: independent review found three meaningful issues:
  - remote discovery error payloads were too lossy
  - remote discovery dropped per-call timeout overrides
  - model capability lookup was still duplicated in client code
- Refinements shipped:
  - preserved `status_code` / `upstream_error` in remote discovery failures
  - threaded `timeout_s` through remote discovery calls
  - routed local/remote capability lookup through the shared helper path
- Cycle 2: final internal audit after those fixes plus the broadened regression
  suite found one remaining blocker:
  - capability lookup still dropped discovery failure metadata
- Final refinements shipped:
  - added a discovery-facing `lookup_model_capabilities(...)` path so the
    public facade preserves `available`, `error`, and `source`
  - kept backward compatibility for custom clients that still only expose
    raw `get_model_capabilities(...)`
  - transport-level remote discovery failures now report
    `route_available=false`
- Final blocker-only review reported no blockers. A later Core follow-up landed
  the missing public cached local vision helper in AbstractCore 2.13.20, and
  Runtime now depends on that public seam instead of `abstractcore.server.*`.

### Tests

- `tests/test_abstractcore_discovery_facade.py`
- regression coverage alongside:
  - `tests/test_abstractcore_host_facade.py`
  - `tests/test_abstractcore_run_facade.py`
  - `tests/test_model_residency_control_plane.py`
  - `tests/test_multimodal_abstractcore_integration.py`
  - `tests/test_remote_llm_client.py`
  - `tests/test_prompt_cache_modules.py`

### Docs

- `docs/integrations/abstractcore.md`
- `docs/api.md`
- `docs/faq.md`
- `docs/adr/README.md`

### Residual risks

- Local discovery remains synchronous helper code. Async hosts should offload it
  to a worker thread if they do not want to block their event loop.
- This repo now provides the Runtime-side migration target, but sibling hosts
  still need to adopt it.

### Follow-ups

- Migrate Gateway/other hosts to call
  `get_abstractcore_discovery_facade(runtime)` for provider/media discovery
  routes.
- Track dependency floors so Runtime keeps requiring a Core version that
  includes `abstractcore.capabilities.get_local_vision_cache_catalog()`.
