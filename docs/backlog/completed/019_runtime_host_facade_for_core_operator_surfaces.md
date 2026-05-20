# Runtime Host Facade For Core Operator Surfaces

## Metadata
- Created: 2026-05-20
- Status: Completed
- Completed: 2026-05-20
- Origin: promoted directly from `docs/backlog/proposed/2026-05-20_runtime_host_facade_for_core_operator_surfaces.md`

## Context

`abstractruntime.integrations.abstractcore` already owned the real control-plane behavior for:

- prompt-cache operations
- model-residency operations
- local/remote/hybrid runtime topology normalization

But hosts had no stable public Runtime-facing contract for those surfaces. In practice, they either:

1. reached through `runtime._abstractcore_llm_client`, or
2. rebuilt control-plane behavior outside Runtime

That was inconsistent with the broader Runtime mission: keep execution topology and control semantics inside Runtime so
hosts request work from Runtime rather than re-deriving integration behavior themselves.

## Current Code Reality At Implementation Time

Already implemented before this item was completed:

- prompt-cache control methods on local, multi-local, and remote AbstractCore clients
- `MODEL_RESIDENCY` as a Runtime effect
- truthful local media-residency constraints

Missing or incomplete before this item:

- a public facade/export path for host code
- documentation telling hosts to use that public path
- tests proving the public facade works across local, remote, and hybrid runtime factories

## Problem

Hosts needed a narrow, documented control-plane surface for prompt-cache and model-residency operations without:

- hard-coupling `Runtime` itself to an AbstractCore-specific public accessor
- normalizing too much Gateway/product-level readiness logic into Runtime
- implying that operator actions are durable run truth

## What We Implemented

Phase 1 shipped the narrow public host facade proposed in the backlog review:

- `AbstractCoreHostFacade`
- `get_abstractcore_host_facade(...)`

Scope kept intentionally narrow:

- prompt-cache operations
- model-residency operations

Explicit non-goals preserved:

- no media/TTS/STT helper methods in this item
- no discovery/catalog/readiness aggregation in this item
- no provider-private prompt-cache save/load promotion in this item
- no change to the rule that durable run truth must still flow through Runtime effects/commands

## Files And Symbols Touched

Implementation:

- `src/abstractruntime/integrations/abstractcore/host_facade.py`
- `src/abstractruntime/integrations/abstractcore/__init__.py`
- `src/abstractruntime/integrations/abstractcore/factory.py`

Tests:

- `tests/test_abstractcore_host_facade.py`

Docs:

- `docs/integrations/abstractcore.md`
- `docs/api.md`
- `docs/faq.md`
- `llms-full.txt`
- `docs/backlog/README.md`

## Expected Outcomes

The following are now true:

- hosts have a documented public facade for prompt-cache and model-residency controls
- the facade delegates to the real AbstractCore-backed Runtime topology instead of duplicating logic
- local, remote, and hybrid runtime factories all expose the same host-facade entrypoint
- docs now point hosts at the public facade instead of instructing them to use the private runtime attachment directly
- factory attachment failures fail fast instead of surfacing as a later opaque facade error

## Validation

Primary validation:

- `pytest -q tests/test_abstractcore_host_facade.py`
- `pytest -q tests/test_abstractcore_host_facade.py tests/test_prompt_cache_modules.py tests/test_model_residency_control_plane.py`

Example proof run:

- `PYTHONPATH=src python` example using `create_remote_runtime(...)` + `get_abstractcore_host_facade(...)` with a stub request sender

## Completion Report

### Date

2026-05-20

### Summary

Implemented a public, integration-scoped host facade for AbstractCore prompt-cache and model-residency control
operations. The facade is exported from `abstractruntime.integrations.abstractcore`, documented as the host entrypoint,
and validated across local, remote, and hybrid runtime construction paths.

### Behavior Changes

- Added `AbstractCoreHostFacade` as the public host-control surface for prompt-cache and model-residency operations.
- Added `get_abstractcore_host_facade(runtime)` as the documented integration-scoped helper.
- Exported the facade from `abstractruntime.integrations.abstractcore`.
- Kept the underlying runtime attachment private, but made factory attachment fail fast if it cannot be installed.
- Replaced public docs guidance that previously told hosts to access `_abstractcore_llm_client` directly.

### Tests

Local validation run in this completion pass:

- `tests/test_abstractcore_host_facade.py`: `12 passed in 0.20s`
- `tests/test_abstractcore_host_facade.py tests/test_prompt_cache_modules.py tests/test_model_residency_control_plane.py`: `32 passed in 0.25s`

Independent test-agent validation:

- `tests/test_abstractcore_host_facade.py`: `12 passed`
- `tests/test_prompt_cache_modules.py`: `10 passed`
- `tests/test_model_residency_control_plane.py`: `10 passed`
- aggregate: `32 passed`, `0 failed`, `0 skipped`, `3.00s` aggregate wall time

### Example Evidence

The example proof run used a factory-created remote runtime plus the public facade and produced:

- prompt-cache capabilities response with `mode=keyed`
- model-residency load response with `runtime_id=rid-42`
- provider API key forwarded as `X-AbstractCore-Provider-API-Key`
- correct root-level control URLs:
  - `http://core.example/acore/prompt_cache/capabilities?...`
  - `http://core.example/acore/models/load`
- combined example execution time: `0.056 ms` with an in-memory stub sender

### Docs Updated

- integration guide now shows `get_abstractcore_host_facade(...)`
- API reference lists the public host facade entrypoint
- FAQ points hosts at the facade rather than the private runtime attachment
- `llms-full.txt` reflects the new public contract

### Residual Risks

- The facade still depends internally on the private `_abstractcore_llm_client` runtime attachment; that coupling is
  now explicit and fail-fast, but it remains an internal integration detail worth revisiting in future refactors.
- The remote-runtime proof path swaps the underlying client's private `_sender` in tests/example setup, so that one
  validation path is not purely black-box from the public facade surface.
- There is no dedicated test for the exceptional attach-failure path in `factory.py`; success-path wiring is covered.
- The docs snippets are not yet executed as tests.

### Backlog / Code Drift Found

- During this completion pass, the stranded residency proposal was cleaned up separately into
  `docs/backlog/completed/022_model_residency_control_plane.md`.
- No `docs/backlog/overview.md` exists in this repository; `docs/backlog/README.md` currently serves as the effective
  backlog overview and was updated accordingly.
- No `docs/backlog/recurrent/` directory exists here, so there were no recurrent-task files to scan in this pass.

### Follow-Up Impact

- This completion narrows and stabilizes the Runtime-owned control plane for host code.
- The remaining adjacent boundary work is still represented by the separate local media truthfulness proposal and the
  sibling Gateway run-truth item discussed in backlog review.
