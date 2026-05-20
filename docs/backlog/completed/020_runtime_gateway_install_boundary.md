# 020 Runtime Gateway Install Boundary

## Metadata
- Created: 2026-05-08
- Status: Completed
- Completed: 2026-05-08
- Origin: moved from `docs/backlog/proposed/2026-05-08_runtime_gateway_install_boundary.md` during backlog housekeeping on 2026-05-20

## Goal

Keep `abstractruntime` usable as a durable kernel without forcing Gateway/Core/media/provider dependency choices into
the base package, while still exposing explicit extras and profile cascades for Runtime's AbstractCore integration.

## What Shipped

- `abstractruntime` keeps its durable kernel boundary separate from optional AbstractCore integration code.
- AbstractCore support remains opt-in through package extras such as `abstractruntime[abstractcore]` and
  `abstractruntime[multimodal]`.
- Runtime now exposes explicit profile-cascade extras (`apple`, `gpu`, `all-apple`, `all-gpu`) instead of requiring
  hosts to guess how to align local-engine install profiles.
- The package metadata floor has since advanced to `abstractcore>=2.13.23`; the original completion landed on an older
  2.13.12-aligned floor and was later tightened without changing the boundary itself.
- Regression tests cover package metadata, optional-stack imports, and the rule that Gateway env/auth does not leak
  into Runtime's remote AbstractCore client boundary.

## Current Code Pointers

- `pyproject.toml`
- `src/abstractruntime/integrations/abstractcore/__init__.py`
- `src/abstractruntime/integrations/abstractcore/factory.py`
- `docs/integrations/abstractcore.md`
- `tests/test_packaging_extras.py`
- `tests/test_runtime_install_boundary.py`
- `tests/test_remote_llm_client.py`

## Validation

Current focused validation run on 2026-05-20:

- `pytest -q tests/test_packaging_extras.py tests/test_runtime_install_boundary.py tests/test_remote_llm_client.py`
- Result: `8 passed in 0.40s`

Broader adjacent validation run on 2026-05-20:

- `pytest -q tests/test_packaging_extras.py tests/test_runtime_install_boundary.py tests/test_remote_llm_client.py tests/test_prompt_cache_modules.py tests/test_session_attachments_registry_and_open_tool.py tests/test_workflow_bundle_registry.py tests/test_model_residency_control_plane.py`
- Result: `51 passed, 3 skipped in 0.83s`

## Completion report

### Date

2026-05-20

### Summary

This item was already implemented in code but had been left in `docs/backlog/proposed/`. The current tree still
matches the intended boundary: Runtime base stays clean, AbstractCore remains explicit, profile cascades are exposed
through extras, and tests prove the packaging/import boundary.

### Behavior changes

- Optional integration/package extras are the mechanism for pulling in AbstractCore-dependent behavior.
- Runtime profile extras now mirror the Core local-engine and aggregate profile aliases explicitly.
- Runtime docs describe the host handoff boundary instead of implying that Gateway config or provider stacks belong in
  the kernel.

### Tests

- `tests/test_packaging_extras.py`
- `tests/test_runtime_install_boundary.py`
- `tests/test_remote_llm_client.py`

### Docs

- `docs/integrations/abstractcore.md`
- `docs/api.md`
- `docs/getting-started.md`
- `llms-full.txt`

### Residual risks

- Version-floor references in historical backlog text had drifted behind the current `abstractcore>=2.13.23` baseline.
  That drift is now explicit here, but other historical backlog notes may still mention `2.13.12`.
- This item validates the package/import boundary, not every higher-level deployment profile that Gateway may choose.

### Backlog/code drift found

- The backlog file had `Status: Completed` metadata but remained stranded under `proposed/`.
- Current code reality is stricter than the original text in one respect: the package floor has advanced past the
  originally documented baseline.

### Follow-ups

- Keep future package-floor bumps synchronized across docs, tests, and backlog history.
- Preserve the rule that Gateway chooses deployment composition while Runtime exposes explicit integration extras.
