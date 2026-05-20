# 021 Runtime Gateway Env Namespace Cleanup

## Metadata
- Created: 2026-05-08
- Status: Completed
- Completed: 2026-05-08
- Origin: moved from `docs/backlog/proposed/2026-05-08_runtime_gateway_env_namespace_cleanup.md` during backlog housekeeping on 2026-05-20

## Goal

Keep Gateway-owned configuration in Gateway. Runtime may expose Runtime-owned env defaults, but it should not change
behavior because `ABSTRACTGATEWAY_*` happens to be present in the process environment.

## What Shipped

- Runtime no longer reads `ABSTRACTGATEWAY_PROMPT_CACHE`.
- Runtime no longer reads `ABSTRACTGATEWAY_MAX_ATTACHMENT_BYTES`.
- Runtime no longer reads `ABSTRACTGATEWAY_FLOWS_DIR`.
- Prompt-cache defaults now flow through explicit Runtime inputs such as `_runtime.prompt_cache`,
  `LLM_CALL.params.prompt_cache_key`, or the Runtime-owned `ABSTRACTRUNTIME_PROMPT_CACHE`.
- Attachment registration limits now flow through explicit Runtime inputs such as
  `TOOL_CALLS.payload.max_attachment_bytes`, `_runtime.max_attachment_bytes`, or the Runtime-owned
  `ABSTRACTRUNTIME_MAX_ATTACHMENT_BYTES`.
- Workflow bundle default directory resolution now prefers shared/framework env names instead of Gateway-specific ones.

## Current Code Pointers

- `src/abstractruntime/integrations/abstractcore/effect_handlers.py`
- `src/abstractruntime/workflow_bundle/registry.py`
- `docs/integrations/abstractcore.md`
- `docs/workflow-bundles.md`
- `tests/test_prompt_cache_modules.py`
- `tests/test_session_attachments_registry_and_open_tool.py`
- `tests/test_workflow_bundle_registry.py`

## Validation

Current focused validation run on 2026-05-20:

- `pytest -q tests/test_prompt_cache_modules.py tests/test_session_attachments_registry_and_open_tool.py tests/test_workflow_bundle_registry.py`
- Result: `33 passed, 3 skipped in 0.62s`

Broader adjacent validation run on 2026-05-20:

- `pytest -q tests/test_packaging_extras.py tests/test_runtime_install_boundary.py tests/test_remote_llm_client.py tests/test_prompt_cache_modules.py tests/test_session_attachments_registry_and_open_tool.py tests/test_workflow_bundle_registry.py tests/test_model_residency_control_plane.py`
- Result: `51 passed, 3 skipped in 0.83s`

## Completion report

### Date

2026-05-20

### Summary

This cleanup was already present in code and tests, but the backlog file had not been moved out of `proposed/`. The
current tree still reflects the intended rule: Gateway-owned env names are ignored by Runtime implementation code, and
hosts must hand off Runtime-relevant values explicitly.

### Behavior changes

- Gateway prompt-cache env state no longer leaks into Runtime auto-keying.
- Gateway attachment-limit env state no longer changes Runtime attachment storage limits.
- Gateway flow-directory env state no longer changes Runtime workflow bundle default resolution.
- Runtime docs consistently describe Gateway-to-Runtime config translation as an explicit handoff.

### Tests

- `tests/test_prompt_cache_modules.py`
- `tests/test_session_attachments_registry_and_open_tool.py`
- `tests/test_workflow_bundle_registry.py`

### Docs

- `docs/integrations/abstractcore.md`
- `docs/faq.md`
- `docs/workflow-bundles.md`
- `docs/api.md`

### Residual risks

- Runtime still depends on hosts to translate Gateway-owned configuration into explicit Runtime inputs correctly.
- Historical backlog text referenced an earlier `abstractcore` floor; that does not affect the env-namespace cleanup
  itself, but it is part of the surrounding backlog drift.

### Backlog/code drift found

- The backlog file already said `Status: Completed` but had not been moved to `completed/`.
- The actual implementation/test coverage is stronger than the stranded proposal implied because the cleanup is now
  covered across prompt cache, attachment limits, and workflow bundle registry resolution.

### Follow-ups

- Preserve the invariant that new Gateway settings are either consumed in Gateway or translated into Runtime-owned
  inputs, never read ad hoc by Runtime from `ABSTRACTGATEWAY_*`.
