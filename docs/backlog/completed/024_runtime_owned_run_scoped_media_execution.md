# 024 Runtime-Owned Run-Scoped Media Execution

## Metadata
- Created: 2026-05-20
- Status: Completed
- Completed: 2026-05-20
- Origin: added during ADR/backlog review of Runtime-owned media boundaries

## Goal

Give hosts a public Runtime-owned path for run-scoped image, TTS, and STT work so Runtime, not host route code,
authors the durable ledger and artifact truth.

## What Shipped

- Added `AbstractCoreRunFacade` and `get_abstractcore_run_facade(runtime)`.
- Added durable child-run helpers:
  - `execute_llm_call(...)`
  - `generate_image(...)`
  - `generate_voice(...)`
  - `transcribe_audio(...)`
- Child runs inherit the parent run’s `actor_id`, `session_id`, `parent_run_id`, and `_runtime` namespace.
- Public docs now distinguish the operator control facade from the durable run facade.

## Current Code Pointers

- `src/abstractruntime/integrations/abstractcore/run_facade.py`
- `src/abstractruntime/integrations/abstractcore/__init__.py`
- `tests/test_abstractcore_run_facade.py`
- `docs/integrations/abstractcore.md`
- `docs/api.md`
- `docs/faq.md`
- `docs/adr/0004_runtime_owns_run_scoped_media_execution_truth.md`

## Validation

Focused validation run on 2026-05-20:

- `pytest -q tests/test_abstractcore_run_facade.py tests/test_model_residency_control_plane.py tests/test_multimodal_abstractcore_integration.py tests/test_abstractcore_host_facade.py`
- Result: `61 passed in 0.27s`

## Completion report

### Date

2026-05-20

### Summary

Runtime now exposes a public durable facade for host-triggered media work on existing runs. The helper creates child
runs and executes the real `LLM_CALL` through Runtime, which preserves ledger truth, artifact ownership, and replayable
state without requiring Gateway or another host to synthesize media history afterward.

### Behavior changes

- Hosts can request durable media work through Runtime without reaching into private internals.
- Each host-triggered media action becomes an ordinary Runtime child run with its own output and ledger.
- Parent run identity and runtime-scoped media defaults are inherited automatically.

### Tests

- `tests/test_abstractcore_run_facade.py`

### Docs

- `docs/integrations/abstractcore.md`
- `docs/api.md`
- `docs/faq.md`
- `docs/adr/README.md`

### Residual risks

- This repo now provides the Runtime-side migration target, but external hosts still need to adopt it.
- The helper executes synchronously by ticking the child run immediately; if a future media path introduces waiting
  semantics, the facade may need an explicit async/start-only mode.

### Follow-ups

- Migrate Gateway/other hosts to call `get_abstractcore_run_facade(runtime)` for run-scoped media endpoints.
