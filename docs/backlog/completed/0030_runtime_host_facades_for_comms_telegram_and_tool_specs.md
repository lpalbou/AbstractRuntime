# 0030 Runtime Host Facades For Comms And Telegram, With Tool-Spec Follow-Up Split Out

## Metadata
- Created: 2026-05-21
- Status: Completed
- Completed: 2026-05-21

## ADR status
- Governing ADRs: None
- ADR impact: None; this extends the existing public host-facade pattern rather than creating a new architecture rule.

## Context

Runtime already owns the public Gateway-facing integration boundary for the
main AbstractCore-backed surfaces:

- `AbstractCoreHostFacade` for operator/control-plane helpers
- `AbstractCoreDiscoveryFacade` for provider/model/media catalogs
- `AbstractCoreRunFacade` for run-scoped durable media execution

Gateway now uses those facades for prompt-cache control, durable blocs, model
residency, discovery, and run-scoped voice/STT/image work. The remaining
direct `abstractcore` imports in Gateway are concentrated in comms/Telegram
helpers and a smaller tool-spec/testing tail.

## Current code reality

Inspected on 2026-05-21:

- `src/abstractruntime/integrations/abstractcore/host_facade.py`
  - exposes prompt-cache, bloc, and model-residency helpers
  - does not expose email/comms helpers
- `src/abstractruntime/integrations/abstractcore/default_tools.py`
  - imports `abstractcore.tools.common_tools`, `comms_tools`, and `telegram_tools`
  - derives JSON-safe tool specs via `abstractcore.tools.core.ToolDefinition`
- `src/abstractruntime/integrations/abstractcore/mcp_worker.py`
  - converts callables to MCP entries through `ToolDefinition.from_function(...)`
- `src/abstractruntime/integrations/abstractcore/tool_executor.py`
  - still constructs `abstractcore.tools.core.ToolCall`
- `../abstractgateway/src/abstractgateway/routes/gateway.py`
  - still imports `abstractcore.tools.comms_tools` directly for email helper routes
- `../abstractgateway/src/abstractgateway/cli.py`
  - still imports `abstractcore.tools.telegram_tdlib` directly for `telegram-auth`
- `../abstractgateway/src/abstractgateway/integrations/telegram_bridge.py`
  - still imports `abstractcore.tools.telegram_tdlib` global-client helpers directly

Runtime therefore has the right facade pattern already, but the remaining
host/operator surfaces have not been moved onto it. The strongest current
Gateway production pressure is comms/email and Telegram bootstrap/global-client
access; tool-spec coupling is lower-pressure follow-up work.

## Problem

Gateway cannot fully stop importing `abstractcore` directly until Runtime owns
two small public surfaces in phase 1:

1. comms/email host helpers
2. Telegram bootstrap/global-client wrappers, plus a thin send helper for notifier parity

The missing pieces are not durable-run problems. They are host/operator
integration surfaces, so they belong next to the existing host-facade pattern.
Tool-spec helpers remain useful, but they are not the strongest current
Gateway blocker and should not hold up the narrower phase-1 cleanup.

## What we want to do

Extend Runtime's public host integration layer so Gateway can depend on
Runtime alone for the remaining comms/Telegram edges, while tracking
tool-spec normalization as a separate follow-up.

## Why

- keeps Gateway source aligned with the Runtime-owned boundary
- reuses the existing facade pattern instead of inventing another Gateway/Core seam
- gives future hosts one place to look for operator-scoped helper contracts
- lets Runtime normalize local/remote Core topology for these paths too when needed later
- avoids over-scoping a release-blocking boundary cleanup around a lower-pressure
  tool-spec concern

## Requirements

- Add public host-facing email helpers for:
  - list accounts
  - list messages
  - read message
  - send message
- Add a Runtime-owned Telegram wrapper surface for:
  - one-shot TDLib auth bootstrap from env
  - a stable "not available" error
  - access to the global TDLib client lifecycle used by the Gateway bridge
  - stop/cleanup of the global client
  - `send_telegram_message(...)` for notifier parity
- Keep these host helpers nondurable. They must not write Runtime run history by themselves.
- Allow Runtime to wrap existing public Core tool modules in phase 1; do not block on a larger Core refactor.
- Do not require a new AbstractCore runtime-facing backend seam for phase 1.

## Suggested implementation

1. Extend `AbstractCoreHostFacade` with email/comms methods:
   - `list_email_accounts(...)`
   - `list_emails(...)`
   - `read_email(...)`
   - `send_email(...)`
2. Add a small Runtime Telegram helper module, for example under
   `abstractruntime.integrations.abstractcore.telegram_facade`, with:
   - `TelegramTdlibNotAvailable`
   - `bootstrap_telegram_auth_from_env(...)`
   - `get_global_telegram_client(start=False)`
   - `stop_global_telegram_client()`
   - `send_telegram_message(...)`
3. Keep Runtime wrappers thin over current public Core helpers first:
   - `abstractcore.tools.comms_tools`
   - `abstractcore.tools.telegram_tdlib`
   - `abstractcore.tools.telegram_tools`
4. Keep full abstractagent/tool-type decoupling explicitly out of scope here.

## Scope

- Runtime host facade extension for email/comms
- Runtime Telegram wrappers for Gateway CLI/bridge/notifier adoption
- tests proving those surfaces are public, stable, and JSON-safe

## Non-goals

- Do not move Gateway workspace/file policy into Runtime.
- Do not redesign Core tool execution or the durable `TOOL_CALLS` effect.
- Do not block this on a full replacement of `abstractcore.tools.ToolCall` inside AbstractAgent.
- Do not block this phase on Runtime-owned tool-spec adapters; track those separately.
- Do not introduce new Core HTTP endpoints just to satisfy Gateway.

## Dependencies and related tasks

- `../completed/019_runtime_host_facade_for_core_operator_surfaces.md`
- `../completed/026_runtime_host_discovery_facade_for_core_catalogs.md`
- `../completed/027_runtime_durable_bloc_prompt_cache_facade.md`
- `../completed/028_runtime_bloc_kv_lifecycle_and_pruning.md`
- `../proposed/0031_runtime_tool_spec_adapters_for_gateway_and_mcp.md`
- `../../../../abstractgateway/docs/backlog/planned/0050_gateway_runtime_boundary_cleanup_for_workspace_comms_and_telegram.md`
- `../../../../abstractcore/docs/backlog/proposed/0796_runtime_facing_comms_and_telegram_backend_surface.md`

## Expected outcomes

- Runtime exposes the public surfaces Gateway needs in order to stop importing
  `abstractcore.tools.comms_tools` directly.
- Runtime exposes the public surfaces Gateway needs in order to stop importing
  `abstractcore.tools.telegram_tdlib` and `telegram_tools` directly.
- Runtime remains the only lower-level dependency that Gateway needs for these
  operator/tooling paths once Gateway adopts the new wrappers.
- Runtime phase 1 proves whether a later Runtime-owned tool-spec surface is
  still needed; that work stays split into `0031`.

## Validation

- focused Runtime tests for the new host-facade email methods
- focused Runtime tests for Telegram wrapper happy/unavailable paths
- focused Runtime wiring proof that the host email helpers still work when the
  runtime is local/remote/hybrid because they are host-local wrappers, not
  remote Core server routes
- aggregate Runtime regression covering the adjacent host-facade behavior

## Progress checklist
- [x] Extend `AbstractCoreHostFacade` with email/comms helpers.
- [x] Add Runtime-owned Telegram wrapper helpers for bootstrap, send, and global-client lifecycle.
- [x] Validate the Runtime surfaces and re-run focused Runtime boundary tests.

## Guidance for the implementing agent

Prefer thin public wrappers over current Core functionality first. If phase 1
can unblock Gateway without new Core changes, do that. Keep Core item `0796`
proposed unless Runtime adoption proves the existing public tool modules are
too unstable or too duplicative. Keep Runtime-owned tool-spec adapters as a
separate follow-up rather than blocking this narrower comms/Telegram item.

## Completion report

### Date

2026-05-21

### Summary

Runtime now exposes the phase-1 public comms and Telegram wrappers that Gateway
needed for its remaining direct AbstractCore imports:

- host-facade email helpers on `AbstractCoreHostFacade`
- a separate Runtime-owned `telegram_facade` module for TDLib bootstrap,
  process-global client lifecycle, and send parity

The implementation stayed intentionally narrow. It did not add remote server
routes, and it did not mix the lower-pressure tool-spec cleanup back into this
item.

### What shipped

- Extended `AbstractCoreHostFacade` with:
  - `list_email_accounts(...)`
  - `list_emails(...)`
  - `read_email(...)`
  - `send_email(...)`
- Added `abstractruntime.integrations.abstractcore.telegram_facade` with:
  - `TelegramTdlibNotAvailable`
  - `bootstrap_telegram_auth_from_env(...)`
  - `get_global_telegram_client(...)`
  - `stop_global_telegram_client()`
  - `send_telegram_message(...)`
- Exported the new Telegram wrappers from
  `abstractruntime.integrations.abstractcore`.
- Updated docs and LLM indexes so the public boundary now points hosts at
  Runtime instead of direct `abstractcore.tools.*` imports.

### Current code pointers

- `src/abstractruntime/integrations/abstractcore/host_facade.py`
- `src/abstractruntime/integrations/abstractcore/telegram_facade.py`
- `src/abstractruntime/integrations/abstractcore/__init__.py`
- `tests/test_abstractcore_host_facade.py`
- `tests/test_abstractcore_telegram_facade.py`
- `docs/integrations/abstractcore.md`
- `docs/tools-comms.md`
- `docs/api.md`
- `docs/faq.md`

### Behavior changes

- Hosts can now use Runtime for operator-scoped email account discovery, email
  listing/reading, and email sending instead of importing
  `abstractcore.tools.comms_tools` directly.
- Hosts can now use Runtime for Telegram TDLib bootstrap, process-global TDLib
  client lifecycle, and notifier-style `send_telegram_message(...)` instead of
  importing `abstractcore.tools.telegram_tdlib` or `telegram_tools` directly.
- The email and Telegram phase-1 wrappers are explicitly **host-local**:
  - they do not proxy through a remote AbstractCore server
  - they do not create Runtime run history on their own
  - the Telegram global client remains process-scoped rather than
    runtime-instance scoped

### Review notes

- No new ADR was added. This is an extension of the already-established public
  Runtime facade pattern rather than a new durable architecture rule.
- The two focused subagent reviews converged on the shipped shape:
  - email belongs on the existing host facade
  - Telegram lifecycle belongs in a separate module because it is process-global
    host state, not normal runtime-instance state
- Tool-spec adapters remain explicitly deferred to `0031` rather than being
  silently dropped.

### Tests

- `pytest -q tests/test_abstractcore_host_facade.py tests/test_abstractcore_telegram_facade.py`
  - Result: `24 passed`
- `pytest -q tests/test_abstractcore_host_facade.py tests/test_abstractcore_telegram_facade.py tests/test_prompt_cache_modules.py tests/test_model_residency_control_plane.py tests/test_packaging_extras.py`
  - Result: `53 passed in 0.29s`
- `python -m compileall src/abstractruntime/integrations/abstractcore`
  - Result: clean
- `mkdocs build -q --site-dir /tmp/abstractruntime-0030-docs`
  - Result: docs build passed; upstream Material for MkDocs emitted a
    non-blocking roadmap warning
- `git diff --check`
  - Result: clean

### Docs

- `README.md`
- `docs/integrations/abstractcore.md`
- `docs/tools-comms.md`
- `docs/api.md`
- `docs/faq.md`
- `llms.txt`
- `llms-full.txt`

### Residual risks

- These wrappers intentionally do not normalize remote/hybrid comms/Telegram
  behavior through a remote Core server. If a later product requirement wants a
  remote backend seam for those paths, that should be a separate decision.
- Gateway adoption is still tracked separately in
  `../../../../abstractgateway/docs/backlog/planned/0050_gateway_runtime_boundary_cleanup_for_workspace_comms_and_telegram.md`.
- Gateway's backlog item `0050` still links to the old planned path for this
  Runtime item. That stale sibling-package link was detected during closure
  hygiene but was not edited here because Runtime write scope stayed confined to
  this repository.

### Follow-ups

- Durable outbound comms sends were corrected separately in
  `completed/0032_runtime_durable_outbound_comms_truth.md` so `0030` can stay
  an honest record of the narrower package-boundary cleanup it originally
  shipped.
- Keep Runtime-owned tool-spec adapters in
  `../proposed/0031_runtime_tool_spec_adapters_for_gateway_and_mcp.md`.
- Keep the optional Core seam idea in
  `../../../../abstractcore/docs/backlog/proposed/0796_runtime_facing_comms_and_telegram_backend_surface.md`
  proposed unless Runtime or Gateway adoption proves the current public Core
  wrappers are insufficient.
