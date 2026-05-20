# Planned: Runtime host facades for comms and Telegram, with tool-spec follow-up split out

## Metadata
- Created: 2026-05-21
- Status: Planned
- Completed: N/A

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

- Gateway can stop importing `abstractcore.tools.comms_tools` directly.
- Gateway can stop importing `abstractcore.tools.telegram_tdlib` and `telegram_tools` directly.
- Runtime remains the only lower-level dependency that Gateway imports for these operator/tooling paths.
- Runtime phase 1 proves whether a later Runtime-owned tool-spec surface is still needed.

## Validation

- focused Runtime tests for new host-facade email methods
- focused Runtime tests for Telegram wrapper happy/unavailable paths
- Gateway adoption tests proving direct `abstractcore` imports are removed from those paths

## Progress checklist
- [ ] Extend `AbstractCoreHostFacade` with email/comms helpers.
- [ ] Add Runtime-owned Telegram wrapper helpers for bootstrap, send, and global-client lifecycle.
- [ ] Validate Gateway adoption and re-run focused Runtime/Gateway boundary tests.

## Guidance for the implementing agent

Prefer thin public wrappers over current Core functionality first. If phase 1
can unblock Gateway without new Core changes, do that. Keep Core item `0796`
proposed unless Runtime adoption proves the existing public tool modules are
too unstable or too duplicative. Keep Runtime-owned tool-spec adapters as a
separate follow-up rather than blocking this narrower comms/Telegram item.
