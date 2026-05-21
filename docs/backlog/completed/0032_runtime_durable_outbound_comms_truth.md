# 0032 Runtime Durable Outbound Comms Truth

## Metadata
- Created: 2026-05-21
- Status: Completed
- Completed: 2026-05-21
- Origin: follow-up correction to `completed/0030_runtime_host_facades_for_comms_telegram_and_tool_specs.md`

## Goal

Close the remaining durability gap left after `0030`: outbound comms sends that
belong to a run must execute through Runtime so the request and outcome become
ledgered child-run truth instead of host-local side effects with no Runtime
record.

## What Shipped

- Extended `AbstractCoreRunFacade` with a generic durable child-run helper:
  - `execute_tool_calls(...)`
- Extended `AbstractCoreRunFacade` with a public durable resume helper:
  - `resume_tool_calls(...)`
- Extended `AbstractCoreRunFacade` with durable outbound comms helpers:
  - `send_email(...)`
  - `send_telegram_message(...)`
- Kept the existing host-local read/bootstrap wrappers intact:
  - host-facade email list/read helpers stay operator-scoped
  - Telegram TDLib bootstrap/global-client helpers stay host-scoped
- Preserved the replay rule:
  - Runtime records the send request and the send outcome
  - replay should show the recorded result
  - replay should not resend the external email or Telegram message

## Current Code Pointers

- `src/abstractruntime/integrations/abstractcore/run_facade.py`
- `tests/test_abstractcore_run_facade.py`
- `docs/integrations/abstractcore.md`
- `docs/api.md`
- `docs/faq.md`
- `docs/tools-comms.md`

## ADR Status

- Governing ADRs: none added
- ADR impact: none; this extends the already-accepted Runtime-owned durable
  child-run pattern from the run facade rather than introducing a new durable
  policy

## Validation

Focused validation:

- `pytest -q tests/test_abstractcore_run_facade.py tests/test_abstractcore_host_facade.py tests/test_abstractcore_telegram_facade.py`
- Result: `32 passed in 0.19s`

Aggregate validation:

- `pytest -q tests/test_abstractcore_run_facade.py tests/test_abstractcore_host_facade.py tests/test_abstractcore_telegram_facade.py tests/test_prompt_cache_modules.py tests/test_model_residency_control_plane.py tests/test_packaging_extras.py`
- Result: `61 passed in 0.21s`

Syntax/doc sanity:

- `python -m compileall src/abstractruntime/integrations/abstractcore`
- `mkdocs build -q --site-dir /tmp/abstractruntime-0032-docs`

## Completion Report

### Date

2026-05-21

### Summary

`0030` solved the package-boundary problem for comms and Telegram, but it left
direct host-wrapper sends nondurable. This item fixes that by making outbound
email and Telegram sends first-class Runtime child runs through the existing
`TOOL_CALLS` path and by giving hosts a public `resume_tool_calls(...)` helper
for approval-gated or passthrough child runs.

The result is narrower than “all comms go through one facade” and cleaner:

- read/bootstrap helpers remain host-local where that still makes sense
- outbound sends that belong to a run are now Runtime-authored truth

### Behavior Changes

- Hosts can now use `get_abstractcore_run_facade(runtime).send_email(...)` for
  a ledgered child run instead of performing an unrecorded host-local send.
- Hosts can now use
  `get_abstractcore_run_facade(runtime).send_telegram_message(...)` for the
  same durable behavior.
- Hosts can now use
  `get_abstractcore_run_facade(runtime).resume_tool_calls(...)` to continue a
  waiting comms/tool child run through the same public Runtime boundary.
- These durable send helpers use the configured Runtime tool executor:
  - local/hybrid runtimes usually execute immediately
  - approval-gated or passthrough executors still produce a durable wait that
    Runtime can resume later
- Direct host-facade `send_email(...)` and direct
  `telegram_facade.send_telegram_message(...)` still exist, but they are now
  explicitly operator/local-helper paths rather than the preferred run-owned
  send path

### Review Notes

- No new Core seam was needed. Runtime already had the correct durable external
  side-effect mechanism in `EffectType.TOOL_CALLS`.
- The important design correction was to put outbound sends on the existing run
  facade rather than inventing another host wrapper or another execution model.
- The last practical gap was resumability for waiting tool child runs. That is
  now part of the same public run-facade contract instead of a test-only
  internal workflow reconstruction trick.
- This keeps replay semantics honest: recorded outcomes are replayed as data,
  not re-executed as real external sends.

### Tests

- `pytest -q tests/test_abstractcore_run_facade.py tests/test_abstractcore_host_facade.py tests/test_abstractcore_telegram_facade.py`
  - Result: `32 passed in 0.19s`
- `pytest -q tests/test_abstractcore_run_facade.py tests/test_abstractcore_host_facade.py tests/test_abstractcore_telegram_facade.py tests/test_prompt_cache_modules.py tests/test_model_residency_control_plane.py tests/test_packaging_extras.py`
  - Result: `61 passed in 0.21s`
- `python -m compileall src/abstractruntime/integrations/abstractcore`
  - Result: clean
- `mkdocs build -q --site-dir /tmp/abstractruntime-0032-docs`
  - Result: docs build passed; upstream Material for MkDocs emitted a
    non-blocking roadmap warning

### Docs

- `README.md`
- `docs/integrations/abstractcore.md`
- `docs/api.md`
- `docs/faq.md`
- `docs/tools-comms.md`
- `llms.txt`
- `llms-full.txt`

### Residual Risks

- Remote runtimes still default to passthrough tools, so durable comms sends may
  enter a wait instead of executing immediately unless the host configures an
  executing or approval-resumable tool executor.
- Operator-only direct sends remain possible through the host-local wrappers.
  That is intentional for maintenance/bootstrap flows, but hosts should use the
  durable run facade when the send belongs to workflow truth.

### Practical Proof

- Immediate durable Telegram send smoke:
  - child run status: `completed`
  - ledger effect records: `3`
  - output: `{"success": true, "transport": "bot_api", "message_ids": [42]}`
- Approval-gated durable email smoke:
  - initial child run status: `waiting`
  - wait reason: `user`
  - wait key prefix: `tool_approval`
  - resumed child run status: `completed`
  - output: `{"success": true, "message_id": "<smoke-1>", "account": "ops"}`

### Follow-Ups

- Keep `0030` as the package-boundary cleanup record and this item as the
  durability correction record; do not collapse them into one historical file.
- Keep `0031_runtime_tool_spec_adapters_for_gateway_and_mcp.md` separate; this
  change does not alter the tool-spec follow-up.
