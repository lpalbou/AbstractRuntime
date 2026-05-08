# Completed: Runtime Gateway Env Namespace Cleanup

## Metadata
- Created: 2026-05-08
- Status: Completed
- Completed: 2026-05-08

## Context

AbstractRuntime should remain a lower-level durable runner. Gateway may compose Runtime and pass
run-scoped defaults, policy, and deployment choices into Runtime, but Runtime should not learn
Gateway-specific environment variable names.

This matters for the two-entry-point design:

- developers may use Runtime/Core without Gateway;
- Gateway may deploy Runtime as one component in a larger server;
- config precedence should be explicit and testable;
- lower packages should not depend on higher-package env namespaces.

## Current Code Reality

Runtime 0.4.8 carries the current install-boundary work:

- `pyproject.toml` now points Runtime AbstractCore extras at `abstractcore>=2.13.12`.
- Runtime exposes `apple`, `gpu`, `all-apple`, and `all-gpu` as explicit Core profile cascades.
- `tests/test_packaging_extras.py` verifies the Core floor and profile cascade dependencies.
- `tests/test_runtime_install_boundary.py` verifies the package root/kernel import boundary.
- `tests/test_remote_llm_client.py` verifies remote AbstractCore clients do not inherit Gateway auth
  or provider-key env vars.

Before this cleanup, Runtime still had Gateway namespace reads in implementation code:

- prompt cache auto-enable reads both `ABSTRACTRUNTIME_PROMPT_CACHE` and
  `ABSTRACTGATEWAY_PROMPT_CACHE`;
- pending-media attachment limits read `ABSTRACTGATEWAY_MAX_ATTACHMENT_BYTES`.
- workflow bundle default registry resolution reads `ABSTRACTGATEWAY_FLOWS_DIR`.

Runtime also reads Runtime-owned or generic env vars such as
`ABSTRACTRUNTIME_GLOBAL_MEMORY_RUN_ID`, `ABSTRACTRUNTIME_MAX_INLINE_BYTES`, and
`ABSTRACT_WORKSPACE_BASE_DIR`. Those are not the problem; the problem is Runtime directly reading
Gateway-owned names.

## Problem

The current behavior partially violates the desired config cascade:

1. Gateway config/env should be consumed by Gateway.
2. Gateway should pass Runtime-relevant values explicitly through run state, effect payloads,
   constructor/config arguments, or a Runtime-owned env name.
3. Runtime should not read `ABSTRACTGATEWAY_*` directly.

Leaving these reads in Runtime creates subtle deployment coupling:

- a standalone Runtime/Core process can change behavior because a Gateway env var happens to be set;
- Gateway cannot cleanly document its config precedence because Runtime also consumes some of it;
- tests currently cover auth/key leakage but not non-auth Gateway env leakage;
- future Gateway settings may be copied into Runtime ad hoc instead of using explicit handoff.

## What We Want To Do

Remove Gateway-specific env reads from Runtime and make the handoff explicit.

Prompt cache:

- keep `_runtime.prompt_cache` as the preferred run-scoped control plane;
- allow a Runtime-owned process default such as `ABSTRACTRUNTIME_PROMPT_CACHE`;
- Gateway should read `ABSTRACTGATEWAY_PROMPT_CACHE` itself and write `_runtime.prompt_cache` when
  it wants prompt cache enabled for a run/session.

Attachment/media limits:

- add an explicit Runtime-side control such as `_runtime.max_attachment_bytes`, effect payload
  `max_attachment_bytes`, or a small typed attachment policy object;
- allow a Runtime-owned process default such as `ABSTRACTRUNTIME_MAX_ATTACHMENT_BYTES` if a process
  env fallback is still needed;
- Gateway should read `ABSTRACTGATEWAY_MAX_ATTACHMENT_BYTES` itself and pass the effective limit to
  Runtime explicitly.

## Requirements

- Runtime must not read `ABSTRACTGATEWAY_*` env vars.
- Runtime may keep `ABSTRACTRUNTIME_*` env vars for Runtime-owned process defaults.
- Gateway-provided values must travel through explicit run state, effect payloads, Runtime config, or
  constructor arguments.
- Existing `_runtime.prompt_cache` behavior should remain supported.
- Add regression tests proving Gateway env vars alone do not change Runtime behavior.
- Add tests proving explicit Runtime controls still work.
- Any compatibility fallback for old Gateway env names must be temporary, documented with
  `#FALLBACK`, and preferably implemented in Gateway instead of Runtime.

## Suggested Implementation

1. Update `_maybe_inject_prompt_cache_key(...)` to remove the
   `ABSTRACTGATEWAY_PROMPT_CACHE` branch.
2. Update the pending-media attachment byte limit resolver to prefer explicit Runtime state/payload
   and then `ABSTRACTRUNTIME_MAX_ATTACHMENT_BYTES`; remove `ABSTRACTGATEWAY_MAX_ATTACHMENT_BYTES`.
3. Add focused tests in `tests/test_remote_llm_client.py` or a new effect-handler test module:
   - setting `ABSTRACTGATEWAY_PROMPT_CACHE=1` alone does not inject a prompt cache key;
   - setting `_runtime.prompt_cache=True` does inject a prompt cache key when session/provider/model
     are present;
   - setting `ABSTRACTGATEWAY_MAX_ATTACHMENT_BYTES` alone does not alter Runtime attachment limits;
   - the chosen explicit Runtime limit path does alter Runtime attachment limits.
4. Update Runtime docs that mention prompt cache or attachment limits.
5. Update the Gateway backlog/implementation to translate Gateway config/env into Runtime handoff
   values instead of relying on Runtime to read Gateway env names.

## Implementation Notes

Runtime 0.4.7 applies this cleanup directly:

- `_maybe_inject_prompt_cache_key(...)` no longer reads `ABSTRACTGATEWAY_PROMPT_CACHE`.
- Prompt cache auto-keying still supports explicit `LLM_CALL.params.prompt_cache_key`,
  `_runtime.prompt_cache`, and Runtime-owned `ABSTRACTRUNTIME_PROMPT_CACHE`.
- Read-file session attachment registration now resolves byte limits from
  `TOOL_CALLS.payload.max_attachment_bytes`, `_runtime.max_attachment_bytes`,
  `ABSTRACTRUNTIME_MAX_ATTACHMENT_BYTES`, then the default 25 MiB.
- `default_workflow_bundles_dir()` no longer reads `ABSTRACTGATEWAY_FLOWS_DIR`; Gateway should pass
  `bundles_dir` explicitly or translate to `ABSTRACTFRAMEWORK_WORKFLOWS_DIR`.
- Regression tests cover prompt-cache env isolation, attachment-limit env isolation, explicit
  Runtime namespace and effect payload attachment limits, and workflow bundle registry env
  isolation.

## Scope

Included:

- Runtime AbstractCore effect-handler config lookup cleanup.
- Runtime tests for env namespace isolation.
- Runtime docs/backlog notes required to preserve the boundary.

Excluded:

- Gateway config implementation.
- Gateway package extras.
- Gateway server route behavior.
- Provider/model/auth decisions, which are already separately covered by existing tests.

## Promotion Criteria

Promote this before declaring Runtime fully clean for a production Gateway release if either of
these is true:

- Gateway still relies on `ABSTRACTGATEWAY_PROMPT_CACHE` or `ABSTRACTGATEWAY_MAX_ATTACHMENT_BYTES`
  being read by Runtime;
- Runtime is being released as the official Gateway-aligned baseline.

If Gateway is updated first to pass explicit Runtime state and no production deployment relies on the
old env reads, this can be implemented as a small cleanup immediately after Gateway config lands.

## Validation

- [x] `python -m pytest tests/test_runtime_install_boundary.py tests/test_packaging_extras.py`
- [x] focused prompt-cache/env isolation tests;
- [x] focused attachment-limit/env isolation tests;
- [x] `rg -n "ABSTRACTGATEWAY" src tests docs` should show no Runtime implementation reads, except
  tests that assert the env name is ignored and docs/backlog references explaining the cleanup.

## Guidance For The Implementing Agent

Re-check current Gateway behavior before deleting compatibility assumptions. The clean target is not
"no env vars"; it is "Runtime owns Runtime env vars, Gateway owns Gateway env vars, and Gateway
passes Runtime-relevant values explicitly."
