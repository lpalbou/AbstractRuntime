# Proposed: Runtime local-admin prompt-cache save/load

## Metadata
- Created: 2026-05-20
- Status: Proposed
- Completed: N/A

## Context

Runtime now exposes public host-facing surfaces for:

- prompt-cache control and model residency through `AbstractCoreHostFacade`
- run-scoped durable media execution through `AbstractCoreRunFacade`
- discovery/catalog snapshot queries through `AbstractCoreDiscoveryFacade`

One important prompt-cache feature still has no clean Runtime boundary:

- save a local provider prompt-cache snapshot
- load a local provider prompt-cache snapshot

Today that behavior still exists as a Gateway-local hack over private Runtime and
provider state. We need the feature, but not in that form.

## Problem

The current shape is wrong for three reasons:

- Gateway reaches through `runtime._abstractcore_llm_client` instead of using a public Runtime API.
- Gateway needs provider-instance access and provider-private cache fields to make it work.
- The feature is really local provider admin behavior, not a generic provider-neutral prompt-cache contract.

## What we want to do

Add an explicit Runtime-owned local-admin facade for prompt-cache save/load.

## Desired shape

- Keep it Runtime-owned and public, so hosts ask Runtime.
- Keep the first version local-only and honest about that.
- Delegate to public Core provider methods such as `prompt_cache_save(...)` and
  `prompt_cache_load(...)` where available.
- Return structured capability/error payloads in the same spirit as the existing
  host control facade.

## Non-goals

- Do not expose provider-private cache internals.
- Do not promise remote/server save/load until AbstractCore Server has a real
  public contract for it.
- Do not turn save/load into a durable workflow effect by default. This is an
  operator/local-admin control surface unless a later design explicitly records
  it as an audit snapshot.

## Open questions

- Should this extend `AbstractCoreHostFacade`, or should Runtime expose a
  separate local-admin facade to keep the main host-control contract narrower?
- What file/path policy should govern local save/load targets?
- Should Runtime record save/load requests as operator audit snapshots even if
  it does not replay them by re-executing the filesystem mutation?

## Dependencies and related work

- `src/abstractruntime/integrations/abstractcore/host_facade.py`
- `src/abstractruntime/integrations/abstractcore/llm_client.py`
- `../abstractcore/abstractcore/providers/base.py`
- `../abstractgateway/docs/backlog/proposed/2026-05-20_gateway_prompt_cache_save_load_via_runtime.md`

## Why it is proposed, not planned

- The public API shape still needs a design decision.
- The filesystem/path policy needs to be nailed down first.
- Gateway can keep the current behavior temporarily, but it should not be
  treated as an acceptable end state.
