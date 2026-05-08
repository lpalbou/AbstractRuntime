# Proposed: Runtime Gateway Install Boundary

## Metadata
- Created: 2026-05-08
- Status: Completed
- Completed: 2026-05-08

## Context

AbstractRuntime is the durable graph runner. It persists runs, effects, waits, ledgers, artifacts,
snapshots, and resume behavior. It should remain stable and dependency-light even when Gateway
deployments pull in Core, agents, media, memory, tools, and provider SDKs.

## Current Code Reality

- Runtime base depends on `abstractsemantics>=0.0.3`.
- AbstractCore integration is explicit through `abstractruntime[abstractcore]`.
- Multimodal helper dependencies are explicit through `abstractruntime[multimodal]`.
- Runtime does not directly depend on AbstractMemory; memory effects are integration/host-wired.
- Runtime config contains run-level defaults and limits, not deployment auth or provider secrets.
- As of the AbstractCore `2.13.12` release, Core exposes the canonical install-profile aliases
  `abstractcore[apple]`, `abstractcore[gpu]`, `abstractcore[all-apple]`, and
  `abstractcore[all-gpu]`, plus the released Voice/Vision catalog-capable floors
  `abstractvoice>=0.9.2` and `abstractvision>=0.3.3`.
- Runtime's package metadata now points its AbstractCore extras at `abstractcore>=2.13.12`
  for Gateway deployment alignment.
- Runtime exposes pass-through profile extras `apple`, `gpu`, `all-apple`, and `all-gpu` so
  Gateway/root aggregate installs can cascade through Core without Runtime owning hardware engines.

## Problem

The desire for cascading Gateway installs can accidentally push Core/Vision/Voice/Memory/local
engine dependencies down into Runtime. That would weaken the clean boundary:

- Runtime should be able to execute non-LLM workflows without Core.
- Runtime should not own provider configuration.
- Runtime should not import Vision, Voice, Music, or Memory in its kernel.
- Runtime should persist artifact refs and JSON-safe results, not model/provider clients.

## Proposed Direction

Keep Runtime base minimal.

Profile guidance:

- `abstractruntime`: durable kernel plus Semantics schema refs.
- `abstractruntime[abstractcore]`: LLM/tool integration, with a floor of
  `abstractcore>=2.13.12` so Gateway receives the current Core server-auth, provider-key,
  generated-media, and capability-catalog contracts.
- `abstractruntime[multimodal]`: Core media/capability helpers for generated images, voice/audio,
  and media inputs, also with `abstractcore>=2.13.12`.
- `abstractruntime[apple]`, `abstractruntime[gpu]`, `abstractruntime[all-apple]`, and
  `abstractruntime[all-gpu]`: pass-through Core profile cascades. Runtime still owns no
  hardware-specific engine.

Gateway should compose Runtime with the selected Core/capability/memory profile. The cascade belongs
to Gateway, not Runtime.

## Gateway Configuration Handoff Rules

Runtime should stay explicit about configuration handoff:

- Gateway may pass run defaults in JSON-safe run state, such as `run.vars["_runtime"]["provider"]`
  and `run.vars["_runtime"]["model"]`.
- Gateway may pass effect-level overrides in LLM/media effect payloads when workflow pins or request
  values require them.
- Gateway may construct local/remote/hybrid AbstractCore clients with explicit Core server URL,
  Core server auth headers, provider/model defaults, tool executor, retry policy, and artifact
  store.
- Runtime should not read `ABSTRACTGATEWAY_*` env vars directly; Gateway-owned config must be
  translated into explicit Runtime inputs or Runtime-owned env names.
- Runtime should not reinterpret Gateway auth tokens as Core server auth tokens or provider keys.
- Runtime should keep persisted run state JSON-safe: provider names, model ids, policy flags, and
  artifact refs are fine; provider clients, auth objects, downloaded model handles, and server
  sessions are not.

## Implementation Notes

Runtime was touched only for release-alignment and boundary validation.

What changed in Runtime 0.4.7:

- `abstractruntime[abstractcore]`, `abstractruntime[multimodal]`, and
  `abstractruntime[mcp-worker]` now require `abstractcore>=2.13.11`.
- Runtime still exposes no `apple`, `gpu`, `all-apple`, or `all-gpu` extras.
- The AbstractCore integration version guard now fails fast on stale Core installs older than
  2.13.11.
- Runtime no longer reads Gateway-owned environment variables for prompt-cache defaults,
  attachment registration limits, or workflow bundle registry paths.
- Tests now cover the optional-stack import boundary, package metadata floor, missing hardware
  profile extras, Gateway env namespace isolation, and the rule that Gateway auth/provider-key
  environment variables are not inherited by remote AbstractCore clients.

Related package guidance remains:

- Gateway should keep Runtime base as the dependency for minimal installs.
- Gateway server profiles may choose `AbstractRuntime[multimodal]`.
- Gateway Apple/GPU profiles should cascade through Core profile names such as
  `abstractcore[all-apple]` or `abstractcore[all-gpu]`.
- Root `abstractframework` should update Runtime version floors to match the current Gateway/Core
  integration baseline.
- AbstractAgent should tighten its Runtime dependency floor.

What changed in Runtime 0.4.8:

- `abstractruntime[abstractcore]`, `abstractruntime[multimodal]`, and
  `abstractruntime[mcp-worker]` now require `abstractcore>=2.13.12`.
- `abstractruntime[apple]` and `abstractruntime[gpu]` delegate to matching Core local-engine
  aliases.
- `abstractruntime[all-apple]` and `abstractruntime[all-gpu]` delegate to matching Core aggregate
  profiles.

## Promotion Criteria

Promoted because the Runtime release needed package metadata alignment, clearer install-boundary
docs, and regression tests for the Gateway handoff.

## Validation

- [x] Subprocess import test for `abstractruntime` with Core/Vision/Voice/Memory/Music blocked.
- [x] Fresh venv import test for `abstractruntime` without AbstractCore installed.
- [x] Fresh venv import test for `abstractruntime[abstractcore]`.
- [x] Runtime LLM/media integration tests gated behind extras.
- [x] Static import checks that Runtime kernel modules do not import Core, Vision, Voice, Memory, or
  Music.
- [x] Package metadata test proving Runtime profile extras are explicit Core cascades.
- [x] Package metadata test proving `abstractruntime[abstractcore]` and `abstractruntime[multimodal]`
  depend on `abstractcore>=2.13.12`.
