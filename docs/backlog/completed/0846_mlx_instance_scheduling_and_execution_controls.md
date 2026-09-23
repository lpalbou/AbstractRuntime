# 0846-abstractruntime: [BUG] Preserve instance scheduling and execution controls at Core boundaries

> Created: 2026-09-20
> Status: Completed
> Completed: 2026-09-20
> Type: bug
> Priority: P1
> Labels: native-inference, mlx, reasoning, transport

## Summary

Repair Runtime's integration with the existing MLX provider: consult the instance's
safe concurrency capability and preserve speculation/reasoning controls and actual
outcome metadata through the local and remote Core paths.

## Current code reality

Audited 2026-09-20: `llm_client.py` chooses a per-model lock from provider name before
constructing the instance. That protects legacy MLX but also serializes instances
with an internal scheduler. Remote requests forward only string `thinking` values,
drop boolean on/off, omit `speculation`, and discard Core's execution envelope from
normalized metadata. Native keyed-cache requests require full-history message shape;
Runtime's old prepare-modules predicate does not identify that lane.

## Scope and non-goals

Use the existing exact-True instance protocol, preserving conservative locking for
legacy/unknown implementations and iterator consumption. Carry bool/string thinking
and bool/dict speculation without altering caller data or false/absent semantics.
Keep request-local actual telemetry, safe full-history keyed-cache handling and
current provider identity. No new provider, broad capability schema/UI, HF/GGUF
MTP driver, automatic eviction or modifications to the operator's live stack.

## Acceptance criteria

- [x] Scheduled instances admit overlapping calls; legacy instances remain serialized through stream completion.
- [x] Capability changes/unavailable probes cannot silently authorize unsafe overlap.
- [x] Thinking strings and booleans, speculation off/depth/strictness, and absence round-trip correctly.
- [x] Local/remote normalized results retain actual execution/speculation metadata and Runtime-owned diagnostics.
- [x] Runtime-derived keyed-cache calls use full-history shape without changing Ollama behavior.
- [x] Independent architecture and integrator reviews, hermetic regressions and bounded native integration evidence are recorded.

## Testing

Run `python -m pytest tests/test_remote_llm_client.py tests/test_model_residency_lock_and_context_estimate.py`
plus new focused overlap, transport, cache-shape and metadata tests. Use isolated
configuration and 60–120 second cooldowns between real GPU test blocks.

## Dependencies and ADR status

Related Core native runtime and Core follow-ups 0848–0853. ADR impact: None; repair
existing provider/transport contracts without changing ownership or public vocabulary.
Expected outcome: existing parameters reach the existing provider, and upstream
safety wrappers no longer hide safe scheduler concurrency.

## Completion report

Completed 2026-09-20 in the workspace, not published or deployed. Original path:
`planned/0846_mlx_instance_scheduling_and_execution_controls.md`; final path:
`completed/0846_mlx_instance_scheduling_and_execution_controls.md`.

Implementation: `llm_client.py` probes the existing instance protocol per call,
retains legacy full-stream locking, transports bool thinking and bool/dict
speculation, preserves allowed actual outcomes and recognizes native keyed
full-history requests. Core's server/endpoint outcome envelope no longer depends
on batching. Independent review found and closed a metadata-only SSE omission.

Evidence:

- 47 new hermetic regressions in `tests/test_native_mlx_application_wiring.py`.
- Combined affected Runtime gate: **237 passed** (including residency locks).
- Core regression gate: **616 passed, 2 opt-in skips**; includes six added HTTP
  envelope cases. These counts overlap the reviewer's independent subset.
- Gated `tests/test_native_mlx_execution_controls_live.py`: four final cases,
  27B/Flash × MTP off/depth 3, each with four concurrent real-model requests
  (two local Runtime, one streaming; two remote Runtime via actual isolated Core
  ASGI route) and one separate keyed request: **20 successful requests**.
  Every concurrent result reported peak batch 4; MTP actual use/depth was correct.
- GPU blocks separated by at least 60 seconds; processes unloaded their models.
- Three architecture charters selected targeted existing-contract integration;
  dedicated independent integrator verdict **Approved**. No fallback swarm.
- Architecture, review and raw live JSON receipts:
  a local scratch directory (`ARCHITECTURE.md`,
  `REVIEW.md`, `27b-off-v2/`, `27b-d3/`, `flash-off/`, `flash-d3/`).

User guidance/changelogs/AI indexes updated in both packages. No version bump,
commit, release or live-service restart. Existing unrelated work preserved.
ADR impact remains None: adapter repair, not a new authority or provider.

Residual limits: different APC managers cannot currently co-batch; fixed MTP
cohorts are not continuous admission. Client-wide callbacks may interleave.
Live Gateway/Flow/Code/Assistant UI adoption was not exercised; no selector or
`_runtime.speculation` inheritance was implemented. Backend drivers, external
Flash heads, pressure eviction and discovery/UI remain Core planned 0848–0853;
cross-key batching/tenancy is recorded in Core 0847. This is not a new oMLX
performance comparison or vision qualification.
