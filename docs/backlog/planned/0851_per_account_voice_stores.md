# 0851 — Per-account voice stores: the voice settings of each account's runtime, and the scope in the voice facade

**Status**: planned · **Priority**: P1 · **Created**: 2026-10-27
**Cross-package**: AbstractVoice backlog 0065 (personal stores and a shared library), AbstractGateway backlog 0942 (personal voices per account, admin library, an AI's voice in its profile).

## Why
Operator, 2026-10-27: each person registers the voices they want (clone, design) and they are
personal, not shared by every account; an administrator can still deploy voices for everyone;
an AI (assistant, coder, entity) gets its voice as part of its profile. Today AbstractVoice keeps
one store per machine user, so every account of a gateway shares every cloned voice, and the
gateway's voice changes are admin-only until this lands.

## Current code reality (2026-10-27)
- `integrations/abstractcore/voice_facade.py` / `voice_library.py` (0.12.1): the voice library
  facade; local clients build a capability registry from `discovery_queries._runtime_capability_registry`
  (owner config: discovery settings only), remote clients call the AbstractCore server's `/v1` routes.
- Synthesis uses the LLM client's provider config (`create_local_runtime(llm_kwargs=...)` keys
  become the capability-plugin config, see `voice_openai_api_key`).
- No setting names a voice store; AbstractVoice reads its machine-user folder.

## Scope
- In: accept AbstractVoice's `voice_store_dir` and `voice_library_dirs` (0065) in the runtime's
  AbstractCore config so synthesis and the voice facade of one runtime use the same stores; the
  facade passes them to the local registry; records carry `scope: personal|library`; a
  `deploy_to_library` facade call for hosts that allow it; remote mode forwards nothing (a remote
  AbstractCore server keeps its own stores) and says so.
- Out: who may deploy (the host's policy), the gateway routes (0942).

## Validation
Two runtimes with two store folders: a voice designed in one is spoken by it and not listed by the
other; both list and speak a library voice; the library refuses changes from a personal runtime.
