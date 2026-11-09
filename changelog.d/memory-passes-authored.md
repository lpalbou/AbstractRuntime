### Added

- **Memory's passes author their records WITH them, through the summaries capability** (`identity/authoring.py`,
  `identity/life.py`, `identity/chat.py`, `identity/entity_runtime.py`, `integrations/abstractmemory/brain_handlers.py`;
  AbstractMemory c46b4cd, Codex review 2 #10). `capability_formation_author(llm_for_capability, artifacts)` and
  `handler_formation_author(handler, artifacts)` build memory's `author(kind, body, context)`: one structured call
  per record, author `capability-model` / `summaries`, the client's reader, the call kept as a home artifact (its
  sources); any failure raises the client's own exception, which memory records (`author_raised`, class, message).
  Wired into every pass the runtime runs: the loop's sleep (`sleep_pass`: consolidation, world models, candidates,
  dream; the old-engine `dream_pass` rung too), a session's card updates (`world_model_update`, both sites) and the
  flow-brain `MEMORY_CONSOLIDATE` (the host's `capability_handlers["summaries"]`). `log_pass_authoring_failures`
  writes one authoring-failure line per `authoring.failures` entry a pass reports (pass, reason, class, message,
  the capability model). The runtime never calls `wake_cue_dedup_pass`.
