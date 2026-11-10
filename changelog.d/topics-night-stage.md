### Added

- **The sleep forms the entity's topics** (operator 2026-11-10: topics "must run server side (runtime) ... during
  sleep/maintenance phase ... so that it becomes part of the knowledge of the entity"). `build_consolidator` runs one
  new stage after METADATA REPAIR and before the night voice: `identity/topics.sleep_communities`. It asks
  AbstractMemory to embed every record of the entity's scopes that has no vector yet (`embed_missing`), then to form
  the night's version of the entity's communities (`communities_pass`). The pass gets the night's own author (the
  summaries capability that titles every record the night forms) and what the repair step left of the night's
  authoring budget (`authoring_cap` minus the tasks the repair took, in model calls). Memory owns every graph
  operation; the runtime holds no community code.
- The night's result carries the stage's receipt under `communities`: memory's receipt, the embedding counts
  (`embedding: {embedded, skipped, failed}`) and the `budget`. One own_time.log line says what formed, and a title the
  pass could not author gets its authoring-failure line. Records memory could not embed are a night note, never a
  failure. `communities_version` joins the completed-maintenance detail of that night (memq_maintenance), with the
  same completion time and seq.
- The stage is fenced. Both verbs get `cancelled=fence.moved`, and memory's commit guard refuses the version after
  the operator's Stop. A stopped night records nothing for its topics: no version, no line, no maintenance note. The
  receipt says `cancelled` (and memory's `cancelled_before_write`), and the night ends with its one Stop line. That
  Stop line now also closes a night stopped during the repair step.
- A memory without the verbs produces one loud `#MISSING_VERB` line per sleep and the receipt
  `{"missing_verb": ...}`. The night's other work stands; the runtime never falls back to computing communities
  itself.
- `_sleep_authoring`'s receipt gains `considered` (the authoring tasks the drainer took).
