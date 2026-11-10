### Added

- **The sleep forms the entity's topics** (operator 2026-11-10: topics "must run server side (runtime) ... during
  sleep/maintenance phase ... so that it becomes part of the knowledge of the entity"). `build_consolidator` runs one
  new stage after METADATA REPAIR and before the night voice: `identity/topics.sleep_communities`. It calls
  AbstractMemory's `abstractmemory.communities.embed_missing` first, so every vector-less record of the entity's scopes
  gets a vector, then `communities_pass` forms the night's version of the entity's communities in one guarded
  transaction. The pass gets the night's own author (the summaries capability that titles every record the night
  forms) and what the repair step left of the night's authoring budget (`authoring_cap` minus the tasks the repair
  took, in model calls). Memory owns every graph operation; the runtime holds no community code.
- The night's result carries memory's receipt under `communities` (`version_seq`, `communities`, `new`, `kept`,
  `renamed`, `dissolved`, `authored`, `pending`, `failed`, `skipped`, ...), plus the embedding counts
  (`embedding: {embedded, skipped, failed}`) and the `budget`.
  - One own_time.log line says what formed (including the unplaced records whose only neighbours are closed,
    and the detection rounds), or why memory wrote no version (unchanged, empty, nothing vectored).
  - A title the pass could not author gets its authoring-failure line.
  - Records that were not embedded get one night note, by memory's reason codes; never a failure.
  - When a version is written, `communities_version` joins the completed-maintenance detail of that night
    (memq_maintenance), with the same completion time and seq.
- The stage is fenced. Both verbs get `cancelled=fence.moved`, the author is checked before every call (no model call
  starts after the operator's Stop), and memory's commit guard refuses the version after a Stop. A stopped night
  records nothing for its topics: no version, no line, no maintenance note. The receipt says `stopped`, and the night
  ends with its one Stop line. That Stop line now also closes a night stopped during the repair step.
- A memory without the verbs produces one loud `#MISSING_VERB` line per sleep and the receipt
  `{"missing_verb": ...}`. A receipt this runtime cannot read produces one loud line naming its keys. In both cases
  the night's other work stands; the runtime never falls back to computing communities itself.
- `_sleep_authoring`'s receipt gains `considered` (the authoring tasks the drainer took).
