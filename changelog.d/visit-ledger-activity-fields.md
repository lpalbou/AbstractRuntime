### Added

- **Three visit ledger fields for a live view of the turn** (read by the Entity app's live activity panel,
  abstractentity a82aa10, root backlog 1196). All additive; a reader without them shows what it showed before.
  - `topic` (string or null) on every handle of a visit's `memory_recall` record (`result.handles[]`, the passive
    recall and the recall of a live correction): the stable key of the topic (memory community) the memory is in
    now, null when it is in none. The visit asks for it with `with_topics: true` in the `MEMORY_RECALL` payload; the
    seam handler (`integrations/abstractmemory/seam_handlers.py`) reads memory's communities payload ONCE per recall
    and looks each handle up by its graph id. Hot-path cost: one communities read per visit recall, linear in the
    home (measured on a synthetic home: 19 ms at 2,000 records, 190 ms at 20,000), never `community_of` per handle
    (461 ms and 3.2 s for 44 handles). A topics read that fails leaves the recall whole, without `topic`, and says
    why in its `warnings`. The model's prompt is unchanged.
  - `iteration` and `max_iterations` (int) on the `effect.payload` of the `reason` model-call record (the react
    visit arm): this call's place in the turn's reasoning loop, from 1, and the turn's cap, read from the runtime's
    `_limits` when the call is planned. Metadata only: no handler sends them to a provider.
  - `hits` (int) on a `search_memory` result: the graph and book matches its text counts.
    `HomeMemoryReader.search_memory` answers a `SearchMemoryText` (a `str` with `hits`), `execute_tool_elections`
    copies it onto the new `ToolElection.hits`, and the `ENTITY_TOOLS_EXECUTE` result entries carry it. A host that
    builds its own tool result entries (AbstractGateway's visit `TOOL_CALLS` executor) copies `election.hits` as it
    copies `election.artifact_id`.
- **The own-time loop's failures stop says why** in `loop_status`: `note` = "Stopped after three failed ticks:
  <the last failure, as the provider gave it>. His memory is intact." (`failures_stop_note`, `LifeReport.last_failure`,
  `write_loop_status(note=)`; the next status write clears it).

### Changed

- **The crashed-loop note is a plain sentence for end users** (`loop_process_status`, shown verbatim on the Entity
  app's card): "His own time stopped without a word (the process is gone: a crash or a reboot). His memory is
  intact." (`life.CRASHED_LOOP_NOTE`), in place of "loop_status said running but the process is gone (crash or
  reboot)". "process is gone" stays in it: AbstractGateway's entity_phase reads a crash from those words.
