### Changed

- **Every memory's title and summary are authored with it** (`identity/authoring.py` new, `identity/chat.py`,
  `identity/visit_workflow.py`, `identity/life.py`, `integrations/abstractmemory/seam_handlers.py`,
  `integrations/abstractmemory/identity_support.py`, `identity/title_failures.py`; operator rulings 2026-11-09: "the
  title + summary must ALWAYS exist for any memory"; "the metadata of any newly generated memory must be generated
  along with the memory"; needs AbstractMemory c4ffc45). An elected interest, lesson or realization keeps the
  entity's `title:` line and words; what its block lacks, and every episode's and session summary's title and
  summary, the `summaries` capability model authors in the SAME step that forms the record (one structured call;
  the call is kept as a home artifact and named in `attributes.metadata_authored`). Only when that step fails does
  the record form PENDING through AbstractMemory (`MEMORY_FORM` records may carry `pending: [title|summary]`; a
  missing title is None, never a cut or a placeholder - the 8e72975 `untitled <kind> [<hex>]` label is retired, a
  legacy one is still never rendered), memory queues its `title.author` / `digest.author` task, and one JSON line
  (now with `missing: title|summary|both`) goes to `<workspace>/untracked/authoring-failures/<date>.jsonl` (renamed
  from title-failures; the log marker is `#AUTHORING_FAILURE`). A public diary entry without a gist forms with its
  title pending. Homes open with `task_queue=True`.
- **Pending records are retried, the sleep repairs what is left** (`drain_authoring`, `declare_missing_metadata`):
  the drainer runs only `title.author` / `digest.author`, at the own-time loop's tick boundary (inside the tick's
  writer window, after a visitor) and at a session's close, bounded by the cognition profile's `cap_per_turn`; it
  writes through `retitle` / `rewrite_digest` (which settle the task and clear the marker), defers on an unreachable
  endpoint without spending an attempt, and logs a reply without the fields. The sleep pass adds one repair step
  after its engine pass: records lacking a title or summary (or carrying the retired placeholder) get their task,
  then pending ones are authored; its receipt carries `authoring: {declared, authored, deferred, failed}`.
- **Capability clients in the loop process**: `spawn_loop_process(capability_endpoint_profiles={cap: {spec,
  profile}})` hands them to the child in one env key (never logged); the child installs them in the ambient endpoint
  resolver and builds `llm_for_capability` (`build_capability_llm_factory`, the day client's patience window) for its
  sessions, its drainer and its sleep.
