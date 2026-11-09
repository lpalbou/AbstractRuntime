### Fixed

- **A pending record whose authoring tasks all ran out gets a fresh one** (`identity/authoring.py`,
  `identity/life.py`; operator 2026-11-09: a title and a summary must always exist). A capability failure that
  exhausted a `title.author` / `digest.author` task's attempts left the record pending with no live task, forever.
  `rearm_exhausted_authoring(ms, scopes=)` finds those records with memory's `pending_records(live_task=False)` and
  re-queues them with memory's `rearm_pending` (the marker stays, its `rearmed` count grows): at the start of every
  drainer pass (every scope) and in the sleep's repair step (the home's scopes; receipt `authoring.rearmed`). A
  refusal is logged `#REARM_REFUSED`; a memory without the verbs is said once per process (`#MISSING_VERB`), and
  nothing is written around memory.
