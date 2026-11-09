### Changed

- **The sleep repairs legacy cards and dreams too, titles and summaries** (`identity/authoring.py`; AbstractMemory
  4f3a77e forms cards and dreams pending, and its passes refuse a home without a task queue - every runtime path that
  runs them opens the home through `open_home`, `task_queue=True`). The repair step finds records stored before that
  with memory's mechanical metadata (a card titled "World model: <subject>" without authored title provenance, a
  dream titled "Dream: ...", a card body in `mechanical-card-v2` or carrying a mechanical delta, a legacy dream's
  body) and any record lacking a title or summary, and DECLARES them pending through memory's `declare_pending`
  (AbstractMemory 11687df: the marker and the `title.author` / `digest.author` tasks formation gives; a declared
  title reads None until authored, the old one kept in its history). The drainer then authors their titles
  (`retitle`) and their summaries (`rewrite_digest`; a card through its own write, a dream in place). A refused
  declaration (e.g. `protected_kind`) is logged (`#REPAIR_REFUSED`) and counted; memory's `rearmed` answer (a
  missing task repaired, AbstractMemory 39e479e) is a success counted with the re-arms; the receipt is
  `authoring: {declared, refused, rearmed, authored, deferred, failed}`.
