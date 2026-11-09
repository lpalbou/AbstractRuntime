### Fixed

- **The sleep's repair authors a legacy public diary entry's summary, not only its title** (`identity/authoring.py`;
  AbstractMemory 8235daf, which declares a legacy public book projection's summary pending under formation's rule,
  `diary_summary_refusal`; Fable review finding R2). A public diary record formed before authored summaries carried
  his first line as its summary: the repair's declaration is now taken (pending marker + `digest.author` task, the
  first line kept as provisional version 0), and the drainer authors the summary in the same step, beside the
  template title. Private projections and owner-direct entries are never declared. A summary memory still refuses
  (a projection claiming a chain position) is said once per night per rule memory names, with that rule
  (`protected_kind - ... (hash_chained)`), instead of the stale "until memory accepts the declaration".
