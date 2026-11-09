### Changed

- **Every record a visit's close forms carries its authored metadata** (`identity/visit_workflow.py`; operator
  2026-11-09). The look-back summary (title and summary) and each interest, lesson or realization elected without a
  `title:` line (title; his words stay the summary) get one LLM_CALL tagged `capability: "summaries"` in the APPLY
  step right before the record forms (`_reflect.authored.<summary|interest-i|lesson-i|realize-i>`), and form with
  the authored metadata (`attributes.metadata_authored`; a floored summary loses its mechanical-floor label once
  authored). Only a failed or unrouted call forms the record pending (memory's task), with its failure line written
  at the next APPLY once the record has its id. A visit never fails on it.
