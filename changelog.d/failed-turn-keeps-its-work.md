### Fixed

- **A failed visit turn keeps what it already did** (`identity/visit_workflow.py` `VISIT_TURN_FAILED`; Codex review
  2026-11-09 #5). The failed turn's messages were deleted from the adapter transcript, erasing completed tool results
  and operator guidance, and the error answer reported `tools_ran: []`. Now the transcript keeps them and an explicit
  failed-turn outcome closes the turn (`[this turn could not be answered: <error>]`, metadata `kind: turn_failed`),
  the visit history folds the turn with that outcome, and the error answer reports the tools that really ran.
