### Fixed

- **A failed model call fails that turn, never the visit** (`identity/visit_workflow.py`; operator 2026-11-09 "a
  visit never fails"; gateway slice finding: a failed mirrored turn ended the visited entity's whole visit run).
  Every model call of a visit turn (the v0 single call and each call the ReAct middle plans) is routed through
  `VISIT_MODEL_GATE`: its final failure, after the effect policy's retries, is absorbed into its result key
  (`_absorb_failure`) and the turn goes to `VISIT_TURN_FAILED`, which answers that turn with `level: "error"`,
  `message: "This turn could not be answered: <error>"` and `turn_error`, takes the failed turn's visitor message
  back out of the adapter transcript (the next turn never stacks two user messages), forms nothing, and parks: the
  visit continues and the next turn is answered.
