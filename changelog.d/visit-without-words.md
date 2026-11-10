### Fixed

- **A visit with no word exchanged is not remembered** (`identity/visit_workflow.py`, `identity/track_record.py`,
  `identity/acquaintances.py`). Operator ruling 2026-11-11 00:40: "(a visit opened and closed without a word) that
  should not create a memory if no word was shared" and "a zero-turn visit ... no and it should not even be
  remembered, fix it". A visit opened and closed (by the operator, the idle deadline or the pause freeze) with no
  turn already formed no record and ran no model call (REFLECT has no sheet to look back on); that is now pinned,
  row for row over every table of the memory store. What still counted it was the life's derived counts: the
  track record listed it as a completed visit (and in the reputation's completion rate) and the acquaintances
  counted an interaction with the visitor. `visit_without_words(run)` names the rule (a visit run whose
  `_visit.turn_n` is 0 or absent; a malformed count raises), and `track_record.is_wordless_visit` keeps such a run
  out of both: it is neither a counted run nor an interaction, and an open visit counts from its first line. The
  ledger keeps the run's open and close rows (the ledger is the record of the run). A visit with one line or more
  is unchanged.
