### Changed

- **What a turn formed is served beside the reply as structured notices; the served reply carries no bracket
  markers** (`identity/chat.py`, `identity/reflection.py`, `identity/visit_workflow.py`, `history_bundle.py`,
  `replay/view.py`; contract agreed with the Entity app owner 2026-11-09). One shape everywhere:
  `[{kind: "diary"|"question"|"lesson"|"interest"|"realization", record_id: <the GRAPH id of the formed record; a
  diary entry's projection, never only its diary_ id>, title: <as formed, "" while it is still being written>,
  text?: <his words, whole>}]`, in the order the acts stand in his reply; a private diary entry is `title ""`,
  `visibility: "private"`, no words; a visit-lane lesson/interest/realization staged for the close is
  `record_id: null, staged: true`. Served on: `TurnReport.reply_notices` (chat lane; `ChatSession.turn()` returns
  the served reply), the visit turn's `_turn.reply_notices` / `_turn.reply` and its ANSWER_USER payload
  (`message` = the served reply, `notices` = the list: the ledger is the source every app reads), the replay
  view's answer message (`notices`), and every session turn row (`turn.notices`, read from the run output's
  `notices`; `[]` when none). The served reply is the marked reply without exactly the markers those notices
  replace (`[kept in diary - ... - reread: diary_read diary_...]`, `[kept a private diary entry ...]`,
  `[learned: "..."]`, `[kept interest: "..."]`, `[realized: "..." - held for sleep]`; removed once, a marker
  alone on its line takes its line); a refusal marker stays (nothing formed to point at). The STORED reply keeps
  every marker (root ADR 0054): `TurnReport.marked_reply` / `_turn.marked_reply`, the episode verbatim and the
  session history he re-reads. The `notices` strings stay the operator's log lines. Reflection gains the marker
  builders `interest_marker`, `lesson_marker`, `realization_marker` (the parsers use them) and `without_markers`.
