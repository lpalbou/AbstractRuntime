### Fixed

- **No authoring model call starts after a Stop** (Codex review 4 #3, P1). Every authoring path resolved its
  capability client and called it with no check in between, so a Stop landing during the resolution (or anything
  between the outer check and the call) still sent the model a call. `ChatSession._author_records` /
  `_author_diary_entry`, the authoring drainer, and memory's formation authors (`capability_formation_author`,
  `handler_formation_author`, which now take a REQUIRED `should_stop` - the admission's fence) check immediately
  before each model call and after it returns, and let the operator's Stop through instead of turning it into a
  deferral or an authoring failure.
