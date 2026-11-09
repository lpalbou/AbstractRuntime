### Fixed

- **A public diary entry's record carries no template: his gist, else an authored title** (`integrations/abstractmemory/
  identity_support.py`, `identity/diary.py`, `identity/chat.py`, `identity/act_only.py`, `identity/entity_runtime.py`;
  Codex review 2 #10, live 2026-11-09: castor's new entry read "Diary entry (note) - <date>: ..."). The projection no
  longer builds "Diary entry (<kind>) - <date>[: <gist>]" titles, entry-id tails or the "Wrote a diary entry ...; no gist
  elected" body. His elected gist is the title, whole; a gistless public entry (and a writer-declared mechanical note,
  whose gist is machine words) gets its title from one summaries capability call BEFORE the write - ChatSession's
  DIARY_WRITE (`authored_metadata` on the payload, the call kept as a home artifact) and the visit lane's act-only
  capture (the host's summaries capability handler). The record's summary is always his words (his gist, else his
  first line, whole): a diary entry is his act and memory never authors its body. Only a failed (or absent) authoring
  forms the title pending, with one authoring-failure line naming the record, the reason, the error class and the
  capability's model. Private entries are unchanged (content-free act frame) pending the operator's ruling.
