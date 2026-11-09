### Fixed

- **A public diary entry's summary is authored, never his first line standing in** (`integrations/abstractmemory/
  identity_support.py`, `identity/diary.py`, `identity/chat.py`, `identity/act_only.py`, `identity/entity_runtime.py`,
  `identity/visit_workflow.py`, `identity/authoring.py`; Codex review 3 #9/#11; AbstractMemory 71de051, which takes an
  authored summary on a public book projection). `diary_record_metadata` is the one rule: title = his elected gist,
  whole (a mechanical note's gist is never a title), else authored; summary = authored, formed with its reading
  (`attributes.understanding`: capability-model author, reader, the authoring call as its source; `digest_method`
  `authored-v1`). ChatSession's DIARY_WRITE and the visit lane's act-only capture author the summary (and the title
  without a gist) BEFORE the write; a failed or absent authoring forms the missing fields pending (his gist / first
  line only their provisional words) with one failure line, and the warning riding the run is word-free. The session
  sheet, the resolution notes and the visit capture metadata name the entry by its record's title (or say it is
  still being written), never by his first line. The sleep's repair declares a legacy public diary record's
  template title ("Diary entry (<kind>) — ...") pending and asks memory to declare its stand-in summary pending
  (memory 71de051 still refuses that with `protected_kind`; said once per pass). Private entries are unchanged
  (root backlog 1186).
