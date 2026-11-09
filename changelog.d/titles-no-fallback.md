### Changed

- **A missing title is recorded, never guessed** (`identity/reflection.py`, `identity/title_failures.py`,
  `identity/chat.py`, `identity/visit_workflow.py`, `identity/diary.py`,
  `integrations/abstractmemory/identity_support.py`; operator ruling 2026-11-09: "we shouldn't need a fallback. if
  it fails, it should add information to a temporary error file in untracked/"). An elected interest, lesson or
  realization takes its title only from the block's `title:` first line, and a public diary entry only from its
  `gist:` line; his first line (or first sentence) no longer stands in. `parse_elected_title(block)` returns
  `(title | None, words, reason)` with reason `missing`, `empty`, `multi_line` (two `title:` lines), `malformed` (the
  `title:` line is not first) or `too_long` (over `TITLE_MAX_CHARS` = 200); `elected_title_and_words` returns
  `(title | None, words)`. The words stay whole. Each failure appends one JSON line (observed_at, entity, record_id,
  kind, raw_block, template_id, template_version, provider, model, reason, stored_title, plus session_id / turn_id /
  entry_id) to `<workspace>/untracked/title-failures/<YYYY-MM-DD>.jsonl`, the workspace being the parent of the data
  dir that holds the home; no environment variable is read. When no such folder resolves the same line goes to the
  home's `own_time.log` after `#TITLE_FAILURE` (and to the session's output). AbstractMemory requires a non-empty
  title, so until it accepts a missing one the record carries the visibly-untitled label `untitled <kind> [<key>]`
  (the key: 6 hex of the block's sha256, none of his words). Block bodies now keep his line breaks (the checks read
  lines). `build_diary_effect_handlers(home_dir=...)` locates the failure file; a host that does not pass it gets the
  marked line in the projection's warnings.
