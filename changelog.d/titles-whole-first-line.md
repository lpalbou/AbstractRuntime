### Fixed

- **An untitled lesson, interest or realization keeps its whole first line as its title** (`identity/reflection.py`
  `elected_title_and_words`; the diary projection's slug in `integrations/abstractmemory/identity_support.py`):
  without a `title:` line, the title is the entity's whole first line - his own line break, never a sentence split.
  The sentence split cut "Dr. Smith taught me patience." to "Dr." and "I learned that U.S. Courts are slow." after
  "U.S.", and that cut title was stored for good (review 2026-11-09). A `title:` line still wins; a display folds a
  long title, the stored one stays whole.

### Changed

- **A new lesson's or realization's title is its title** (`identity/chat.py` mid-turn, look-back and realization
  sites; `identity/visit_workflow.py` APPLY): the `lesson: ` and `realization: ` prefixes are gone, as `interest: `
  already was. Records stored before keep theirs; the Entity app's labels already strip them.
