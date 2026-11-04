### Added

- **Acquaintances of an entity or a session** (`identity/acquaintances.py`, framework backlog 1119):
  `entity_acquaintances(home_dir, run_store)` and `session_acquaintances(session_id, run_store)` answer two
  lists, the other entities and the humans it interacted with, each with first and last seen, the number of
  interactions and the task kinds. Computed from existing records, never stored: the participants the
  gateway stamps on visit, summon and meet runs, the person who started a run (`_meta.origin`), and who left
  a task in the task inbox. Identifiers only; interactions with no usable identifier (an `agent:` participant,
  a participant without a namespace, a person-started run without an actor) are listed under `not_counted`,
  never guessed. See `docs/track-record.md#acquaintances`.
