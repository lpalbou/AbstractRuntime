### Fixed

- **An entity re-reading its own session reads its own replies bare** (`core/turn_author.py`,
  `session_history.py`; review 2026-11-09 REVIEW-1 #7). Replayed history labelled every entity reply
  "<name> (entity): ...", including the replies of the entity that was reading it (a re-summoned entity
  replays its own session): a model may copy the prefix, and the bytes no longer matched what it said,
  so every prefix cache of that session missed. `session_chat_messages(..., reader_entity_id=...)` and
  `speaker_line(author, text, reader_entity_id=...)` now leave the reader's own replies exactly as
  stored; every other reader still sees who spoke. `entity:<name>`, `entity:<name>@<home_id>` and a bare
  name are the same reader. Hosts that seed an entity run's history pass the run's entity id
  (AbstractGateway: the summon seeding in `bundle_host`).
