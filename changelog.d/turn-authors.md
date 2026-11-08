### Added

- **Who speaks in a turn: `author` on session turns, replayed history and the replay view**
  (`core/turn_author.py`, `history_bundle.py`, `session_history.py`, `replay/view.py`). One shape everywhere, every
  key present: `{kind: "entity"|"agent"|"person", name, entity_id, identity_ref, avatar: {style, color}|null}`.
  Session turn rows and every history bundle's `run` section (so session history blocs) carry `prompt_author` (who asked) and `author` (who
  answered). A host stamps them at `vars._meta.prompt_author` / `vars._meta.author` (`stamp_turn_authors`); without a
  stamp the answer is the entity of a summon run (its signed `_runtime.entity` stamp), else the agent named by
  `vars.workflow_selection.name` (then the workflow id's bundle id; never a model name), and the prompt is the
  origin's person (`_meta.origin.actor`), name null on a run older than origins. The history replayed to the next
  agent run (`session_chat_messages`) puts the author in each message's `metadata.author`, and an ENTITY's reply is
  rendered `<name> (entity): <reply>` (`speaker_line`) so an agent never reads an entity's words as its own; agent
  and person messages keep their exact bytes. Replay view messages with role `user`/`assistant` carry `author` from
  the root run (relabelled when the snapshot is folded after the records); the 15 conformance goldens are
  regenerated for the new key (cursor tokens change with the snapshot digest).
- **`repeat_of`: a turn whose prompt repeats another turn's** (`core/turn_author.py` `prompt_repeat_of`). A host
  stamps `vars._meta.repeat_of = <run id>` (the gateway does it for an `@entity` mention that answers the person's own
  turn again). Session turn rows and history bundle `run` sections carry `repeat_of` (null when none); the replay
  view's input message carries it when set; replayed history keeps the prompt (the pair stays whole, so the
  entity's reply still reaches the next agent run) and marks the user message's `metadata.repeat_of`.
- **A replayed answer carries its own time** (`session_history.py`): the assistant message's `metadata.ts` is the
  run's `updated_at` (when it answered, never before the prompt's `created_at`), so a reader sorting by `ts` never
  shows an answer above its question; the user message keeps `created_at`.
