### Changed

- **A new interest's title is its title** (`identity/visit_workflow.py` APPLY, `identity/chat.py` look-back and
  mid-turn sites): the `interest: ` prefix the runtime put in front of the entity's own title is gone (operator
  2026-11-08). Interests stored before keep theirs; AbstractMemory's world-model key already strips it, so no card is
  re-keyed.
