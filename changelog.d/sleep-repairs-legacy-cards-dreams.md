### Changed

- **The sleep repairs legacy cards and dreams too** (`identity/authoring.py`; AbstractMemory 4f3a77e forms cards
  and dreams pending, and its passes refuse a home without a task queue - every runtime path that runs them opens
  the home through `open_home`, `task_queue=True`). The repair step now declares records stored before that with
  memory's mechanical metadata: a card titled "World model: <subject>" without authored title provenance, a dream
  titled "Dream: ...", a card body in `mechanical-card-v2` or carrying a mechanical delta, a legacy dream's body. The
  drainer authors their titles through `retitle`. Their summaries cannot be written yet: memory refuses
  `rewrite_digest` on a card or dream without a pending marker (`use_author_world_model` / `use_narrate_dream`) and
  has no verb to declare a stored record pending, so the drainer releases the task and logs `memory_refused` in the
  authoring-failures file (no write around memory's rule).
