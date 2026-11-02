### Added
- **Public updates carry their type in replay** (`replay/view.py`). A `public_update` message has
  `update_kind`: `"finding"` when the actor typed the event `kind: "finding"` (a fact or piece of
  evidence it established), otherwise `"progress"` (what it is doing), including events written
  before the field existed. Clients use it as given, for example to list the findings of a
  conversation. See [docs/replay.md](docs/replay.md#public-conversation-messages).
