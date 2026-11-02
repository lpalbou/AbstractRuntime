### Added
- **Public updates carry their type in replay** (`replay/view.py`). A `public_update` message has
  `update_kind`: `"finding"` when the actor typed the event `kind: "finding"` (a fact or piece of
  evidence it established), otherwise `"progress"` (what it is doing), including events written
  before the field existed. Clients use it as given, for example to list the findings of a
  conversation. See [docs/replay.md](docs/replay.md#public-conversation-messages).
- **Replayed tool calls keep the model's call id** (`replay/view.py`). An `activities[].tool_calls[]`
  row keyed by a runtime call id (`id`) also carries `call_id`, the model's own id, which is what a
  public update's `source_call_ids` cite; a client can now resolve a finding to the results it cites.
