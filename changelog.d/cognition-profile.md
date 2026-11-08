### Added

- **An entity's cognition profile: a model per part of its thinking, how many memories a turn brings to mind, and
  how readily it revisits what it understood** (`identity/cognition_profile.py`, `docs/entity-runtime.md`). A home
  can carry `cognition.yaml` beside `substrate.yaml`. `capabilities` names the model for `summaries` (digests,
  titles, keywords, cards), `understanding` (ripples, dream narration) and `extraction`; each is `{inherit: mind}`
  by default or `{provider, model, thinking?, max_input_tokens?}` (`max_input_tokens` 4096 to 140000, 140000 when
  absent). `recall.shelf_size` (1 to 400) is the number of memories retrieved per turn in visits and own time;
  `rebuild.sensitivity` (0 to 1, default 0.5) and `rebuild.cap_per_turn` (0 to 10, default 3) set how readily
  memory work re-opens what is understood. `lookback.context` (`day` by default, or `tick`) says what the look-back at
  the end of a tick of its own time sees: the whole day so far, or this tick's exchange plus a recap of earlier
  moments; `lookback_context(home)` reads it. `resolve_capability(home, cap, mind=None)` returns
  `{provider, model, thinking, max_input_tokens, source}` with `source` `profile` or `mind`; a capability that
  inherits resolves through `resolve_home_substrate` (a host passes its resolved mind with `mind=`), and when
  nothing names a model it raises `SubstrateUnset`. `recall_shelf_size(home, default)`, `rebuild_policy(home)` and
  `read_cognition_profile(home)` (every field with its source) read it; a malformed file or field reads as its
  default with a warning naming the field. `write_cognition_profile(home, profile)` validates first
  (`CognitionProfileInvalid` carries the field and a sentence to show) and writes atomically; the gateway is its
  one caller.
