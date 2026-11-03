### Added

- **The idle unload reaches the host: `set_model_idle_unload(seconds)`** (host facade, `LocalAbstractCoreLLMClient`,
  `MultiLocalAbstractCoreLLMClient`; framework backlog 1115). It relays AbstractCore's process-wide idle-unload time
  (ADR 0012, default 60 minutes): a model no call has used for that long is unloaded, a locked model never is, and a
  running call is never interrupted. `None` or `0` turns it off; a refused value answers `ok: false`. Local residency
  rows of in-process text models carry `idle_unload_s`, `last_used_at` and `idle_unload_at`, and pass through Core's
  `in_use` and `locked_by`, so a host can show a model that is in use, or kept loaded by another account.
- **`set_model_residency_locks(locks)`**: the locked models become exactly the listed ones (resident ones locked,
  missing ones loaded then locked, the others unlocked), each pair with its own outcome. A lock names a provider and a
  model; the `base_url` of a configured endpoint is not part of the key. A host stores its "Keep loaded" set and
  replays it at start; as a `set_` name it is replayed when the gateway's model host rebuilds a client.

### Changed

- **AbstractCore 2.34.0 or newer** (base install, `apple` and `gpu` settings): the idle unload and its residency rows
  come from it.
- **A load with `ttl_s` or `keep_alive` on an in-process model** still lists them under `unsupported_options`; its
  warning now names the idle unload that applies instead ("the idle unload setting applies (60 min unused, never while
  it is kept loaded)"), or says it is off.
