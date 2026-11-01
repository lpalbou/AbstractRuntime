### Added

- Tool approval waits can carry optional `details.public_action` (`text`, `update_id`, `run_id`,
  `iteration`, `call_ids`) supplied by an actor adapter alongside its proposed effect batch.
  Runtime validates its owning run and exact call identities, narrows it to the actual gated
  subset, and withholds mismatched or private-fenced context. The canonical replay preserves
  this public explanation; approval decisions and policies are unchanged.
