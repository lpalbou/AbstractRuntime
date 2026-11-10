### Added

- **A revocation of own time records who revoked it.** `write_personal_grant(..., revoked_by=)` takes
  "kill_switch" (the operator's Stop) or "operator" (his own revoke). Mode disabled then writes
  `{mode: disabled, revoked_by: ...}`, `read_personal_grant` answers it, and the next grant clears it. Naming a
  revoker on an arming write, or an unknown revoker, refuses and writes nothing.
- The runtime's own revokers name themselves: `hard_stop_loop` records `kill_switch`, and the CLI's
  `--revoke-personal` records `operator`. The gateway's revoke doors pass theirs.
