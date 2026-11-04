### Added

- **An entity's digital identity: its look and its voice, one record in its home**
  (`abstractruntime.identity.digital_identity`, framework ADR 0052 draft, backlog 1119; `docs/digital-identity.md`).
  `<home>/identity.json` holds the avatar design, its colour and the voice (`schema_version` 1). `write_identity` checks
  every bound (`IdentityInvalid` names the field), runs the gateway's `check` hook, appends an `identity_changed` row to
  `state_history.jsonl` (the `phase_changed` field names) before the file is replaced atomically, and never gives a
  `rev` twice. A file it cannot read in full raises `IdentityUnreadable` on read and on write, never "no identity".
  `default_identity` is the recorded default anyone can recompute (SHA-256 of the entity id, eight engraved Watch
  designs, an HSL colour, the gateway's default voice as a value); `record_default_identity` writes it once and never
  overwrites. `identity_at_rev` answers an old rev from the biography. The reference `entity:castor#3` rides
  `vars._runtime.identity_ref` beside the signed entity stamp (`stamp_run_identity_ref`, `read_run_identity_ref`,
  `parse_identity_ref`). `identity_fingerprint` derives the individual's `actor_id` from the manifest, with the diary's
  chain head. Homes without `identity.json` are untouched.
- **The summon tells the entity when its look or voice changed**: one sentence ("Your voice changed on 4 November (by
  laurent): now fr-male-rene.") in the summon prelude, after the identity core, when the change is newer than the
  previous summon (`render_summon_prelude(home_dir=, identity_since=)`, `ChatSession(identity_since=)`,
  `build_visit_workflow(identity_since=)` or `_visit.identity_since` seeded by the door).
