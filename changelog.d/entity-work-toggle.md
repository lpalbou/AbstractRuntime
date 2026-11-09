### Added

- **resume_to is a runtime-owned phases.yaml bucket** (`identity/life.py`; contract-phases v2 §6.4).
  `resume_to: {phase: personal|work|null, written_by, written_at}` (the own-time phase an interruption took over,
  spec v23) has one writer, `write_resume_to`, read by `read_resume_to`. The writer field-merges (every other key of
  phases.yaml rides through; a corrupt file or a newer schema is refused). Until a home's bucket exists the read falls
  back to the gateway sidecar `<entities_dir>/.life_phase/<slug>.json` (`resume_to`); `migrate_phase_sidecar(home)`
  copies it once, idempotently (result `{"resume_to": "migrated"|"present"|"absent", "sidecar"}`), after which the
  sidecar is a projection only.

### Changed

- **There is no work toggle: work starts when a job is queued** (`identity/life.py`; CONTRACT v3 D1, operator
  2026-11-09). The day gate runs a work day whenever a work order stands, whatever the personal grant says, and never
  enters personal unless it is granted. A `work` key in phases.yaml or in the gateway sidecar (written by a
  development build) is ignored: nothing reads or migrates it, and a phases.yaml write carries it through untouched.
  `ChatSession.yield_check` stays a visitor check (True = a visitor is at the door).
