### Changed

- **The vendored phase graph is spec v24** (`identity/spec/entity_phases.vendored.json`): `maintenance_done` is
  shipped in `cause_evaluators` (the loop writes `sleep->personal|work#maintenance_done` when his own sleep's pass
  completes), so the operator's blueprint edits may now target it. Byte copy of abstractentity b49e8ed.
