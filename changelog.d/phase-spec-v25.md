### Changed

- **The vendored phase graph is spec v25, the five-state life cycle** (`identity/spec/entity_phases.vendored.json`,
  `identity/life.py`; byte copy of abstractentity 814c574): a newborn starts idle (`initial_phase: idle`, a state kept
  in `statuses`, never a phase key), the edges into work are guarded by `job_queued`, and `work->personal#operator` is
  removed with the work toggle. The loop never names that edge: the return to his own time after the jobs is
  `work->personal#no_task`, and when a job is still queued but the blueprint removed the work landing, the gate lands
  idle (`edge_removed`) instead of personal.
