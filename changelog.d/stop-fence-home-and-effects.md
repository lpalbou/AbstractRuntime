### Fixed

- **A home opened with its lifecycle fence refuses every write after the operator's Stop** (STOP-LANE-DESIGN §2.2;
  Codex review 3 #1; AbstractMemory 4074406). `open_home(..., fence=)` passes `fence.check` as memory's
  `commit_guard` (called inside every write transaction, before COMMIT: a Stop rolls the write back), and guards the
  diary book (`DiaryStore(guard=)`, before an entry is appended) and the artifact store (`FileArtifactStore(guard=)`,
  before a store, metadata update, access record or delete). Its effect handler table is wrapped once
  (`lifecycle_fence.fence_handlers`): an effect after the Stop is refused without running, a model reply that returned
  after it is abandoned, and a handler whose commit the guard refused reports the Stop - always
  `EffectOutcome.cancelled` with `KILL_SWITCH_REASON`, never failed and never retried. The loop's day sessions and its
  night open their home with the loop's fence.
- **`open_entity_runtime(..., fence=)` is required**: the visit lane's Runtime runs over a fenced home and its FINAL
  handler table (the host's LLM and tool handlers, the capability routing) is wrapped, so a Stop ends the run
  CANCELLED with the operator's reason. The run's own ledger keeps an unguarded artifact store (the records of a Stop
  are written after it). **Breaking for hosts**: the gateway's visit and mail lanes must pass the run's fence.
- **A Stop inside formation authoring is never an authoring failure**: `ChatSession._author_records` /
  `_author_diary_entry`, `authoring.author_metadata` and `diary_authoring_payload` let the operator's Stop through
  (it used to become a pending record with a failure line), and `ChatSession._effect` raises `StoppedByOperator` for
  a fenced refusal instead of "<effect> failed". A Stop that memory's guard raises inside the night's engine ends the
  night with its one clean line.
