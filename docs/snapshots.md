## Snapshots / bookmarks

A **snapshot** is a named, searchable checkpoint of a run state.

Motivation:
- debugging (“return to a known-good state”)
- observability (“inspect state at time T”)
- manual experimentation (“branch from snapshot later”)

---

### Data model

A snapshot stores:
- `snapshot_id`
- `run_id`
- optional `step_id`
- `name`, `description`, `tags`
- timestamps
- `run_state` as a JSON dict

Implementation:
- `src/abstractruntime/storage/snapshots.py`

---

### Stores

- `InMemorySnapshotStore` (tests/dev)
- `JsonSnapshotStore` (file-per-snapshot)

Search (MVP):
- filter by `run_id`
- filter by single `tag`
- substring match in `name` / `description`

---

### Restore semantics

Restoring a snapshot is a **host-level operation**:
- load snapshot
- write `snapshot.run_state` back into your `RunStore`

Compatibility note:
- v0.1 does not guarantee safe restore if the workflow spec/node code changed.

