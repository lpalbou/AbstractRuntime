## 006_snapshots_bookmarks (completed)

### Goal
Provide named snapshots/bookmarks for durable run state.

### What shipped
- `src/abstractruntime/storage/snapshots.py`
  - `Snapshot`
  - `SnapshotStore`
  - `InMemorySnapshotStore`
  - `JsonSnapshotStore`

### Search semantics (MVP)
- filter by `run_id`
- filter by single `tag`
- substring match in name/description

