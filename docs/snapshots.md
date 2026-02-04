# Snapshots (bookmarks)

A **snapshot** is a named, searchable checkpoint of a run state.

Motivation:
- debugging (“return to a known-good state”)
- observability (“inspect state at time T”)
- manual experimentation (“branch from snapshot later”)

Implementation: `src/abstractruntime/storage/snapshots.py`

## Data model

A snapshot stores:
- `snapshot_id`, `run_id`, optional `step_id`
- `name`, `description`, `tags`
- timestamps
- `run_state` (as a JSON dict)

## Stores

Included stores:
- `InMemorySnapshotStore` (tests/dev)
- `JsonSnapshotStore` (file-per-snapshot)

Search (MVP):
- filter by `run_id`
- filter by single `tag`
- substring match in `name` / `description`

## Restore semantics

Restoring a snapshot is a **host-level** operation:
1. load a snapshot from `SnapshotStore`
2. write `snapshot.run_state` back into your configured `RunStore`

Compatibility note:
- snapshot restore cannot guarantee safety if the workflow spec/node code has changed since the snapshot was taken.

## See also

- `architecture.md` — how snapshots fit with RunStore/LedgerStore/ArtifactStore

