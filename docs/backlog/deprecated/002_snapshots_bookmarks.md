## Todo 2 — Snapshots / bookmarks (durable named checkpoints)

### Goal
Add a small, kernel-friendly snapshot system so a host (AbstractFlow / UI / operators) can:
- capture a run’s durable state at a point in time
- label it with name/description/tags
- list/search snapshots (basic)
- restore by loading the captured `RunState`

This is **not** workflow versioning, and not a full artifact store.

---

### Design constraints

- Snapshots must be **JSON-serializable**.
- Snapshots must not require the runtime to keep RAM state.
- Keep storage interface minimal; no UI assumptions.

---

### Data model (recommended)

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Snapshot:
    snapshot_id: str
    run_id: str

    # Optional: useful if you want to reference “what ledger step triggered this snapshot”
    step_id: Optional[str] = None

    name: str = ""
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    # Store durable run state as JSON dict (serialized RunState)
    run_state: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run_state(
        cls,
        *,
        run_state: Dict[str, Any],
        run_id: str,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        step_id: Optional[str] = None,
    ) -> "Snapshot":
        return cls(
            snapshot_id=str(uuid.uuid4()),
            run_id=run_id,
            step_id=step_id,
            name=name,
            description=description,
            tags=tags or [],
            run_state=run_state,
        )
```

Notes:
- Store `run_state` as a dict rather than a `RunState` instance to keep the snapshot store totally JSON-focused.
- Reconstructing `RunState` can reuse the same “enum reconstruction” logic used by `JsonFileRunStore.load()`.

---

### Storage interface

```python
from abc import ABC, abstractmethod
from typing import List, Optional

class SnapshotStore(ABC):
    @abstractmethod
    def save(self, snapshot: "Snapshot") -> None: ...

    @abstractmethod
    def load(self, snapshot_id: str) -> Optional["Snapshot"]: ...

    @abstractmethod
    def delete(self, snapshot_id: str) -> bool: ...

    @abstractmethod
    def list(
        self,
        *,
        run_id: Optional[str] = None,
        tag: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> List["Snapshot"]:
        """Basic search: by run_id, single tag, substring match in name/description."""
```

We deliberately keep search minimal (MVP):
- `tag`: single tag filter (AND)
- `query`: substring match (case-insensitive) on `name` and `description`

---

### JSON backend (MVP)

Implement `JsonSnapshotStore` similar to `JsonFileRunStore`:
- Base directory: `snapshots/`
- One file per snapshot: `snapshot_{snapshot_id}.json`

Search approach:
- Scan directory and load headers (or whole file) for each snapshot.
- This is fine for MVP; if it becomes large, add an index later.

Implementation details:
- Use `dataclasses.asdict(snapshot)` for persistence.
- Ensure tags are always a list of strings.

---

### Helper: create snapshot from run

Keep this out of the core runtime. Provide helper in `storage/snapshots.py` or `integrations/.../factory.py`.

Pseudo:

```python
from dataclasses import asdict


def create_snapshot_for_run(
    *,
    snapshot_store: SnapshotStore,
    run_store: RunStore,
    run_id: str,
    name: str,
    description: str | None = None,
    tags: list[str] | None = None,
    step_id: str | None = None,
) -> Snapshot:
    run = run_store.load(run_id)
    if run is None:
        raise KeyError(f"Unknown run_id: {run_id}")

    snap = Snapshot.from_run_state(
        run_state=asdict(run),
        run_id=run_id,
        step_id=step_id,
        name=name,
        description=description,
        tags=tags,
    )
    snapshot_store.save(snap)
    return snap
```

---

### Restore semantics

Restoring a snapshot is **not** automatically “safe” if your workflow spec has changed.
MVP semantics:
- The host loads `Snapshot.run_state` and uses it to replace the current `RunState` checkpoint.
- If the workflow spec differs, behavior is undefined. This is acceptable for v0.1 (document it).

---

### Deliverable checklist

- [ ] `Snapshot` model created
- [ ] `SnapshotStore` interface created
- [ ] `JsonSnapshotStore` implemented (file-per-snapshot)
- [ ] Basic search: by tag + substring in name/description
- [ ] Roundtrip: save → load preserves snapshot data


