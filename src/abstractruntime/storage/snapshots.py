"""abstractruntime.storage.snapshots

Named snapshots/bookmarks for durable run state.

A snapshot is a *user/operator-facing checkpoint* with:
- name/description/tags
- timestamps
- embedded (JSON) run state

This is intentionally simple for v0.1 (file-per-snapshot).
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.models import RunState


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Snapshot:
    snapshot_id: str
    run_id: str

    step_id: Optional[str] = None

    name: str = ""
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    # JSON-only representation of the run state
    run_state: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run(
        cls,
        *,
        run: RunState,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        step_id: Optional[str] = None,
    ) -> "Snapshot":
        return cls(
            snapshot_id=str(uuid.uuid4()),
            run_id=run.run_id,
            step_id=step_id,
            name=name,
            description=description,
            tags=list(tags or []),
            run_state=asdict(run),
        )


class SnapshotStore(ABC):
    @abstractmethod
    def save(self, snapshot: Snapshot) -> None: ...

    @abstractmethod
    def load(self, snapshot_id: str) -> Optional[Snapshot]: ...

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
    ) -> List[Snapshot]: ...


class InMemorySnapshotStore(SnapshotStore):
    def __init__(self):
        self._snaps: Dict[str, Snapshot] = {}

    def save(self, snapshot: Snapshot) -> None:
        snapshot.updated_at = utc_now_iso()
        self._snaps[snapshot.snapshot_id] = snapshot

    def load(self, snapshot_id: str) -> Optional[Snapshot]:
        return self._snaps.get(snapshot_id)

    def delete(self, snapshot_id: str) -> bool:
        return self._snaps.pop(snapshot_id, None) is not None

    def list(
        self,
        *,
        run_id: Optional[str] = None,
        tag: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> List[Snapshot]:
        snaps = list(self._snaps.values())
        return _filter_snapshots(snaps, run_id=run_id, tag=tag, query=query, limit=limit)


class JsonSnapshotStore(SnapshotStore):
    """File-based snapshot store (one JSON file per snapshot)."""

    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _path(self, snapshot_id: str) -> Path:
        return self._base / f"snapshot_{snapshot_id}.json"

    def save(self, snapshot: Snapshot) -> None:
        snapshot.updated_at = utc_now_iso()
        p = self._path(snapshot.snapshot_id)
        with p.open("w", encoding="utf-8") as f:
            json.dump(asdict(snapshot), f, ensure_ascii=False, indent=2)

    def load(self, snapshot_id: str) -> Optional[Snapshot]:
        p = self._path(snapshot_id)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return Snapshot(
            snapshot_id=data["snapshot_id"],
            run_id=data["run_id"],
            step_id=data.get("step_id"),
            name=data.get("name") or "",
            description=data.get("description"),
            tags=list(data.get("tags") or []),
            created_at=data.get("created_at") or utc_now_iso(),
            updated_at=data.get("updated_at") or utc_now_iso(),
            run_state=dict(data.get("run_state") or {}),
        )

    def delete(self, snapshot_id: str) -> bool:
        p = self._path(snapshot_id)
        if not p.exists():
            return False
        p.unlink()
        return True

    def list(
        self,
        *,
        run_id: Optional[str] = None,
        tag: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> List[Snapshot]:
        snaps: List[Snapshot] = []
        for p in sorted(self._base.glob("snapshot_*.json")):
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                snaps.append(
                    Snapshot(
                        snapshot_id=data["snapshot_id"],
                        run_id=data["run_id"],
                        step_id=data.get("step_id"),
                        name=data.get("name") or "",
                        description=data.get("description"),
                        tags=list(data.get("tags") or []),
                        created_at=data.get("created_at") or utc_now_iso(),
                        updated_at=data.get("updated_at") or utc_now_iso(),
                        run_state=dict(data.get("run_state") or {}),
                    )
                )
            except Exception:
                # Corrupt snapshot files are ignored for now (MVP); verification tools can be added later.
                continue

        return _filter_snapshots(snaps, run_id=run_id, tag=tag, query=query, limit=limit)


def _filter_snapshots(
    snaps: List[Snapshot],
    *,
    run_id: Optional[str],
    tag: Optional[str],
    query: Optional[str],
    limit: int,
) -> List[Snapshot]:
    out = snaps

    if run_id:
        out = [s for s in out if s.run_id == run_id]

    if tag:
        tag_l = tag.lower()
        out = [s for s in out if any(t.lower() == tag_l for t in (s.tags or []))]

    if query:
        q = query.lower()
        def _matches(s: Snapshot) -> bool:
            name = (s.name or "").lower()
            desc = (s.description or "").lower()
            return q in name or q in desc
        out = [s for s in out if _matches(s)]

    # Newest first (best-effort ISO ordering)
    out.sort(key=lambda s: s.created_at or "", reverse=True)
    return out[: max(0, int(limit))]

