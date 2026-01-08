"""abstractruntime.storage.json_files

Simple file-based persistence:
- RunState checkpoints as JSON (one file per run)
- Ledger as JSONL (append-only)

This is meant as a straightforward MVP backend.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import LedgerStore, RunStore
from ..core.models import RunState, StepRecord, RunStatus, WaitState, WaitReason


class JsonFileRunStore(RunStore):
    """File-based run store with query support.

    Implements both RunStore (ABC) and QueryableRunStore (Protocol).

    Query operations scan all run_*.json files, which is acceptable for MVP
    but needs lightweight indexing for interactive workloads (e.g. WS tick loops)
    once the run directory grows.
    """

    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._index_lock = threading.Lock()
        self._children_index: Optional[Dict[str, set[str]]] = None
        self._run_parent_index: Dict[str, Optional[str]] = {}

    def _path(self, run_id: str) -> Path:
        return self._base / f"run_{run_id}.json"

    def _ensure_children_index(self) -> None:
        if self._children_index is not None:
            return
        with self._index_lock:
            if self._children_index is not None:
                return

            children: Dict[str, set[str]] = {}
            run_parent: Dict[str, Optional[str]] = {}

            for run in self._iter_all_runs():
                parent = run.parent_run_id
                run_parent[run.run_id] = parent
                if isinstance(parent, str) and parent:
                    children.setdefault(parent, set()).add(run.run_id)

            self._children_index = children
            self._run_parent_index = run_parent

    def _drop_from_children_index(self, run_id: str) -> None:
        with self._index_lock:
            if self._children_index is None:
                return
            parent = self._run_parent_index.pop(run_id, None)
            if isinstance(parent, str) and parent:
                siblings = self._children_index.get(parent)
                if siblings is not None:
                    siblings.discard(run_id)
                    if not siblings:
                        self._children_index.pop(parent, None)

    def _update_children_index_on_save(self, run: RunState) -> None:
        run_id = run.run_id
        new_parent = run.parent_run_id

        with self._index_lock:
            if self._children_index is None:
                return

            old_parent = self._run_parent_index.get(run_id)
            if isinstance(old_parent, str) and old_parent and old_parent != new_parent:
                siblings = self._children_index.get(old_parent)
                if siblings is not None:
                    siblings.discard(run_id)
                    if not siblings:
                        self._children_index.pop(old_parent, None)

            self._run_parent_index[run_id] = new_parent
            if isinstance(new_parent, str) and new_parent:
                self._children_index.setdefault(new_parent, set()).add(run_id)

    def save(self, run: RunState) -> None:
        p = self._path(run.run_id)
        # Atomic write to prevent corrupted/partial JSON when multiple threads/processes
        # (e.g. WS tick loop + UI pause/cancel) write the same run file concurrently.
        tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(asdict(run), f, ensure_ascii=False, indent=2)
            tmp.replace(p)
        finally:
            # Best-effort cleanup if replace() failed.
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
        self._update_children_index_on_save(run)

    def load(self, run_id: str) -> Optional[RunState]:
        p = self._path(run_id)
        if not p.exists():
            return None
        return self._load_from_path(p)

    def _load_from_path(self, p: Path) -> Optional[RunState]:
        """Load a RunState from a file path."""
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

        # Reconstruct enums and nested dataclasses
        raw_status = data.get("status")
        status = raw_status if isinstance(raw_status, RunStatus) else RunStatus(str(raw_status))

        waiting: Optional[WaitState] = None
        raw_waiting = data.get("waiting")
        if isinstance(raw_waiting, dict):
            raw_reason = raw_waiting.get("reason")
            if raw_reason is None:
                raise ValueError("Persisted waiting state missing 'reason'")
            reason = raw_reason if isinstance(raw_reason, WaitReason) else WaitReason(str(raw_reason))
            waiting = WaitState(
                reason=reason,
                wait_key=raw_waiting.get("wait_key"),
                until=raw_waiting.get("until"),
                resume_to_node=raw_waiting.get("resume_to_node"),
                result_key=raw_waiting.get("result_key"),
                prompt=raw_waiting.get("prompt"),
                choices=raw_waiting.get("choices"),
                allow_free_text=bool(raw_waiting.get("allow_free_text", True)),
                details=raw_waiting.get("details"),
            )

        return RunState(
            run_id=data["run_id"],
            workflow_id=data["workflow_id"],
            status=status,
            current_node=data["current_node"],
            vars=data.get("vars") or {},
            waiting=waiting,
            output=data.get("output"),
            error=data.get("error"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            actor_id=data.get("actor_id"),
            session_id=data.get("session_id"),
            parent_run_id=data.get("parent_run_id"),
        )

    def _iter_all_runs(self) -> List[RunState]:
        """Iterate over all stored runs."""
        runs: List[RunState] = []
        for p in self._base.glob("run_*.json"):
            run = self._load_from_path(p)
            if run is not None:
                runs.append(run)
        return runs

    # --- QueryableRunStore methods ---

    def list_runs(
        self,
        *,
        status: Optional[RunStatus] = None,
        wait_reason: Optional[WaitReason] = None,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[RunState]:
        """List runs matching the given filters."""
        results: List[RunState] = []

        for run in self._iter_all_runs():
            # Apply filters
            if status is not None and run.status != status:
                continue
            if workflow_id is not None and run.workflow_id != workflow_id:
                continue
            if wait_reason is not None:
                if run.waiting is None or run.waiting.reason != wait_reason:
                    continue

            results.append(run)

        # Sort by updated_at descending (most recent first)
        results.sort(key=lambda r: r.updated_at or "", reverse=True)

        return results[:limit]

    def list_due_wait_until(
        self,
        *,
        now_iso: str,
        limit: int = 100,
    ) -> List[RunState]:
        """List runs waiting for a time threshold that has passed."""
        results: List[RunState] = []

        for run in self._iter_all_runs():
            # Must be WAITING with reason UNTIL
            if run.status != RunStatus.WAITING:
                continue
            if run.waiting is None:
                continue
            if run.waiting.reason != WaitReason.UNTIL:
                continue
            if run.waiting.until is None:
                continue

            # Check if the wait time has passed (ISO string comparison works for UTC)
            if run.waiting.until <= now_iso:
                results.append(run)

        # Sort by waiting.until ascending (earliest due first)
        results.sort(key=lambda r: r.waiting.until if r.waiting else "")

        return results[:limit]

    def list_children(
        self,
        *,
        parent_run_id: str,
        status: Optional[RunStatus] = None,
    ) -> List[RunState]:
        """List child runs of a parent."""
        self._ensure_children_index()
        with self._index_lock:
            child_ids = list((self._children_index or {}).get(parent_run_id, set()))

        results: List[RunState] = []
        for run_id in sorted(child_ids):
            run = self.load(run_id)
            if run is None:
                self._drop_from_children_index(run_id)
                continue
            if status is not None and run.status != status:
                continue
            results.append(run)

        return results


class JsonlLedgerStore(LedgerStore):
    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self._base / f"ledger_{run_id}.jsonl"

    def append(self, record: StepRecord) -> None:
        p = self._path(record.run_id)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False))
            f.write("\n")

    def list(self, run_id: str) -> List[Dict[str, Any]]:
        p = self._path(run_id)
        if not p.exists():
            return []
        out: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
