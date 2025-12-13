"""abstractruntime.storage.json_files

Simple file-based persistence:
- RunState checkpoints as JSON (one file per run)
- Ledger as JSONL (append-only)

This is meant as a straightforward MVP backend.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import LedgerStore, RunStore
from ..core.models import RunState, StepRecord, RunStatus, WaitState, WaitReason


class JsonFileRunStore(RunStore):
    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self._base / f"run_{run_id}.json"

    def save(self, run: RunState) -> None:
        p = self._path(run.run_id)
        with p.open("w", encoding="utf-8") as f:
            json.dump(asdict(run), f, ensure_ascii=False, indent=2)

    def load(self, run_id: str) -> Optional[RunState]:
        p = self._path(run_id)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

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
        )


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


