"""abstractruntime.storage.in_memory

In-memory durability backends (testing/dev).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .base import LedgerStore, RunStore
from ..core.models import RunState, StepRecord


class InMemoryRunStore(RunStore):
    def __init__(self):
        self._runs: Dict[str, RunState] = {}

    def save(self, run: RunState) -> None:
        # store a shallow copy to avoid accidental mutation surprises
        self._runs[run.run_id] = run

    def load(self, run_id: str) -> Optional[RunState]:
        return self._runs.get(run_id)


class InMemoryLedgerStore(LedgerStore):
    def __init__(self):
        self._records: Dict[str, List[Dict[str, Any]]] = {}

    def append(self, record: StepRecord) -> None:
        self._records.setdefault(record.run_id, []).append(asdict(record))

    def list(self, run_id: str) -> List[Dict[str, Any]]:
        return list(self._records.get(run_id, []))


