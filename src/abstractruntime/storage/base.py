"""abstractruntime.storage.base

Storage interfaces (durability backends).

These are intentionally minimal for v0.1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core.models import RunState, StepRecord


class RunStore(ABC):
    @abstractmethod
    def save(self, run: RunState) -> None: ...

    @abstractmethod
    def load(self, run_id: str) -> Optional[RunState]: ...


class LedgerStore(ABC):
    """Append-only journal store."""

    @abstractmethod
    def append(self, record: StepRecord) -> None: ...

    @abstractmethod
    def list(self, run_id: str) -> List[Dict[str, Any]]: ...


