"""A bounded record of which ids changed after a counter value (stores' change signals).

Used by `JsonFileRunStore.changed_run_ids` and `ObservableLedgerStore.changed_run_ids`."""

from __future__ import annotations

from collections import deque
from typing import Any, Optional


class ChangeRing:
    """The ids behind the last `maxlen` bumps of a store's change counter.

    `since(seq)` answers "which ids changed after counter value `seq`?", or None when entries
    after it were evicted (the caller then falls back to a full read). Not thread-safe: callers
    hold the lock that guards their counter."""

    def __init__(self, maxlen: int = 8192) -> None:
        self._ring: Any = deque(maxlen=int(maxlen))
        self._floor = 0  # the highest evicted seq: answers are complete only after it

    def note(self, seq: int, rid: str) -> None:
        if len(self._ring) == self._ring.maxlen:
            self._floor = int(self._ring[0][0])
        self._ring.append((int(seq), rid))

    def since(self, seq: int) -> Optional[set]:
        if int(seq) < self._floor:
            return None
        return {rid for s, rid in self._ring if s > int(seq)}
