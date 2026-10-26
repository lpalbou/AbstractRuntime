"""The active-consumers index: which ACTIVE automations read what (one per runtime).

Every "who listens?" question — `email.stream_consumers` / `wake_stream_consumers`,
`email_trigger_consumers` / `wake_email_automations`, `triggers.connection_usage` (the
connections listing and every feed poll), `connection_trigger_consumers`,
`prune_email_inbox` — is answered from ONE snapshot built by one pass over the controller
runs: `{automation_id: trigger binding}` for the automations whose status is `active`
(`models.automation_status`: not paused, archived, completed or failed). A query is a dict
lookup or a filter over the active bindings: microseconds, no file read.

Before (measured 2026-10-26 on the operator's gateway): every call listed the whole run
index (one stat per run file, 11,499 files on the operator's plane) and loaded every
controller run, once per connection row, per feed poll, per mailbox poll and per wake:
15-200 s for the connections listing beside a busy process, 190% CPU while idle.

Rebuilt only when something changed:

- the CHANGE SIGNAL is the automation generation of the run store
  (`invalidate_active_consumers`), one dict read per query. It is bumped where the answer
  can change: a JSON-file run store bumps it on every save whose controller row
  (`automation_consumer`: the automation's status and trigger) is new or differs from the
  one it replaces — so a controller created, paused, resumed, revised, archived, revived,
  failed or cancelled through any path of this process is seen at once, and a run's
  ordinary saves (an occurrence started, a cursor moved) bump nothing. Explicitly, for every
  store: `create_automation`, every applied automation command (after its follow-ups, so an
  unarchive's revived controller is seen), and any decision or reconciliation whose delta
  changes the definition or the `paused` / `exhausted` state. A directory-wide signal (the
  run index's stat tokens, the number of controller runs) would cost the very scan this
  index removes.
- a bounded TTL (`DEFAULT_TTL_S`, 30 s) is the safety net for what this process cannot see:
  another process writing the same folder (or a store without the save signal changing a
  controller outside the protocol). A missed signal is stale for at most the TTL, and a
  stale answer is harmless where it matters: a wake re-checks the controller (`_wake` skips a
  run that is not parked on its wake wait), and a controller reads its stream at every wake.

A rebuild lists only the RUNNING and WAITING controller index rows
(`list_run_index(role="controller", status=...)`: a JSON-file store answers it from its
active index, one stat per live run instead of one per run file; a completed, failed or
cancelled controller is never active). A JSON-file store's controller rows carry the
automation's status and trigger (`automation_consumer`, `consumer_index_fields`), so the
rebuild reads no run file; a store whose rows do not loads those controllers, once each.

The generation is keyed by the run store's location (its folder or database file), so two
runtime objects over one store in one process invalidate each other.
"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

DEFAULT_TTL_S = 30.0
# Controller run statuses that can still be an active automation (`automation_status`:
# completed -> completed, failed/cancelled -> failed).
_LIVE_RUN_STATUSES = frozenset({"running", "waiting"})

_generation_lock = threading.Lock()
_generations: Dict[Any, int] = {}


def _innermost(store: Any) -> Any:
    cur = store
    for _ in range(8):
        inner = getattr(cur, "_inner", None)
        if inner is None:
            return cur
        cur = inner
    return cur


def store_key(run_store: Any) -> Any:
    """The identity a generation is kept under: the store's folder / database file when it
    has one (`JsonFileRunStore._base`, `SqliteRunStore._db._path`), else the store object."""
    inner = _innermost(run_store)
    for path in (getattr(inner, "_base", None), getattr(getattr(inner, "_db", None), "_path", None)):
        if isinstance(path, (str, Path)):
            try:
                return ("path", str(Path(path).resolve()))
            except OSError:  # pragma: no cover - resolve() on a vanished folder
                return ("path", str(path))
    return ("object", id(inner))


def _generation(key: Any) -> int:
    return _generations.get(key, 0)


def invalidate_active_consumers(target: Any) -> None:
    """Mark every active-consumers index over this run store stale (pass a runtime or a
    run store). Called by the automation write paths; cheap (one locked increment)."""
    run_store = getattr(target, "run_store", target)
    if run_store is None:
        return
    key = store_key(run_store)
    with _generation_lock:
        _generations[key] = _generations.get(key, 0) + 1


def automation_generation(target: Any) -> int:
    """The current automation generation of this run store (pass a runtime or a run store): a
    counter that moves whenever `invalidate_active_consumers` is called for it. Readers that
    cache something derived from the set of automations or their status (the gateway's
    automations listing) key it on this value."""
    run_store = getattr(target, "run_store", target)
    if run_store is None:
        return 0
    with _generation_lock:
        return _generation(store_key(run_store))


def delta_changes_consumers(delta: Optional[Mapping[str, Any]]) -> bool:
    """Whether a decision delta can change the index: a new definition (trigger, archive
    mark) or the `paused` / `exhausted` state."""
    if not isinstance(delta, Mapping):
        return False
    if "definition" in delta:
        return True
    state = delta.get("state")
    return isinstance(state, Mapping) and ("paused" in state or "exhausted" in state)


def consumer_index_fields(run: Any) -> Dict[str, Any]:
    """`{status, trigger}` of a controller run: `automation_status(run)` and its trigger binding
    (None when the definition has none). A JSON-file run store keeps it on the controller's
    index row (`automation_consumer`, validated by the file's identity like every memo field),
    so a rebuild reads index rows only."""
    from .ledger import definition_of
    from .models import automation_status

    try:
        trigger = definition_of(run).get("trigger")
    except LookupError:
        trigger = None
    return {
        "status": automation_status(run),
        "trigger": copy.deepcopy(dict(trigger)) if isinstance(trigger, Mapping) else None,
    }


@dataclass(frozen=True)
class ConsumersSnapshot:
    """The active automations and their trigger bindings at one generation."""

    bindings: Tuple[Tuple[str, Mapping[str, Any]], ...]  # (automation_id, trigger), id order
    by_connection: Mapping[str, Tuple[str, ...]]  # connection_id -> automation ids
    generation: int
    built_at: float
    store: Any = field(compare=False, repr=False, default=None)

    def automation_ids(self, wanted: Optional[Callable[[Mapping[str, Any]], bool]] = None) -> List[str]:
        """Ids of the active automations whose binding passes `wanted` (all when None), sorted."""
        if wanted is None:
            return [aid for aid, _t in self.bindings]
        return [aid for aid, trigger in self.bindings if wanted(trigger)]

    def connection_consumers(self, connection_id: Optional[str] = None) -> List[str]:
        """Ids of the active automations bound to connection `connection_id` (every
        connection-bound one when None), sorted."""
        if connection_id is not None:
            return list(self.by_connection.get(str(connection_id), ()))
        return sorted(aid for ids in self.by_connection.values() for aid in ids)

    def stream_consumers(self, stream: str, *, self_account_ref: Optional[str] = None) -> List[str]:
        """Ids of the active automations that read inbox `stream` (`automation_reads_stream`)."""
        from ..email.automations import TRIGGER_STREAM_PREFIX, automation_reads_stream

        if isinstance(stream, str) and stream.startswith(TRIGGER_STREAM_PREFIX):
            # Only a connection-bound binding reads a `trigger:<connection_id>` stream.
            return list(self.by_connection.get(stream[len(TRIGGER_STREAM_PREFIX):], ()))
        return [aid for aid, trigger in self.bindings if automation_reads_stream(trigger, stream, self_account_ref=self_account_ref)]


def _build(run_store: Any, *, generation: int, now: float) -> ConsumersSnapshot:
    from ..core.models import RunStatus
    from .ledger import definition_of
    from .models import automation_status

    bindings: List[Tuple[str, Mapping[str, Any]]] = []
    rows: List[Mapping[str, Any]] = []
    # Only a RUNNING or WAITING controller can be an active automation (completed ->
    # completed, failed/cancelled -> failed): asked by status, a JSON-file store answers from
    # its ACTIVE index (the live runs only) instead of one stat per run file.
    for status in (RunStatus.RUNNING, RunStatus.WAITING):
        rows.extend(run_store.list_run_index(limit=1_000_000, role="controller", status=status))
    seen: set = set()
    for row in rows:
        rid = str(row.get("run_id"))
        if rid in seen or str(row.get("status")) not in _LIVE_RUN_STATUSES:
            continue
        seen.add(rid)
        consumer = row.get("automation_consumer")
        if isinstance(consumer, Mapping):  # the row carries it (JSON-file store): no load
            trigger = consumer.get("trigger")
            if consumer.get("status") == "active" and isinstance(trigger, Mapping):
                bindings.append((rid, copy.deepcopy(dict(trigger))))
            continue
        run = run_store.load(rid)
        if run is None:
            continue
        try:
            trigger = definition_of(run)["trigger"]
        except (LookupError, KeyError, TypeError):
            continue
        if not isinstance(trigger, Mapping) or automation_status(run) != "active":
            continue
        bindings.append((str(run.run_id), copy.deepcopy(dict(trigger))))
    bindings.sort(key=lambda pair: pair[0])
    by_connection: Dict[str, List[str]] = {}
    for aid, trigger in bindings:
        config = trigger.get("config")
        cid = config.get("connection_id") if isinstance(config, Mapping) else None
        if cid:
            by_connection.setdefault(str(cid), []).append(aid)
    return ConsumersSnapshot(
        bindings=tuple(bindings),
        by_connection={cid: tuple(ids) for cid, ids in by_connection.items()},
        generation=generation,
        built_at=now,
        store=run_store,
    )


class ActiveConsumersIndex:
    """One runtime's active-consumers index (see the module doc). `snapshot()` returns the
    current `ConsumersSnapshot`, rebuilding it first when the run store's automation
    generation moved, the TTL elapsed, or the runtime's run store changed. Concurrent
    callers wait for one rebuild instead of each running their own. `rebuilds` counts them."""

    def __init__(self, runtime: Any, *, ttl_s: float = DEFAULT_TTL_S, clock: Callable[[], float] = time.monotonic) -> None:
        self._runtime = runtime
        self.ttl_s = float(ttl_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._snapshot: Optional[ConsumersSnapshot] = None
        self._key: Optional[Tuple[int, Any]] = None  # (id(run_store), store_key)
        self.rebuilds = 0

    def _store_key(self, run_store: Any) -> Any:
        cached = self._key
        if cached is not None and cached[0] == id(run_store):
            return cached[1]
        key = store_key(run_store)
        self._key = (id(run_store), key)
        return key

    def _fresh(self, snap: Optional[ConsumersSnapshot], run_store: Any, key: Any) -> bool:
        return (
            snap is not None
            and snap.store is run_store
            and snap.generation == _generation(key)
            and (self._clock() - snap.built_at) < self.ttl_s
        )

    def snapshot(self) -> ConsumersSnapshot:
        run_store = self._runtime.run_store
        key = self._store_key(run_store)
        snap = self._snapshot
        if self._fresh(snap, run_store, key):
            return snap  # type: ignore[return-value]
        with self._lock:
            snap = self._snapshot
            if self._fresh(snap, run_store, key):
                return snap  # type: ignore[return-value]
            # The generation is read BEFORE the scan: a change landing during the scan
            # leaves this snapshot one generation behind, so the next call rebuilds.
            generation = _generation(key)
            snap = _build(run_store, generation=generation, now=self._clock())
            self._snapshot = snap
            self.rebuilds += 1
            return snap

    def invalidate(self) -> None:
        self._snapshot = None


_ATTR = "_active_consumers_index"
_attach_lock = threading.Lock()


def active_consumers_index(runtime: Any) -> ActiveConsumersIndex:
    """The runtime's index, created on first use and kept on the runtime object. A host
    object that refuses attributes gets a fresh (uncached) index per call."""
    index = getattr(runtime, _ATTR, None)
    if isinstance(index, ActiveConsumersIndex):
        return index
    with _attach_lock:
        index = getattr(runtime, _ATTR, None)
        if isinstance(index, ActiveConsumersIndex):
            return index
        index = ActiveConsumersIndex(runtime)
        try:
            setattr(runtime, _ATTR, index)
        except (AttributeError, TypeError):  # pragma: no cover - slotted/frozen host objects
            pass
        return index


def consumers_snapshot(runtime: Any) -> ConsumersSnapshot:
    """`active_consumers_index(runtime).snapshot()`."""
    return active_consumers_index(runtime).snapshot()


__all__ = [
    "ActiveConsumersIndex",
    "ConsumersSnapshot",
    "DEFAULT_TTL_S",
    "active_consumers_index",
    "automation_generation",
    "consumer_index_fields",
    "consumers_snapshot",
    "delta_changes_consumers",
    "invalidate_active_consumers",
    "store_key",
]
