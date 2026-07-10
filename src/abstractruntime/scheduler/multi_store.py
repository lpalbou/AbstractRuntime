"""Fair scheduling across N run stores (plan item 11, R3 — a2a 0018 spec).

One process may host many runtimes, each bound to its OWN run store (the
per-entity topology: N entities = N `Runtime` instances over N stores;
project workplaces are the coming second consumer). The flagship
`Scheduler` polls ONE store; this module is the multi-store layer agreed
on thread 0018 (all sign-offs in: agency P3 commons 124, gateway GW-D —
registry lives RUNTIME-side, the door feeds it via its entity registry;
core C4 consulted-only, commons 423):

- `TickCandidate` — the store-agnostic unit of schedulable work.
- `TickSource` — wraps ONE store's due-scan (`list_due_wait_until`, which
  surfaces UNTIL waits and EVENT waits carrying a deadline) and stamps
  `store` + `channel`. The channel is the door's STAMPED channel read
  from run vars; the source never invents it — absent = "unstamped",
  which sorts as background. Label, never guess.
- `AdmissionHook` — host-injectable ORDERING policy (the door owns
  priority: visit outranks background tick). The runtime owns the one
  mechanical guarantee policy cannot remove: the STARVATION FLOOR — any
  candidate waiting longer than `max_starvation_s` is promoted ahead of
  policy order (weighted-fair, never strict priority; the fleet review's
  "strict visit>tick>dream starves dreams fleet-wide" finding, made a
  floor the hook composes with instead of a norm it must remember).
- `MultiStoreScheduler` — a registry of named sources; one poll loop:
  union due candidates -> admission (hook, then floor) -> tick each run
  through ITS OWN runtime. Per-tick errors are PER-RUN (one failing home
  never stalls the sweep); stats name per-store counts so starvation is
  visible as data. This is also the EAGER deadline sweep the request-
  driven door lacks: registering the entity stores makes parked-visit
  idle deadlines (D3) fire without a client touch.

Pins carried from the spec (so nobody re-litigates):
- Run-id -> store resolution stays BEHIND this API: the registry is host
  machinery; clients never name a store.
- Admission governs traffic THROUGH this point only — it schedules
  TICKS, not tokens; the LLM ceiling is core C3/C4's lane, and budget
  exhaustion arrives as an ordinary loud per-run failure (core c423),
  never back-pressure into the hook.
- ADMISSION-TICKET-BEFORE-LEASE (agency P3): the hook orders candidates
  BEFORE any tick driver acquires a directory lease — per-store `ticker`
  callables (the door's lease-acquiring drive) run strictly after
  admission ordering.
- R4 needs NO new machinery here: the steering sidecar is per-run and
  drains on the owning tick; N stores = N independently-drained hooks by
  construction.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Protocol

from ..core.models import RunState, WaitReason
from ..core.runtime import Runtime
from ..core.spec import WorkflowSpec

logger = logging.getLogger(__name__)

UNSTAMPED_CHANNEL = "unstamped"
DEFAULT_MAX_STARVATION_S = 300.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def stamped_channel(run: RunState) -> str:
    """The door's stamped channel from run vars — read, never invented.

    The summon stamp rides `_runtime.entity.channel` (gateway door); runs
    without a stamp (workplace flows, tests) are honestly "unstamped" and
    sort as background."""
    vars_ = getattr(run, "vars", None)
    if isinstance(vars_, dict):
        rn = vars_.get("_runtime")
        if isinstance(rn, dict):
            entity = rn.get("entity")
            if isinstance(entity, dict):
                channel = str(entity.get("channel") or "").strip()
                if channel:
                    return channel
    return UNSTAMPED_CHANNEL


@dataclasses.dataclass(frozen=True)
class TickCandidate:
    """One schedulable unit of due work, store-agnostic."""

    store: str
    run_id: str
    workflow_id: str
    wait_kind: str  # "until" | "event_deadline"
    due_at: str  # ISO deadline that made it due
    channel: str  # door-stamped channel; "unstamped" when absent
    waiting_since: str  # run.updated_at when the scan saw it


class AdmissionHook(Protocol):
    """Host-injectable ordering policy over one sweep's candidates.

    Receives the UNION of every registered store's due candidates and
    returns them in tick order. Dropping candidates is allowed (they
    reappear next sweep — deferral, not loss). The starvation floor runs
    AFTER the hook and can only PROMOTE, never demote or drop."""

    def admit(self, candidates: List[TickCandidate]) -> List[TickCandidate]: ...


class DueOrderAdmission:
    """The default policy: due-order (oldest deadline first) — today's
    single-store behavior generalized across stores."""

    def admit(self, candidates: List[TickCandidate]) -> List[TickCandidate]:
        return sorted(candidates, key=lambda c: c.due_at)


def apply_starvation_floor(
    ordered: List[TickCandidate],
    all_candidates: List[TickCandidate],
    *,
    max_starvation_s: float,
    now: Optional[datetime] = None,
) -> List[TickCandidate]:
    """The runtime's mechanical guarantee, composed OVER any policy:

    - Candidates the hook DROPPED but that have starved past the floor are
      restored (a policy may defer work, never bury it).
    - Starved candidates are PROMOTED ahead of the policy's order, oldest
      wait first; the policy's relative order is preserved for the rest.
    """
    now_dt = now or datetime.now(timezone.utc)

    def starved(c: TickCandidate) -> bool:
        since = _parse_iso(c.waiting_since)
        if since is None:
            return False
        return (now_dt - since).total_seconds() > float(max_starvation_s)

    ordered_ids = {c.run_id for c in ordered}
    dropped_starved = [c for c in all_candidates if c.run_id not in ordered_ids and starved(c)]

    promoted: List[TickCandidate] = []
    rest: List[TickCandidate] = []
    for c in ordered:
        (promoted if starved(c) else rest).append(c)
    promoted.extend(dropped_starved)
    promoted.sort(key=lambda c: c.waiting_since)  # oldest starvation first
    return promoted + rest


@dataclasses.dataclass
class TickSource:
    """One store's due-scan, stamped with the store's registry name.

    `runtime` owns the store; `resolve_workflow(run)` returns the spec to
    tick with (the door rebuilds visit specs from the run's own stamp);
    `ticker(run, workflow)` is the drive callable — defaults to
    `runtime.tick`, and the door registers its lease-acquiring driver
    here (admission-ticket-BEFORE-lease: ordering happened already)."""

    name: str
    runtime: Runtime
    resolve_workflow: Callable[[RunState], WorkflowSpec]
    ticker: Optional[Callable[[RunState, WorkflowSpec], RunState]] = None

    def due_candidates(self, *, now_iso: str, limit: int = 100) -> List[TickCandidate]:
        runs = self.runtime.run_store.list_due_wait_until(now_iso=now_iso, limit=limit)
        out: List[TickCandidate] = []
        for run in runs:
            waiting = getattr(run, "waiting", None)
            if waiting is None:
                continue
            kind = "until" if waiting.reason == WaitReason.UNTIL else "event_deadline"
            out.append(
                TickCandidate(
                    store=self.name,
                    run_id=run.run_id,
                    workflow_id=run.workflow_id,
                    wait_kind=kind,
                    due_at=str(waiting.until or ""),
                    channel=stamped_channel(run),
                    waiting_since=str(getattr(run, "updated_at", "") or ""),
                )
            )
        return out

    def tick(self, run: RunState) -> RunState:
        workflow = self.resolve_workflow(run)
        if self.ticker is not None:
            return self.ticker(run, workflow)
        return self.runtime.tick(workflow=workflow, run_id=run.run_id)


@dataclasses.dataclass
class StoreSweepStats:
    candidates_seen: int = 0
    ticked: int = 0
    failures: int = 0


@dataclasses.dataclass
class MultiSweepStats:
    sweeps: int = 0
    promoted_by_floor: int = 0
    last_sweep_at: Optional[str] = None
    per_store: Dict[str, StoreSweepStats] = dataclasses.field(default_factory=dict)
    errors: List[str] = dataclasses.field(default_factory=list)

    def store(self, name: str) -> StoreSweepStats:
        if name not in self.per_store:
            self.per_store[name] = StoreSweepStats()
        return self.per_store[name]


class MultiStoreScheduler:
    """One poll loop over N registered stores: union -> admit -> tick.

    Deliberately SMALL: no workflow registry of its own (each source
    resolves specs), no event API (the flagship `Scheduler` and the door
    own resume paths), no priority policy (the hook is the host's). What
    it guarantees mechanically: every store is scanned every sweep, the
    starvation floor holds over any policy, and one run's failure never
    stalls the sweep (per-run try; per-store stats)."""

    def __init__(
        self,
        *,
        admission: Optional[AdmissionHook] = None,
        max_starvation_s: float = DEFAULT_MAX_STARVATION_S,
        poll_interval_s: float = 1.0,
        scan_limit_per_store: int = 100,
        max_errors_kept: int = 100,
    ) -> None:
        self._sources: Dict[str, TickSource] = {}
        self._admission: AdmissionHook = admission or DueOrderAdmission()
        self._max_starvation_s = float(max_starvation_s)
        self._poll_interval = float(poll_interval_s)
        self._scan_limit = int(scan_limit_per_store)
        self._max_errors = int(max_errors_kept)
        self._stats = MultiSweepStats()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------ registry
    def register(self, source: TickSource) -> None:
        """Register a named store. Names are the door's registry keys
        (slugs) — relocation-stable, never addresses. Duplicate names
        refuse loudly (two sources over one name would double-tick)."""
        name = str(source.name or "").strip()
        if not name:
            raise ValueError("TickSource.name must be non-empty")
        with self._lock:
            if name in self._sources:
                raise ValueError(f"a TickSource named {name!r} is already registered")
            self._sources[name] = source

    def unregister(self, name: str) -> None:
        with self._lock:
            self._sources.pop(str(name), None)

    @property
    def stats(self) -> MultiSweepStats:
        return self._stats

    # --------------------------------------------------------------- sweep
    def sweep_once(self, *, now_iso: Optional[str] = None) -> int:
        """One sweep: union due candidates, order (hook then floor), tick
        each through its own source. Returns the number ticked."""
        now = now_iso or _utc_now_iso()
        self._stats.sweeps += 1
        self._stats.last_sweep_at = now
        with self._lock:
            sources = dict(self._sources)

        candidates: List[TickCandidate] = []
        runs_by_id: Dict[str, RunState] = {}
        for name, source in sources.items():
            try:
                due = source.due_candidates(now_iso=now, limit=self._scan_limit)
            except Exception as e:  # noqa: BLE001 - one store's scan failure never stalls the sweep
                self._stats.store(name).failures += 1
                self._record_error(f"store {name}: due-scan failed: {e}")
                continue
            self._stats.store(name).candidates_seen += len(due)
            candidates.extend(due)
            # The tick needs the RunState; re-load through the source's
            # runtime so each store answers only for its own runs.
            for c in due:
                try:
                    run = source.runtime.get_state(c.run_id)
                except Exception as e:  # noqa: BLE001
                    self._record_error(f"store {name}: load {c.run_id} failed: {e}")
                    continue
                runs_by_id[c.run_id] = run

        if not candidates:
            return 0

        try:
            ordered = list(self._admission.admit(list(candidates)))
        except Exception as e:  # noqa: BLE001 - a broken policy degrades to due-order, loudly
            self._record_error(f"#FALLBACK admission hook failed ({e}); due-order used")
            ordered = DueOrderAdmission().admit(list(candidates))

        floored = apply_starvation_floor(
            ordered, candidates, max_starvation_s=self._max_starvation_s
        )
        promoted = len(floored) - len(ordered)
        if promoted > 0:
            self._stats.promoted_by_floor += promoted

        ticked = 0
        for c in floored:
            source = sources.get(c.store)
            run = runs_by_id.get(c.run_id)
            if source is None or run is None:
                continue
            try:
                source.tick(run)
                self._stats.store(c.store).ticked += 1
                ticked += 1
            except Exception as e:  # noqa: BLE001 - per-run isolation: the sweep continues
                self._stats.store(c.store).failures += 1
                self._record_error(f"store {c.store}: tick {c.run_id} failed: {e}")
        return ticked

    # ---------------------------------------------------------- background
    def start(self) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("MultiStoreScheduler is already running")
            self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="abstractruntime-multi-scheduler", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.sweep_once()
            except Exception as e:  # noqa: BLE001 - the loop survives its own bugs, loudly
                logger.exception("multi-store sweep failed: %s", e)
                self._record_error(f"sweep failed: {e}")
            self._stop_event.wait(timeout=self._poll_interval)

    def _record_error(self, message: str) -> None:
        logger.error("%s", message)
        self._stats.errors.append(f"{_utc_now_iso()}: {message}")
        if len(self._stats.errors) > self._max_errors:
            self._stats.errors = self._stats.errors[-self._max_errors :]
