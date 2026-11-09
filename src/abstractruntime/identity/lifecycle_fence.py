"""THE Stop fence (STOP-LANE-DESIGN 2026-11-09 §2): one primitive, a closed set of commit points, and the unfenced
write made impossible by signature.

The operator's Stop bumps the home's lifecycle generation (phases.yaml `lifecycle_generation`, under the home state
lock). Everything an entity does is ADMITTED at one generation - a loop at its boot, a turn at its door - and carries
a `LifecycleFence` holding that number. Every commit point (the state writers in `life.py`, memory's commit guard,
the diary and artifact stores, the effect handlers) takes the fence as a REQUIRED argument and refuses once the
generation has moved: a write decided before a Stop never lands after it, whatever process holds it.

The check is an int compare in process plus ONE stat. `_SIGNAL` is a process-wide table, home -> (signature of the
signal file when last read, the newest generation this process knows). The bumping process (the gateway in every
operator Stop) updates it in the same critical section that writes phases.yaml, so its in-flight fences see its own
Stop at their next compare. Every process - the bumper too - sees a Stop by ANY process through
`<home>/.lifecycle_generation`, a one-line file the bump replaces atomically under the same lock, AFTER phases.yaml:
a stat (no open, no read, no lock) at each fenced effect, the int read only when the stat changed. Truth is re-read where the home lock is already held and phases.yaml already parsed (the state
writers, `commit`) and wherever a host reads the generation (`read_lifecycle_generation`), for free.

Fail closed, never "not fenced": an unreadable signal or phases.yaml reads as MOVED, an unknown home as moved, a
signal file that disappears after it was seen as moved. A fenced refusal is the operator's Stop, never a failure:
`StoppedByOperator` is a `TurnCancelled`, which every lane top already maps to a clean "stopped by the operator"
record.

Two writers bypass the check, and only for what a Stop or a Resume writes: `KILL_SWITCH` (the Stop itself: paused,
own time revoked, the return cleared, loop status stopped) and `RESUME` (awake, nothing restored). The public writers
refuse a sentinel for anything else.

No process is exempt from the stat (Fable review F1, 2026-11-09: a bumper that skipped it never saw a later Stop by
another process - a CLI hard_stop_loop, a service twin - until a truth read). It costs about 1-3 us per check.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, TypeVar, Union

from .chat import TurnCancelled

LIFECYCLE_SIGNAL_FILENAME = ".lifecycle_generation"
# The words a lane records for a fenced refusal (the gateway's KILL_SWITCH_REASON is the same sentence).
KILL_SWITCH_REASON = "kill_switch: stopped by the operator"

T = TypeVar("T")
_Signature = Optional[Tuple[int, int, int]]  # (st_mtime_ns, st_size, st_ino) of the signal file; None = absent


class StoppedByOperator(TurnCancelled):
    """The operator's Stop moved the lifecycle generation past the one this work was admitted at: nothing after the
    check that saw it runs or lands. `where` names the boundary; `report` is the turn so far when a turn raised it
    (None from a commit point). Never a failure: lanes record it as stopped by the operator."""

    def __init__(self, where: str, *, report: Any = None, expected: Optional[int] = None,
                 found: Optional[int] = None) -> None:
        TurnCancelled.__init__(self, report, where)
        self.expected = expected
        self.found = found
        if expected is not None:
            self.args = (f"{where} refused: it was decided at lifecycle generation {expected}, the home is at "
                         f"{found if found is not None else 'an unreadable generation'} (the operator's Stop "
                         "happened since)",)
        else:
            self.args = (f"stopped by the operator ({where})",)


class StaleLifecycleGeneration(StoppedByOperator):
    """A state write fenced on a generation that has moved (the name the state writers have raised since f881cac)."""

    def __init__(self, what: str, expected: int, found: int) -> None:
        super().__init__(what, expected=int(expected), found=int(found))


class LifecycleUnreadable(ValueError):
    """The home's lifecycle generation cannot be read (a corrupt phases.yaml or signal file): nothing is admitted."""


class _Sentinel:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name.upper()


KILL_SWITCH = _Sentinel("kill_switch")
RESUME = _Sentinel("resume")
_SENTINELS = (KILL_SWITCH, RESUME)

# ------------------------------------------------------------------------------------------- the process-wide table
_SIGNAL: Dict[str, Tuple[_Signature, int]] = {}
_table_lock = threading.Lock()
# home -> the threading.Events of the model calls in flight under a fence of that home (LifecycleFence.cancel_event):
# the bump sets them in the bumping process, so a provider that honours `cancel_event=` stops within a token (§2.3).
_INFLIGHT: Dict[str, set] = {}


def home_key(home_dir: Union[str, Path]) -> str:
    """The table's key for a home: its resolved path (the same key the home state lock uses)."""
    return str(Path(home_dir).resolve())


def signal_path(home_dir: Union[str, Path]) -> Path:
    return Path(home_dir) / LIFECYCLE_SIGNAL_FILENAME


def _stat_signature(path: str) -> _Signature:
    """The signal file's signature; None when it does not exist. Any other OSError propagates (fail closed)."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _publish(key: str, signature: _Signature, generation: int, *, authoritative: bool) -> None:
    """Record what a read saw. A non-authoritative read never moves the table backwards (a slow reader that parsed
    phases.yaml before a Stop must not undo the Stop another thread already published); an authoritative one (read
    under the home state lock, where no bump can be in flight) is the truth and replaces it."""
    with _table_lock:
        current = _SIGNAL.get(key)
        if authoritative or current is None or generation >= current[1]:
            _SIGNAL[key] = (signature, int(generation))


def _holds_home_lock(key: str) -> bool:
    from .life import _home_state_held

    return bool((getattr(_home_state_held, "depth", None) or {}).get(key))


def read_truth(home_dir: Union[str, Path]) -> int:
    """phases.yaml's lifecycle generation, published into the table. The signal file is stat-ed BEFORE phases.yaml
    is parsed (the bump writes them in the other order), so a published (signature, generation) pair is never newer
    on the signature side than on the generation side. Raises LifecycleUnreadable (a ValueError) when either cannot
    be read."""
    from .life import _generation_of, _load_phases_for_write

    key = home_key(home_dir)
    try:
        signature = _stat_signature(str(signal_path(home_dir)))
    except OSError as e:
        raise LifecycleUnreadable(f"the lifecycle signal of {home_dir} cannot be read ({e})") from e
    try:
        generation = _generation_of(_load_phases_for_write(home_dir, "reading the lifecycle generation"))
    except ValueError as e:
        raise LifecycleUnreadable(str(e)) from e
    _publish(key, signature, generation, authoritative=_holds_home_lock(key))
    return generation


def publish_under_lock(home_dir: Union[str, Path], generation: int) -> None:
    """A writer holding the home state lock parsed phases.yaml and found `generation`: that is the truth now."""
    key = home_key(home_dir)
    try:
        signature = _stat_signature(str(signal_path(home_dir)))
    except OSError:
        with _table_lock:
            _SIGNAL.pop(key, None)  # the signal cannot be seen: every fence of this home reads moved (fail closed)
        return
    _publish(key, signature, generation, authoritative=True)


def signal_bump(home_dir: Union[str, Path], generation: int) -> None:
    """THE bump's second half, under the home state lock and after phases.yaml was replaced: write the signal file
    atomically and publish the new generation (this process's fences see the Stop at their next compare)."""
    from ..utils.atomic_files import atomic_write_text

    key = home_key(home_dir)
    atomic_write_text(signal_path(home_dir), f"{int(generation)}\n")
    signature = _stat_signature(str(signal_path(home_dir)))
    _publish(key, signature, generation, authoritative=True)
    with _table_lock:
        inflight = list(_INFLIGHT.get(key, ()))
    for event in inflight:
        event.set()  # the operator's own act stops the decode in flight (not an arbitrary interrupt, §2.3)


def _reread_signal(key: str, path: str, signature: _Signature) -> Optional[int]:
    """The signal file changed since it was last read: read its int (the only open on the check's path, once per
    Stop). None when it vanished after being seen or cannot be parsed (the caller reads that as moved)."""
    if signature is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            value = int(fh.read().strip())
    except (OSError, ValueError):
        return None
    _publish(key, signature, value, authoritative=False)
    return value


# ------------------------------------------------------------------------------------------------------ the fence
class LifecycleFence:
    """The generation a piece of work was admitted at, for one home. `moved()` / `check()` are the hot path (an int
    compare, plus one stat of the signal file in a process that did not bump); `commit()` re-reads the truth under
    the home state lock for a host's own check-then-write."""

    __slots__ = ("home_dir", "generation", "_key", "_signal")

    def __init__(self, home_dir: Union[str, Path], generation: int) -> None:
        self.home_dir = Path(home_dir)
        self.generation = int(generation)
        self._key = home_key(home_dir)
        self._signal = str(signal_path(home_dir))
        if self._key not in _SIGNAL:
            try:
                read_truth(self.home_dir)  # first sight of this home in this process
            except LifecycleUnreadable:
                pass  # no table entry: moved() reads True (fail closed)

    @classmethod
    def admit(cls, home_dir: Union[str, Path]) -> "LifecycleFence":
        """Admit work at the home's CURRENT generation. Raises LifecycleUnreadable when it cannot be read (fail closed:
        nothing is admitted on an unknown generation)."""
        return cls(home_dir, read_truth(home_dir))

    def moved(self) -> bool:
        """True once the home's generation is no longer the admitted one - or cannot be known (fail closed)."""
        entry = _SIGNAL.get(self._key)
        if entry is None or entry[1] != self.generation:
            return True
        try:
            signature = _stat_signature(self._signal)
        except OSError:
            return True
        if signature == entry[0]:
            return False
        return _reread_signal(self._key, self._signal, signature) != self.generation

    def check(self, where: str) -> None:
        """Raise StoppedByOperator(where) when the generation moved. Also the memory commit guard's callable."""
        if self.moved():
            entry = _SIGNAL.get(self._key)
            raise StoppedByOperator(where, expected=self.generation, found=None if entry is None else entry[1])

    def truth_moved(self) -> bool:
        """The boundary check (a loop's poll): `moved()`, then phases.yaml itself re-read - the parse the loop has
        always paid at its boundaries, which also sees a bump by a writer that predates the signal file. Raises
        LifecycleUnreadable when phases.yaml cannot be read (the caller ends, fail closed)."""
        if self.moved():
            return True
        return read_truth(self.home_dir) != self.generation

    @contextmanager
    def cancel_event(self) -> Iterator[threading.Event]:
        """A `threading.Event` for ONE model call (`cancel_event=` of a provider that honours it): set by the bump of
        this home in this process, or already set when the generation moved. Registered for the call only."""
        event = threading.Event()
        with _table_lock:
            _INFLIGHT.setdefault(self._key, set()).add(event)
        try:
            if self.moved():
                event.set()
            yield event
        finally:
            with _table_lock:
                held = _INFLIGHT.get(self._key)
                if held is not None:
                    held.discard(event)
                    if not held:
                        _INFLIGHT.pop(self._key, None)

    def commit(self, what: str, fn: Callable[[], T]) -> T:
        """Run `fn` inside the home state lock after re-reading the generation there: refused (StaleLifecycleGeneration,
        nothing run) when it moved. For a host's own check-then-write; never around a model call."""
        from .life import home_state_lock

        with home_state_lock(self.home_dir):
            found = read_truth(self.home_dir)
            if found != self.generation:
                raise StaleLifecycleGeneration(what, self.generation, found)
            return fn()

    def __repr__(self) -> str:
        return f"LifecycleFence({str(self.home_dir)!r}, generation={self.generation})"


Fence = Union[LifecycleFence, _Sentinel]


def verify(fence: Any, home_dir: Union[str, Path], what: str, found: int, *, sentinel_ok: bool) -> None:
    """THE commit points' check, called inside the home state lock with the generation just read there. A sentinel
    passes only where the writer allows it (`sentinel_ok`: what a Stop or a Resume writes)."""
    if fence is KILL_SWITCH or fence is RESUME:
        if not sentinel_ok:
            raise TypeError(f"{what}: {fence!r} writes only what a Stop or a Resume writes - pass a LifecycleFence")
        return
    if not isinstance(fence, LifecycleFence):
        raise TypeError(f"{what}: fence must be a LifecycleFence, KILL_SWITCH or RESUME, not {fence!r}")
    if fence._key != home_key(home_dir):
        raise ValueError(f"{what}: the fence belongs to {fence.home_dir}, not {home_dir}")
    if int(found) != fence.generation:
        raise StaleLifecycleGeneration(what, fence.generation, int(found))


def is_sentinel(fence: Any) -> bool:
    return fence is KILL_SWITCH or fence is RESUME


# ------------------------------------------------------------------------------------------- the effect dispatch
def stopped_outcome(effect: str, where: str) -> Any:
    """What a fenced effect returns instead of running (or instead of its late reply): status `cancelled` - never
    retried, never a failure; the Runtime ends the run CANCELLED with KILL_SWITCH_REASON as its error."""
    from ..core.runtime import EffectOutcome

    return EffectOutcome.cancelled({"reason": KILL_SWITCH_REASON, "cancelled_by": "operator", "where": where,
                                    "effect": effect})


def is_stopped_outcome(outcome: Any) -> bool:
    result = getattr(outcome, "result", None)
    return getattr(outcome, "status", None) == "cancelled" and isinstance(result, dict) \
        and result.get("reason") == KILL_SWITCH_REASON


def fence_handlers(handlers: Dict[Any, Any], fence: LifecycleFence) -> Dict[Any, Any]:
    """THE EFFECT DISPATCH FENCE (STOP-LANE-DESIGN §2.2): every handler of a table, wrapped ONCE (a handler already
    wrapped with this fence is kept). Before each effect: refused once the generation moved. An LLM_CALL whose reply
    returned after the Stop is ABANDONED (§2.3: it lands nowhere). A handler that raised the operator's Stop (memory's
    commit guard refusing inside the write) or failed after it reports the Stop. Every refusal is
    `stopped_outcome` - cancelled, never failed. Cost: one fence compare per effect (two for a model call)."""
    return {etype: _fence_handler(etype, handler, fence) for etype, handler in handlers.items()}


def _fence_handler(etype: Any, inner: Callable[..., Any], fence: LifecycleFence) -> Callable[..., Any]:
    if getattr(inner, "__lifecycle_fence__", None) is fence:
        return inner
    import inspect

    from ..core.models import EffectType

    try:
        params = list(inspect.signature(inner).parameters.values())
        three = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params) or len(params) >= 3
    except (TypeError, ValueError):
        three = True
    name = str(getattr(etype, "value", etype))
    is_llm = etype == EffectType.LLM_CALL
    before = f"before {name}"
    after = f"after {name} returned"

    def fenced(run: Any, effect: Any, default_next_node: Any = None) -> Any:
        if fence.moved():
            return stopped_outcome(name, before)
        try:
            out = inner(run, effect, default_next_node) if three else inner(run, effect)
        except TurnCancelled as e:
            return stopped_outcome(name, str(getattr(e, "where", "") or after))
        if (is_llm or getattr(out, "status", None) != "completed") and fence.moved():
            return stopped_outcome(name, after)
        return out

    fenced.__lifecycle_fence__ = fence  # type: ignore[attr-defined]
    fenced.__lifecycle_inner__ = inner  # type: ignore[attr-defined]
    return fenced


__all__ = [
    "KILL_SWITCH",
    "KILL_SWITCH_REASON",
    "LIFECYCLE_SIGNAL_FILENAME",
    "RESUME",
    "Fence",
    "LifecycleFence",
    "LifecycleUnreadable",
    "StaleLifecycleGeneration",
    "StoppedByOperator",
    "fence_handlers",
    "home_key",
    "is_sentinel",
    "is_stopped_outcome",
    "publish_under_lock",
    "read_truth",
    "signal_bump",
    "stopped_outcome",
    "verify",
]
