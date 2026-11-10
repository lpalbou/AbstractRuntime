"""THE Stop fence (STOP-LANE-DESIGN 2026-11-09 §2): one primitive, a closed set of commit points, and the unfenced
write made impossible by signature.

The operator's Stop bumps the home's lifecycle generation (phases.yaml `lifecycle_generation`, under the home state
lock). Everything an entity does is ADMITTED at one generation - a loop at its boot, a turn at its door - and carries
a `LifecycleFence` holding that number. Every commit point (the state writers in `life.py`, memory's commit guard,
the diary and artifact stores, the effect handlers) takes the fence as a REQUIRED argument and refuses once the
generation has moved: a write decided before a Stop never lands after it, whatever process holds it.

The check is an int compare in process plus TWO stats (the signal and phases.yaml, Codex review 4 #9). `_SIGNAL` is a process-wide table, home -> (signature of the
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
# THE HOME'S WRITE BARRIER (Codex review 4 #2): writers of the runtime's own stores (the diary book, the artifact
# store) hold it SHARED around their fence check AND their write; the bump holds it EXCLUSIVE around the generation
# write - so a write whose check passed always lands before a bump, never after it. A dedicated file (stable inode).
WRITE_BARRIER_FILENAME = ".write_barrier.lock"
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


class LifecycleSignalUnwritten(OSError):
    """The bump moved phases.yaml to `generation` but the signal file could not be written (Fable F9): this process's
    table has the Stop, the stale signal was removed (a process that saw it reads "vanished" = moved), and processes
    that never saw a signal see the Stop at their next truth read."""

    def __init__(self, generation: int, error: BaseException) -> None:
        super().__init__(f"the lifecycle generation moved to {generation} but its signal could not be written: {error}")
        self.generation = int(generation)


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
# home -> (signature of the signal file, signature of phases.yaml, the generation) as this process last read them.
_SIGNAL: Dict[str, Tuple[_Signature, _Signature, int]] = {}
_table_lock = threading.Lock()
# home -> the threading.Events of the model calls in flight under a fence of that home (LifecycleFence.cancel_event):
# the bump sets them in the bumping process, so a provider that honours `cancel_event=` stops within a token (§2.3).
_INFLIGHT: Dict[str, set] = {}


def home_key(home_dir: Union[str, Path]) -> str:
    """The table's key for a home: its resolved path (the same key the home state lock uses)."""
    return str(Path(home_dir).resolve())


def signal_path(home_dir: Union[str, Path]) -> Path:
    return Path(home_dir) / LIFECYCLE_SIGNAL_FILENAME


def _phases_path(home_dir: Union[str, Path]) -> Path:
    from .life import PHASES_FILENAME

    return Path(home_dir) / PHASES_FILENAME


def _stat_signature(path: str) -> _Signature:
    """A file's signature; None when it does not exist. Any other OSError propagates (fail closed)."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _publish(key: str, signal_sig: _Signature, phases_sig: _Signature, generation: int, *,
             authoritative: bool) -> None:
    """Record what a read saw. A non-authoritative read never moves the table backwards (a slow reader that parsed
    phases.yaml before a Stop must not undo the Stop another thread already published); an authoritative one (read
    under the home state lock, where no bump can be in flight) is the truth and replaces it."""
    with _table_lock:
        current = _SIGNAL.get(key)
        if authoritative or current is None or generation >= current[2]:
            _SIGNAL[key] = (signal_sig, phases_sig, int(generation))


def _invalidate(key: str) -> None:
    """A truth read failed: no cached generation stands for this home (every fence of it reads moved)."""
    with _table_lock:
        _SIGNAL.pop(key, None)


def _holds_home_lock(key: str) -> bool:
    from .life import _home_state_held

    return bool((getattr(_home_state_held, "depth", None) or {}).get(key))


def read_truth(home_dir: Union[str, Path]) -> int:
    """phases.yaml's lifecycle generation, published into the table with both signatures. The two files are stat-ed
    BEFORE phases.yaml is parsed (the bump writes them in the other order), so a published entry is never newer on
    its signature side than on its generation side. Raises LifecycleUnreadable (a ValueError) when either cannot be
    read - and drops this home's cached truth (Codex review 4 #9: a failed read must not leave the old one standing)."""
    from .life import _generation_of, _load_phases_for_write

    key = home_key(home_dir)
    try:
        signal_sig = _stat_signature(str(signal_path(home_dir)))
        phases_sig = _stat_signature(str(_phases_path(home_dir)))
    except OSError as e:
        _invalidate(key)
        raise LifecycleUnreadable(f"the lifecycle files of {home_dir} cannot be read ({e})") from e
    try:
        generation = _generation_of(_load_phases_for_write(home_dir, "reading the lifecycle generation"))
    except ValueError as e:
        _invalidate(key)
        raise LifecycleUnreadable(str(e)) from e
    _publish(key, signal_sig, phases_sig, generation, authoritative=_holds_home_lock(key))
    return generation


def publish_under_lock(home_dir: Union[str, Path], generation: int) -> None:
    """A writer holding the home state lock parsed (or just wrote) phases.yaml at `generation`: that is the truth."""
    key = home_key(home_dir)
    try:
        signal_sig = _stat_signature(str(signal_path(home_dir)))
        phases_sig = _stat_signature(str(_phases_path(home_dir)))
    except OSError:
        _invalidate(key)  # the files cannot be seen: every fence of this home reads moved (fail closed)
        return
    _publish(key, signal_sig, phases_sig, generation, authoritative=True)


def _establish_signal(home_dir: Union[str, Path], generation: int) -> None:
    """ADMISSION REQUIRES A VALID SIGNAL (Codex review 4 #9): under the home state lock, write the signal when it is
    missing, unreadable or behind the truth (never lower one that is ahead: a fence admitted below it reads moved).
    Raises LifecycleUnreadable when it cannot be established - nothing is admitted then."""
    from ..utils.atomic_files import atomic_write_text

    path = signal_path(home_dir)
    try:
        current: Optional[int] = int(path.read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        current = None
    except ValueError:
        current = None
    except OSError as e:
        raise LifecycleUnreadable(f"the lifecycle signal of {home_dir} cannot be read ({e})") from e
    if current is not None and current > int(generation):
        # The signal is AHEAD of phases.yaml (a restored or hand-edited file): no admission below it (fail closed).
        raise LifecycleUnreadable(f"the lifecycle signal of {home_dir} says generation {current} but phases.yaml says "
                                  f"{generation} - repair phases.yaml before admitting anything")
    if current is None or current < int(generation):
        try:
            atomic_write_text(path, f"{int(generation)}\n")
        except OSError as e:
            raise LifecycleUnreadable(f"the lifecycle signal of {home_dir} cannot be written ({e})") from e
    publish_under_lock(home_dir, generation)


def signal_bump(home_dir: Union[str, Path], generation: int) -> None:
    """THE bump's second half, under the home state lock and after phases.yaml was replaced: publish the new
    generation FIRST (Fable F9: this process's fences see the Stop at their next compare whatever the disk does), then
    write the signal file atomically. A failed write removes the stale signal (a process that saw it reads "vanished"
    = moved) and raises LifecycleSignalUnwritten carrying the generation the Stop moved to; if even that removal fails,
    every other process still sees the Stop through phases.yaml's own signature (Codex review 4 #9)."""
    from ..utils.atomic_files import atomic_write_text

    key = home_key(home_dir)
    path = signal_path(home_dir)
    with _table_lock:
        previous = _SIGNAL.get(key)
    _publish(key, previous[0] if previous else None, previous[1] if previous else None, generation,
             authoritative=True)
    try:
        try:
            atomic_write_text(path, f"{int(generation)}\n")
        except OSError as e:
            try:
                path.unlink()
            except OSError:
                pass
            raise LifecycleSignalUnwritten(generation, e) from e
        publish_under_lock(home_dir, generation)
    finally:
        with _table_lock:
            inflight = list(_INFLIGHT.get(key, ()))
        for event in inflight:
            event.set()  # the operator's own act stops the decode in flight (not an arbitrary interrupt, §2.3)


def _reread_signal(key: str, path: str, signal_sig: _Signature, phases_sig: _Signature) -> Optional[int]:
    """The signal file changed since it was last read: read its int (once per Stop). None when it cannot be parsed
    (the caller reads that as moved)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            value = int(fh.read().strip())
    except (OSError, ValueError):
        return None
    _publish(key, signal_sig, phases_sig, value, authoritative=False)
    return value


# ------------------------------------------------------------------------------------------------------ the fence
class LifecycleFence:
    """The generation a piece of work was admitted at, for one home. `moved()` / `check()` are the hot path (an int
    compare plus two stats: the signal and phases.yaml); `commit()` re-reads the truth under the home state lock for
    a host's own check-then-write."""

    __slots__ = ("home_dir", "generation", "_key", "_signal", "_phases")

    def __init__(self, home_dir: Union[str, Path], generation: int) -> None:
        self.home_dir = Path(home_dir)
        self.generation = int(generation)
        self._key = home_key(home_dir)
        self._signal = str(signal_path(home_dir))
        self._phases = str(_phases_path(home_dir))
        if self._key not in _SIGNAL:
            try:
                read_truth(self.home_dir)  # first sight of this home in this process
            except LifecycleUnreadable:
                pass  # no table entry: moved() reads True (fail closed)

    @classmethod
    def admit(cls, home_dir: Union[str, Path]) -> "LifecycleFence":
        """Admit work at the home's CURRENT generation, inside the home state lock, with its signal established
        (Codex review 4 #9). Raises LifecycleUnreadable when either cannot be read or written (fail closed: nothing
        is admitted on an unknown generation)."""
        from .life import home_state_lock

        with home_state_lock(home_dir):
            generation = read_truth(home_dir)
            _establish_signal(home_dir, generation)
            return cls(home_dir, generation)

    def moved(self) -> bool:
        """True once the home's generation is no longer the admitted one - or cannot be known (fail closed): a missing
        signal, an unreadable or changed-and-unreadable phases.yaml."""
        key = self._key
        entry = _SIGNAL.get(key)
        if entry is None or entry[2] != self.generation:
            return True
        try:
            signal_sig = _stat_signature(self._signal)
            phases_sig = _stat_signature(self._phases)
        except OSError:
            return True
        if signal_sig is None:
            return True  # admission established a signal: missing means removed or never published - refuse
        if signal_sig != entry[0]:
            if _reread_signal(key, self._signal, signal_sig, entry[1]) != self.generation:
                return True
        if phases_sig != entry[1]:
            # phases.yaml changed without the signal saying so (a grant, a resume_to - or a bump whose signal failed,
            # or a corruption): its truth is read once, here.
            try:
                return read_truth(self.home_dir) != self.generation
            except LifecycleUnreadable:
                return True
        return False

    def check(self, where: str) -> None:
        """Raise StoppedByOperator(where) when the generation moved. Also the memory commit guard's callable."""
        if self.moved():
            entry = _SIGNAL.get(self._key)
            raise StoppedByOperator(where, expected=self.generation, found=None if entry is None else entry[2])

    def truth_moved(self) -> bool:
        """The boundary check (a loop's poll): `moved()`, then phases.yaml itself re-read - the parse the loop has
        always paid at its boundaries. Raises LifecycleUnreadable when phases.yaml cannot be read (the caller ends,
        fail closed)."""
        truth = read_truth(self.home_dir)  # first: an unreadable file raises, never reads as a Stop
        return truth != self.generation or self.moved()

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


# ------------------------------------------------------------------------- the operation's own admission (Codex 4 #5)
# An operation keeps the admission it STARTED under, whatever the home's holder says later: the fenced handler table
# pushes the effect's admission here for the effect's duration, a ChatSession pushes its own fence for each of its
# entry points, and the holder's checks (the memory commit guard, the diary and artifact guards, the authoring
# callbacks) resolve to it. Per thread, per home; a nested effect (a child run inside START_SUBWORKFLOW) inherits it.
_ADMITTED = threading.local()

# A host may stamp the generation a RUN was admitted at into its vars (an int): every effect of that run is then
# checked against it, so a run admitted before a Stop never acts after a Resume readmitted the runtime.
RUN_GENERATION_VAR = "_lifecycle_generation"


_barrier_held = threading.local()


@contextmanager
def write_barrier(home_dir: Union[str, Path], *, exclusive: bool) -> Iterator[None]:
    """Hold the home's write barrier (see WRITE_BARRIER_FILENAME): `exclusive` for the bump, shared for a write
    section. Cross-process (fcntl.flock on a dedicated file), waiting without a limit - a Stop never fails, and a
    write section is one fence check plus one store write. Re-entrant per thread; asking for it exclusive while this
    thread holds it shared raises (it would wait on itself). Lock order: the home state lock OUTER, this barrier
    inner, the stores' own locks innermost; a write section never takes the home state lock."""
    key = home_key(home_dir)
    held = getattr(_barrier_held, "by_key", None)
    if held is None:
        held = _barrier_held.by_key = {}
    mode = held.get(key)
    if mode is not None:
        if exclusive and mode == "shared":
            raise RuntimeError("the lifecycle generation cannot move from inside a fenced write section")
        yield  # already held by this thread (shared in shared, anything in exclusive)
        return
    try:
        import fcntl
    except ImportError:  # non-POSIX: no cross-process barrier (the in-process fence still holds)
        yield
        return
    fh = open(Path(home_dir) / WRITE_BARRIER_FILENAME, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        held[key] = "exclusive" if exclusive else "shared"
        try:
            yield
        finally:
            held.pop(key, None)
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def write_section(guard: Any, barrier_home: Any, store: str) -> Callable[[str], Any]:
    """A store's `writing(where)` from its guard (Codex review 4 #2): the home's fence holder (its own `.writing`), or
    a host's callable guard together with `barrier_home` - the home whose write barrier the write holds while the
    guard is checked. A callable without `barrier_home` is refused (it would check outside the barrier)."""
    if callable(getattr(guard, "writing", None)):
        return guard.writing
    if not callable(guard):
        raise TypeError(f"{store}(guard=) takes the home's fence holder or a callable guard, not {guard!r}")
    if barrier_home is None:
        raise TypeError(f"{store}(guard=<callable>) needs barrier_home= (the home whose write barrier its writes hold)")

    @contextmanager
    def writing(where: str) -> Iterator[None]:
        with write_barrier(barrier_home, exclusive=False):
            guard(where)
            yield

    return writing


@contextmanager
def admitted(fence: LifecycleFence) -> Iterator[LifecycleFence]:
    """Act under `fence` (an operation's immutable admission) on this thread until the block ends."""
    by_key = getattr(_ADMITTED, "by_key", None)
    if by_key is None:
        by_key = _ADMITTED.by_key = {}
    stack = by_key.setdefault(fence._key, [])
    stack.append(fence)
    try:
        yield fence
    finally:
        stack.pop()


def _operation_admission(key: str) -> Optional[LifecycleFence]:
    by_key = getattr(_ADMITTED, "by_key", None)
    stack = by_key.get(key) if by_key else None
    return stack[-1] if stack else None


class FenceHolder:
    """THE FENCE A LONG-LIVED OPENED HOME ACTS UNDER (Fable F3, 2026-11-09): one object its handler table, memory
    commit guard, diary and artifact guards read at every check, so a host that keeps the home open (the gateway's
    cached EntityRuntime per slug) re-admits it after Resume with `readmit(fence)` instead of reopening it. The swap
    happens inside the home state lock and refuses a fence that is not the home's current admission (a fence decided
    before a Stop is never readmitted).

    The swap serves NEW admissions only (Codex review 4 #5): every check resolves first to the admission of the
    operation running on this thread (`admitted`: an effect keeps the fence it started under, a session its own), so
    a write still in flight from before the Stop stays refused after a readmission."""

    __slots__ = ("fence",)

    def __init__(self, fence: LifecycleFence) -> None:
        if not isinstance(fence, LifecycleFence):
            raise TypeError(f"a FenceHolder holds a LifecycleFence, not {fence!r}")
        self.fence = fence

    def current(self) -> LifecycleFence:
        """The admission a check made now is judged against: the running operation's, else the home's newest."""
        return _operation_admission(self.fence._key) or self.fence

    def moved(self) -> bool:
        return self.current().moved()

    def check(self, where: str) -> None:
        self.current().check(where)

    @contextmanager
    def writing(self, where: str) -> Iterator[None]:
        """A store write as ONE section with its check (Codex review 4 #2): the write barrier held shared, the fence
        checked inside it, the write made inside it - a racing bump waits for the section to end."""
        with write_barrier(self.fence.home_dir, exclusive=False):
            self.check(where)
            yield

    def readmit(self, fence: LifecycleFence) -> None:
        from .life import home_state_lock

        if not isinstance(fence, LifecycleFence):
            raise TypeError(f"readmit takes a LifecycleFence, not {fence!r}")
        if fence._key != self.fence._key:
            raise ValueError(f"readmit: the fence belongs to {fence.home_dir}, not {self.fence.home_dir}")
        with home_state_lock(fence.home_dir):
            found = read_truth(fence.home_dir)
            if found != fence.generation:
                raise StaleLifecycleGeneration("readmitting the home", fence.generation, found)
            self.fence = fence


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
def stopped_outcome(effect: str, where: str, *, ran: bool = True) -> Any:
    """What a fenced effect returns instead of running (or instead of its late reply): status `cancelled` - never
    retried, never a failure; the Runtime ends the run CANCELLED with KILL_SWITCH_REASON as its error. `ran=False`
    (refused before it started): the Runtime's node trace keeps no payload for it (live E2E 2026-11-10: the prompt
    of a call that never ran was saved as artifacts after the Stop)."""
    from ..core.runtime import EffectOutcome

    return EffectOutcome.cancelled({"reason": KILL_SWITCH_REASON, "cancelled_by": "operator", "where": where,
                                    "effect": effect, "ran": bool(ran)})


def is_stopped_outcome(outcome: Any) -> bool:
    result = getattr(outcome, "result", None)
    return getattr(outcome, "status", None) == "cancelled" and isinstance(result, dict) \
        and result.get("reason") == KILL_SWITCH_REASON


def fence_handlers(handlers: Dict[Any, Any], fence: Union[LifecycleFence, "FenceHolder"]) -> Dict[Any, Any]:
    """THE EFFECT DISPATCH FENCE (STOP-LANE-DESIGN §2.2): every handler of a table, wrapped ONCE (a handler already
    wrapped with this fence is kept). Before each effect: refused once the generation moved. After it, EVERY outcome
    is checked again (Fable F2: a subworkflow whose child the Stop cancelled answers `completed` with `cancelled: True`
    and the parent went on): a model reply that returned after the Stop is ABANDONED (§2.3: it lands nowhere), and any
    effect that returns after it reports the Stop - its run ends there. A handler that raised the operator's Stop
    (memory's commit guard refusing inside the write) reports it too. Every refusal is `stopped_outcome` - cancelled,
    never failed. Cost: two fence checks per effect. `fence` may be a FenceHolder (an opened home's): the wrappers then
    read whatever fence it holds at each check (readmit)."""
    holder = fence if isinstance(fence, FenceHolder) else FenceHolder(fence)
    return {etype: _fence_handler(etype, handler, holder) for etype, handler in handlers.items()}


def _fence_handler(etype: Any, inner: Callable[..., Any], fence: "FenceHolder") -> Callable[..., Any]:
    if getattr(inner, "__lifecycle_fence__", None) is fence:
        return inner
    import inspect

    try:
        params = list(inspect.signature(inner).parameters.values())
        three = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params) or len(params) >= 3
    except (TypeError, ValueError):
        three = True
    name = str(getattr(etype, "value", etype))
    before = f"before {name}"
    after = f"after {name} returned"

    def fenced(run: Any, effect: Any, default_next_node: Any = None) -> Any:
        # THE EFFECT'S ADMISSION, fixed for its whole duration (Codex review 4 #5): the enclosing operation's, else
        # the generation its run was admitted at (RUN_GENERATION_VAR, when the host stamped it), else the home's
        # newest. A readmission during the effect never re-authorizes it.
        admission = _operation_admission(fence.fence._key)
        if admission is None:
            stamped = (getattr(run, "vars", None) or {}).get(RUN_GENERATION_VAR) if run is not None else None
            admission = LifecycleFence(fence.fence.home_dir, int(stamped)) if isinstance(stamped, int) \
                else fence.fence
        if admission.moved():
            return stopped_outcome(name, before, ran=False)
        with admitted(admission):
            try:
                out = inner(run, effect, default_next_node) if three else inner(run, effect)
            except TurnCancelled as e:
                return stopped_outcome(name, str(getattr(e, "where", "") or after))
        if admission.moved():
            return stopped_outcome(name, after)
        return out

    fenced.__lifecycle_fence__ = fence  # type: ignore[attr-defined]
    fenced.__lifecycle_inner__ = inner  # type: ignore[attr-defined]
    return fenced


__all__ = [
    "KILL_SWITCH",
    "KILL_SWITCH_REASON",
    "LIFECYCLE_SIGNAL_FILENAME",
    "RUN_GENERATION_VAR",
    "WRITE_BARRIER_FILENAME",
    "admitted",
    "write_barrier",
    "RESUME",
    "Fence",
    "FenceHolder",
    "LifecycleFence",
    "LifecycleSignalUnwritten",
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
