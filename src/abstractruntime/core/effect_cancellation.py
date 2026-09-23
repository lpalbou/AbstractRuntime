"""abstractruntime.core.effect_cancellation

How a cancel reaches the effect that is ALREADY RUNNING.

THE INCIDENT (2026-09-22)
-------------------------
A basic-agent run fell into a runaway repetitive generation (33k-token prompt,
no output limit). The operator pressed Stop; the gateway applied the cancel
command within seconds (`cancel_run` → status CANCELLED), and the model kept
decoding at full CPU for an hour. `cancel_run` only wrote run state: the tick
thread blocked inside `_execute_effect` → LLM_CALL handler → `llm.generate(...)`
was never told, and nothing in the process knew which effect was in flight on
which thread.

THE MECHANISM (same out-of-band discipline as `core.progress_channel`)
----------------------------------------------------------------------
1. `Runtime._execute_effect_with_retry` creates ONE `threading.Event` per
   attempt and REGISTERS an `InflightEffect` (run, parent, node, step, effect
   type, start time, thread) in a PROCESS-WIDE registry for exactly the
   duration of the handler call, and installs it in a ContextVar.
   The registry is process-wide on purpose: hosts (the gateway) build a fresh
   `Runtime` object per command, so an instance-level table would never see
   the tick thread's effect.
2. The handler reads the event with `current_effect_cancel_event()` and puts
   it in ITS OWN params (the LLM_CALL handler → provider kwarg
   `cancel_event=`); `effect.payload` stays JSON and is never touched.
3. `Runtime.cancel_run(run_id, cancelled_by=...)` — after persisting
   CANCELLED — calls `request_effect_cancel(run_id, ...)`, which sets the
   event of every registered effect of that run and stamps who/when/why.
   The provider stops decoding within one token (MLX) and raises; the runtime
   turns the attempt into a `cancelled` outcome (never `failed`, never
   retried) and appends a ledger record with status `cancelled` and
   `cancelled_by`.
4. A cancel that lands between the tick loop's control probe and the
   registration is closed IN MEMORY: `request_effect_cancel` remembers the
   run under the registry lock, and an effect registering afterwards is born
   cancelled (its handler is never invoked).

`inflight_effects(...)` is the read side the gateway's kill switch uses: an
effect still registered N seconds after its cancel was requested did not
stop, and the host escalates (see AbstractGateway `stop_kill_switch_s`).

THREADING
---------
Effects run synchronously on the tick thread, so a ContextVar reaches the
handler; a handler that hands work to another thread must carry the Event
OBJECT, exactly like the progress callback. The registry is guarded by one
lock; reads return plain dict snapshots.
"""

from __future__ import annotations

import contextlib
import datetime
import itertools
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@dataclass(eq=False)
class InflightEffect:
    """One effect attempt currently executing in this process."""

    run_id: str
    node_id: str
    step_id: str
    effect_type: str
    attempt: int = 1
    parent_run_id: Optional[str] = None
    started_at: str = field(default_factory=_utc_now_iso)
    started_monotonic: float = field(default_factory=time.monotonic)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    provider: Optional[str] = None
    model: Optional[str] = None
    thread_name: str = field(default_factory=lambda: threading.current_thread().name)
    thread_ident: int = field(default_factory=threading.get_ident)
    # Stamped by the FIRST request_cancel (later requests never overwrite the
    # attribution: the first cause is the cause).
    cancelled_by: Optional[str] = None
    cancel_reason: Optional[str] = None
    cancel_requested_at: Optional[str] = None
    cancel_requested_monotonic: Optional[float] = None
    # Stamped by `kill_inflight_effect` (the host's hard stop): the attempt is
    # being unwound by an injected `EffectKilled`.
    killed_by: Optional[str] = None
    kill_reason: Optional[str] = None
    killed_at: Optional[str] = None
    kill_requested_monotonic: Optional[float] = None

    def request_cancel(self, *, cancelled_by: str, reason: Optional[str]) -> bool:
        """Set the event; True when THIS call was the first request."""

        first = self.cancelled_by is None
        if first:
            self.cancelled_by = str(cancelled_by or "unknown")
            self.cancel_reason = reason
            self.cancel_requested_at = _utc_now_iso()
            self.cancel_requested_monotonic = time.monotonic()
        self.cancel_event.set()
        return first

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def snapshot(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        t = time.monotonic() if now is None else now
        out: Dict[str, Any] = {
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "node_id": self.node_id,
            "step_id": self.step_id,
            "effect_type": self.effect_type,
            "attempt": self.attempt,
            "started_at": self.started_at,
            "elapsed_s": round(max(0.0, t - self.started_monotonic), 3),
            "provider": self.provider,
            "model": self.model,
            "thread": self.thread_name,
            "cancel_requested": self.cancel_event.is_set(),
        }
        if self.killed_by is not None:
            out["killed_by"] = self.killed_by
            out["killed_at"] = self.killed_at
        if self.cancelled_by is not None:
            out["cancelled_by"] = self.cancelled_by
            out["cancel_reason"] = self.cancel_reason
            out["cancel_requested_at"] = self.cancel_requested_at
            if self.cancel_requested_monotonic is not None:
                out["since_cancel_s"] = round(max(0.0, t - self.cancel_requested_monotonic), 3)
        return out


_lock = threading.Lock()
_ids = itertools.count(1)
_inflight: Dict[int, InflightEffect] = {}
# run_id -> (cancelled_by, reason) for every run cancelled in this process.
# Closes the registration race WITHOUT a store read per effect: an effect that
# registers after its run's cancel was requested is born cancelled. A cancelled
# run is terminal and never runs again, so an entry is never wrong; it is a few
# dozen bytes per cancelled run for the life of the process and is deliberately
# NOT capped (a capped set could forget a cancel and let an effect start).
_cancelled_runs: Dict[str, tuple] = {}

_CURRENT: "ContextVar[Optional[InflightEffect]]" = ContextVar(
    "abstractruntime_inflight_effect", default=None
)


@contextlib.contextmanager
def effect_inflight_scope(entry: InflightEffect) -> Iterator[InflightEffect]:
    """Register `entry` (process-wide + ContextVar) for the body; always unregister."""

    with _lock:
        token_id = next(_ids)
        _inflight[token_id] = entry
        prior = _cancelled_runs.get(entry.run_id)
    if prior is not None:
        entry.request_cancel(cancelled_by=prior[0], reason=prior[1])
    ctx_token = _CURRENT.set(entry)
    try:
        yield entry
    finally:
        _CURRENT.reset(ctx_token)
        with _lock:
            _inflight.pop(token_id, None)


def current_inflight_effect() -> Optional[InflightEffect]:
    try:
        return _CURRENT.get()
    except LookupError:  # pragma: no cover - default makes this unreachable
        return None


def current_effect_cancel_event() -> Optional[threading.Event]:
    """The cancel event of the effect being executed on this context, if any."""

    entry = current_inflight_effect()
    return entry.cancel_event if entry is not None else None


def annotate_current_effect(*, provider: Any = None, model: Any = None) -> None:
    """Let a handler name what it is running (kill-switch log/ledger attribution)."""

    entry = current_inflight_effect()
    if entry is None:
        return
    if isinstance(provider, str) and provider.strip():
        entry.provider = provider.strip()
    if isinstance(model, str) and model.strip():
        entry.model = model.strip()


def request_effect_cancel(
    run_id: str, *, cancelled_by: str, reason: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Signal every in-flight effect of `run_id` AND of its in-flight descendants.

    Descendants are found through the registry itself (`parent_run_id` of the
    in-flight entries, closed transitively): a synchronous subworkflow's child
    effect runs INSIDE the parent's effect on the same thread, so cancelling the
    root must reach the child's model call even when no host walks the run tree.
    Descendant runs are remembered as cancelled too, so an effect they register
    later is born cancelled. Returns snapshots of the effects signalled.
    """

    rid = str(run_id or "")
    by = str(cancelled_by or "unknown")
    with _lock:
        _cancelled_runs.setdefault(rid, (by, reason))
        tree = {rid}
        grew = True
        while grew:
            grew = False
            for e in _inflight.values():
                if e.run_id not in tree and e.parent_run_id in tree:
                    tree.add(e.run_id)
                    _cancelled_runs.setdefault(e.run_id, (by, f"parent run {e.parent_run_id} cancelled"))
                    grew = True
        targets = [e for e in _inflight.values() if e.run_id in tree]
    now = time.monotonic()
    out: List[Dict[str, Any]] = []
    for entry in targets:
        entry.request_cancel(cancelled_by=cancelled_by, reason=reason)
        out.append(entry.snapshot(now=now))
    return out


def request_model_effects_cancel(
    provider: Any, model: Any, *, cancelled_by: str = "model_eject", reason: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Signal every in-flight effect currently running `provider`/`model`.

    The EJECT path (model residency unload): the model is about to be freed,
    so the calls using it are stopped FIRST, with an attribution the ledger
    and the UI can show (`cancelled_by="model_eject"`), instead of the
    provider cancelling them anonymously. Matching uses the names handlers
    record through `annotate_current_effect`: the model must match exactly;
    the provider must match when the effect recorded one. Returns snapshots
    of the effects signalled."""

    model_s = str(model or "").strip()
    provider_s = str(provider or "").strip().lower()
    if not model_s:
        return []
    with _lock:
        targets = [
            e for e in _inflight.values()
            if (e.model or "").strip() == model_s
            and (not e.provider or not provider_s or e.provider.strip().lower() == provider_s)
        ]
    now = time.monotonic()
    out: List[Dict[str, Any]] = []
    for entry in targets:
        entry.request_cancel(cancelled_by=cancelled_by, reason=reason)
        out.append(entry.snapshot(now=now))
    return out


def run_cancel_requested(run_id: str) -> Optional[Dict[str, Any]]:
    """{cancelled_by, reason} when a cancel was requested for `run_id` in this process."""

    with _lock:
        prior = _cancelled_runs.get(str(run_id or ""))
    if prior is None:
        return None
    return {"cancelled_by": prior[0], "reason": prior[1]}


class EffectKilled(BaseException):
    """Injected into the thread of an effect that did not stop on its cancel event.

    A BaseException ON PURPOSE (like KeyboardInterrupt): it must unwind through
    every `except Exception` between the decode loop and the runtime — provider
    error swallowers, retry layers, handler catch-alls — releasing locks and
    closing generators through their `finally`/`with` blocks, and is caught
    ONLY by `Runtime._execute_effect_with_retry` (and, if it ever lands between
    effects, by `Runtime.tick`). CPython delivers an asynchronous exception at
    the target thread's next bytecode boundary: a Python-level decode loop
    (mlx-lm, mlx-vlm, the native scheduler's consumer) stops within one token;
    a thread blocked inside ONE native call receives it only when that call
    returns — `kill_inflight_effect` cannot shorten a single Metal op.
    """


def kill_inflight_effect(step_id: str, *, killed_by: str = "kill_switch", reason: Optional[str] = None) -> Dict[str, Any]:
    """HARD STOP of one in-flight effect, in process: set its cancel event and
    inject `EffectKilled` into the thread executing it. Returns what was done:
    {"injected": bool, "reason": ...} plus the effect snapshot.

    The injection happens under the registry lock and only while the entry is
    still registered on the SAME thread, so it cannot be aimed at an effect
    that already finished; the residual window (the target leaving its scope
    between this call and the next bytecode) is handled by the catch sites,
    which recognise a kill not meant for them (the effect's `killed_by` is
    unset) and recover instead of failing that work.
    """

    import ctypes

    sid = str(step_id or "")
    with _lock:
        entry = next((e for e in _inflight.values() if e.step_id == sid), None)
        if entry is None:
            return {"injected": False, "reason": "not in flight (already finished)"}
        if entry.killed_by is None:
            entry.killed_by = str(killed_by or "unknown")
            entry.kill_reason = reason
            entry.killed_at = _utc_now_iso()
            entry.kill_requested_monotonic = time.monotonic()
        entry.request_cancel(cancelled_by=killed_by, reason=reason)
        if entry.thread_ident == threading.get_ident():
            return {"injected": False, "reason": "refusing to kill the calling thread", **entry.snapshot()}
        affected = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(entry.thread_ident), ctypes.py_object(EffectKilled)
        )
        if affected != 1:
            if affected > 1:  # pragma: no cover - CPython contract: undo on >1
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(entry.thread_ident), None)
            return {"injected": False, "reason": f"thread {entry.thread_ident} not found ({affected})", **entry.snapshot()}
        return {"injected": True, **entry.snapshot()}


def inflight_effects(run_ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Snapshots of effects executing now (optionally only for `run_ids`)."""

    wanted = None if run_ids is None else {str(r) for r in run_ids}
    with _lock:
        entries = list(_inflight.values())
    now = time.monotonic()
    return [e.snapshot(now=now) for e in entries if wanted is None or e.run_id in wanted]


__all__ = [
    "EffectKilled",
    "InflightEffect",
    "kill_inflight_effect",
    "annotate_current_effect",
    "current_effect_cancel_event",
    "current_inflight_effect",
    "effect_inflight_scope",
    "inflight_effects",
    "request_effect_cancel",
    "request_model_effects_cancel",
    "run_cancel_requested",
]
