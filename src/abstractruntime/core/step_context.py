"""abstractruntime.core.step_context

The OUT-OF-BAND channel that tells an effect handler WHICH LEDGER ROW its own
step is (ADR 0011).

WHY
---
A memory a handler writes must name the ledger source it came from
(AbstractMemory ADR 0003). The only ledger address is `(run_id, seq)`:
`step_id` is shared by the STARTED row and the terminal row of a step, and the
`seq` of the STARTED row is assigned by the store at append time. A handler
receives `(run, effect, default_next_node)` and could not know either, and a
handler must never guess a seq.

THE MECHANISM
-------------
`Runtime._execute_effect_with_retry` appends the attempt's STARTED record;
`LedgerStore.append` returns the seq it wrote (None when the backend does not
know it). The runtime then installs a `StepContext` for exactly the duration
of ONE handler invocation, the same discipline as the progress callback
(`core.progress_channel`): a `ContextVar`, reset in a `finally`, shadowed (not
clobbered) by a nested runtime that drives a child run inside a handler.

A handler reads it with `current_step_context(run_id)`. Passing the handler's
own `run.run_id` returns the context only when it belongs to that run, so a
handler invoked DIRECTLY (outside any runtime step: the home-direct entity
driver, a test) or inside some other run's step sees None. None means "no
ledger row is known for this write" and the writer stamps no step ref.

The effect payload is never touched: it stays JSON (see progress_channel for
why that matters).
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional


@dataclass(frozen=True)
class StepContext:
    """The step whose handler is executing.

    `seq` is the ledger row address of the step's STARTED record (the record
    that holds the effect payload verbatim); None when the ledger backend did
    not report one. `step_id` is shared by the STARTED and terminal rows.
    """

    run_id: str
    step_id: str
    seq: Optional[int]
    node_id: str
    attempt: int
    effect_type: str
    # sha256 of the STARTED record as written (`storage.serialize.record_digest`), None when unknown: what a
    # memory's step ref vouches the row said, so a later edit of the row is detectable.
    digest: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_id": self.step_id,
            "seq": self.seq,
            "node_id": self.node_id,
            "attempt": self.attempt,
            "effect_type": self.effect_type,
            "digest": self.digest,
        }


_STEP_CONTEXT: "ContextVar[Optional[StepContext]]" = ContextVar(
    "abstractruntime_effect_step_context",
    default=None,
)


def current_step_context(run_id: Optional[str] = None) -> Optional[StepContext]:
    """The step whose handler is executing, or None.

    With `run_id`, the context is returned only when it belongs to that run
    (a handler passes its own `run.run_id`): a context left by an enclosing
    step of ANOTHER run is not this write's source.
    """
    try:
        ctx = _STEP_CONTEXT.get()
    except LookupError:  # pragma: no cover - default makes this unreachable
        return None
    if ctx is None:
        return None
    if run_id is not None and str(run_id) != ctx.run_id:
        return None
    return ctx


@contextlib.contextmanager
def effect_step_context(ctx: Optional[StepContext]) -> Iterator[None]:
    """Install `ctx` for the body, then restore the previous value."""
    token = _STEP_CONTEXT.set(ctx)
    try:
        yield
    finally:
        _STEP_CONTEXT.reset(token)


__all__ = ["StepContext", "current_step_context", "effect_step_context"]
