"""abstractruntime.core.progress_channel

The OUT-OF-BAND channel that hands one effect's durable progress callback to
the handler that is about to execute it.

WHY IT IS NOT IN THE PAYLOAD
----------------------------
`Effect.payload` is JSON. Every durable consumer treats it that way: the
ledger persists it, `slim_terminal_effect` digests it, the gateway streams it
over SSE, and a long tail of tests and host handlers do a plain
``json.dumps(effect.payload)`` on the way in. When the runtime started
offering a progress callback to EVERY `LLM_CALL` (phase feedback: prefill /
generate / complete, not just generated media), the then-current mechanism —
deep-copy the payload and inject a Python callable at
``payload["params"]["on_progress"]`` — put a function inside that JSON and
broke 77 abstractagent tests with ``Object of type function is not JSON
serializable``. The payload is a message; a live callable is not a message.

THE MECHANISM
-------------
The callback travels beside the effect instead of inside it:

1. `Runtime._execute_effect_with_retry` builds the per-attempt callback
   (it closes over run/node/step/idempotency_key/attempt, which is why it
   cannot be built once and cached) and installs it here for exactly the
   duration of ONE handler invocation, via `effect_progress_callback(...)`.
2. The handler for that effect — `make_llm_call_handler` in the AbstractCore
   integration — reads it with `current_effect_progress_callback()` and puts
   it in ITS OWN local params dict, which becomes provider kwargs
   (`on_progress=`). `effect.payload` is never touched or copied.
3. The token is reset in a `finally`, so a raising handler cannot leak the
   callback into the next effect, and a nested runtime (an effect handler that
   drives a child run) shadows rather than clobbers.

THREADING AND ASYNC
-------------------
Effect handlers run SYNCHRONOUSLY on the thread that called `tick()` —
`Runtime._execute_effect` invokes `handler(run, effect, default_next_node)`
directly and there is no async/await or executor anywhere in the tick path —
so a plain `ContextVar` reaches the handler with no propagation trick. Two
consequences worth stating, because they are what makes this safe:

- A handler that hands work to a bare `threading.Thread` (CPython 3.12 starts
  such a thread with an EMPTY context) must carry the callback OBJECT, not
  re-read this var. Every consumer already does: the handler captures it in
  `params` once, and the object stays valid after the token is reset —
  a provider may legitimately call it from its own thread, and does.
- A pool that wants the var visible must submit through
  `contextvars.copy_context().run`, exactly as `tool_executor` already does
  for its own reasons.

PRECEDENCE
----------
An explicit caller-supplied callable in `params["on_progress"]` always wins;
the runtime's callback is a default, offered only when the caller named none.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Optional


ProgressCallback = Callable[..., Any]

# None = no effect is currently executing, or the running effect has no
# durable progress channel (anything that is not an LLM_CALL today).
_EFFECT_PROGRESS_CALLBACK: "ContextVar[Optional[ProgressCallback]]" = ContextVar(
    "abstractruntime_effect_progress_callback",
    default=None,
)


def current_effect_progress_callback() -> Optional[ProgressCallback]:
    """The progress callback the runtime offers for the effect being executed.

    Returns None when there is none — callers must treat that as "this run
    wants no progress records", never as an error.
    """

    try:
        callback = _EFFECT_PROGRESS_CALLBACK.get()
    except LookupError:  # pragma: no cover - default makes this unreachable
        return None
    return callback if callable(callback) else None


@contextlib.contextmanager
def effect_progress_callback(callback: Optional[ProgressCallback]) -> Iterator[None]:
    """Install `callback` for the body, then restore the previous value.

    Passing None installs None: an effect with no progress channel must not
    inherit the previous effect's callback and write another step's records.
    """

    token = _EFFECT_PROGRESS_CALLBACK.set(callback if callable(callback) else None)
    try:
        yield
    finally:
        _EFFECT_PROGRESS_CALLBACK.reset(token)


# ---------------------------------------------------------------------------
# LIVE TOKEN DELTAS (token streaming, runtime half — S-DESIGN 2026-09-26)
# ---------------------------------------------------------------------------
# The second out-of-band callback, with exactly the same discipline as the
# progress callback above: built per effect attempt by the runtime, installed
# here for the duration of ONE handler invocation, read by the LLM handler
# into its private params (`params["_on_delta"]`), never placed in
# `effect.payload`, reset in a `finally`.
#
# What differs is durability. Progress callbacks write ledger records; delta
# callbacks write NOTHING durable. They hand `(text, channel)` fragments of the
# answer being generated to the host's live sink (`Runtime.set_live_delta_sink`)
# and are offered only when the run asked for streaming (`_runtime.stream is
# True`) AND a host registered a sink. None is the normal case.

DeltaCallback = Callable[..., Any]

_EFFECT_DELTA_CALLBACK: "ContextVar[Optional[DeltaCallback]]" = ContextVar(
    "abstractruntime_effect_delta_callback",
    default=None,
)


def current_effect_delta_callback() -> Optional[DeltaCallback]:
    """The live delta callback offered for the effect being executed, or None.

    The callable takes ``(text: str, channel: str = "content")`` where channel is
    ``"content"`` or ``"reasoning"``. None means "nobody is watching live" —
    callers must treat it as a normal state, never as an error.
    """

    try:
        callback = _EFFECT_DELTA_CALLBACK.get()
    except LookupError:  # pragma: no cover - default makes this unreachable
        return None
    return callback if callable(callback) else None


@contextlib.contextmanager
def effect_delta_callback(callback: Optional[DeltaCallback]) -> Iterator[None]:
    """Install `callback` for the body, then restore the previous value.

    Passing None installs None, so an effect without a live channel never
    inherits the previous effect's callback (which would stream one step's
    text under another step's call id).
    """

    token = _EFFECT_DELTA_CALLBACK.set(callback if callable(callback) else None)
    try:
        yield
    finally:
        _EFFECT_DELTA_CALLBACK.reset(token)


__all__ = [
    "DeltaCallback",
    "ProgressCallback",
    "current_effect_delta_callback",
    "current_effect_progress_callback",
    "effect_delta_callback",
    "effect_progress_callback",
]
