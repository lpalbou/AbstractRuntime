"""abstractruntime.core.live_deltas

Live token deltas for ONE LLM call: the runtime half of token streaming.

WHAT THIS IS
------------
When a run asks for streaming (`run.vars["_runtime"]["stream"] is True`) and the
host registered a sink (`Runtime.set_live_delta_sink`), the runtime builds one
`LiveDeltaEmitter` per LLM_CALL attempt and offers it to the handler through
`core.progress_channel` (never through `effect.payload`). The provider layer
calls it with each generated fragment; the emitter coalesces fragments and
hands the host plain dicts:

    {"kind": "llm.delta", "run_id", "parent_run_id", "node_id", "call_id", "seq", "text", "channel"}
    {"kind": "llm.delta_end", "run_id", "parent_run_id", "node_id", "call_id", "seq", "reason"
     [, "detail"]}

- `parent_run_id` is the emitting run's parent (None for a root run), so a
  host can route a child run's deltas to the conversation of its root run.

- `call_id` is the attempt's `StepRecord.step_id` (the durable LLM_CALL record
  that later carries the final answer has the same id).
- `seq` starts at 0 and increases by one per event of the call, delta_end
  included, so a consumer can detect a gap.
- `channel` is `"content"` (answer text) or `"reasoning"` (thinking text, which
  clients usually hide or fold).
- `reason` is `"completed"`, `"failed"`, `"cancelled"` or `"unavailable"`.
  `"unavailable"` means the call did not stream (or stopped streaming) and says
  why in `detail` (see `UNAVAILABLE_DETAILS`); the durable answer is complete
  either way. Exactly one delta_end is emitted per call, after the call's
  durable record is written.

NOTHING HERE IS DURABLE. The ledger never sees a delta: the durable LLM_CALL
record is identical whether or not anyone streamed. Losing deltas (a slow or
failing sink) loses only the live preview, never the answer.

BATCHING
--------
The first fragment is emitted at once (time to first visible token matters);
later fragments arriving within `flush_interval_s` (40 ms) of the previous
emission are coalesced into one event, flushed by a short timer so a pause in
generation never strands text. A channel switch flushes first, so the order of
content and reasoning text is preserved. `end()` flushes whatever is pending.

THE SINK CONTRACT
-----------------
`sink(event: dict) -> None` is called from the provider's thread (and from the
flush timer's thread), under this emitter's lock, so events of one call arrive
in `seq` order. It must return quickly (queue, do not do I/O). A sink that
raises is disabled for the rest of this call with one warning and the call is
marked unavailable (`sink_error`); the sink is tried once more for the
`llm.delta_end`, and the call itself is never affected.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

LiveDeltaSink = Callable[[Dict[str, Any]], None]

DELTA_CHANNELS = ("content", "reasoning")
END_REASONS = ("completed", "failed", "cancelled", "unavailable")
# Why a call did not stream. Every "no stream" outcome is named, never silent.
UNAVAILABLE_DETAILS = (
    "usage_unavailable",  # the provider cannot report token usage when streaming
    "prompt_cache_unavailable",  # the provider's streamed lane drops prompt-cache telemetry
    "structured_output",  # structured / artifact output calls are never streamed
    "provider_cannot_stream",  # the provider answered in one piece
    "remote_core",  # remote mode: the AbstractCore server call is not streamed
    "node_stream_off",  # the LLM_CALL payload says `stream: False`
    "sink_error",  # the host sink raised; the rest of the call was not delivered
    "tool_envelope_holdback",  # the whole answer was a tool call / hidden channel, held back
)
DEFAULT_FLUSH_INTERVAL_S = 0.040


class LiveDeltaEmitter:
    """Coalescing per-call delta emitter. Callable as ``emitter(text, channel)``."""

    def __init__(
        self,
        sink: LiveDeltaSink,
        *,
        run_id: str,
        node_id: str,
        call_id: str,
        parent_run_id: Optional[str] = None,
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(sink):
            raise TypeError("live delta sink must be callable")
        self._sink: Optional[LiveDeltaSink] = sink
        self.run_id = str(run_id)
        self.parent_run_id = str(parent_run_id) if parent_run_id else None
        self.node_id = str(node_id)
        self.call_id = str(call_id)
        self._interval = max(0.0, float(flush_interval_s))
        self._clock = clock
        self._lock = threading.RLock()
        self._seq = 0
        self._pending_text: list[str] = []
        self._pending_channel: Optional[str] = None
        self._last_emit: Optional[float] = None
        self._timer: Optional[threading.Timer] = None
        self._ended = False
        self.emitted_deltas = 0
        self.unavailable_detail: Optional[str] = None
        # A gap recorded on the durable record only (the live view is not told
        # "not streamed" under text that DID stream).
        self.record_only_detail: Optional[str] = None
        self._failed_sink: Optional[LiveDeltaSink] = None

    # -- producer side -----------------------------------------------------
    def __call__(self, text: Any, channel: str = "content") -> None:
        if not isinstance(text, str) or not text:
            return
        if channel not in DELTA_CHANNELS:
            raise ValueError(f"unknown live delta channel {channel!r}; expected one of {DELTA_CHANNELS}")
        with self._lock:
            if self._ended:
                return
            if self._pending_channel is not None and self._pending_channel != channel:
                self._flush_locked()
            self._pending_channel = channel
            self._pending_text.append(text)
            now = self._clock()
            if self._last_emit is None or (now - self._last_emit) >= self._interval:
                self._flush_locked()
            else:
                self._arm_timer_locked(self._interval - (now - self._last_emit))

    def mark_unavailable(self, detail: str) -> None:
        """Record why this call does not (or no longer) stream. First reason wins.

        The call's `llm.delta_end` then carries `reason: "unavailable"` with this
        `detail` when the call otherwise completed. Unknown details fail loudly.
        """

        if detail not in UNAVAILABLE_DETAILS:
            raise ValueError(f"unknown stream-unavailable detail {detail!r}; expected one of {UNAVAILABLE_DETAILS}")
        with self._lock:
            if self.unavailable_detail is None:
                self.unavailable_detail = detail

    @property
    def has_streamed_text(self) -> bool:
        """True once any text of this call was delivered or is queued for delivery."""
        with self._lock:
            return self.emitted_deltas > 0 or bool(self._pending_text)

    def note_for_record(self, detail: str) -> None:
        """Record a gap on the durable record without changing the end reason."""
        if detail not in UNAVAILABLE_DETAILS:
            raise ValueError(f"unknown stream-unavailable detail {detail!r}; expected one of {UNAVAILABLE_DETAILS}")
        with self._lock:
            if self.record_only_detail is None:
                self.record_only_detail = detail

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def end(self, reason: str, detail: Optional[str] = None) -> None:
        """Flush pending text and emit the call's single `llm.delta_end`.

        A call that `completed` but was marked unavailable ends as
        `reason: "unavailable"` with its detail; failed and cancelled keep their
        reason (the failure is the more important fact).
        """

        if reason not in END_REASONS:
            raise ValueError(f"unknown delta_end reason {reason!r}; expected one of {END_REASONS}")
        if detail is not None:
            self.mark_unavailable(detail)
        with self._lock:
            if self._ended:
                return
            if reason in ("completed", "unavailable") and self.unavailable_detail is not None:
                reason = "unavailable"
            if reason == "unavailable" and self.unavailable_detail is None:
                raise ValueError("delta_end reason 'unavailable' requires a detail")
            self._flush_locked()
            self._ended = True
            self._cancel_timer_locked()
            event: Dict[str, Any] = {
                "kind": "llm.delta_end",
                "run_id": self.run_id,
                "parent_run_id": self.parent_run_id,
                "node_id": self.node_id,
                "call_id": self.call_id,
                "seq": self._seq,
                "reason": reason,
            }
            if reason == "unavailable":
                event["detail"] = self.unavailable_detail
            # A sink that failed mid-call gets one more chance for the end
            # event, so the host can close its live view with the reason.
            if self._sink is None and self._failed_sink is not None:
                self._sink = self._failed_sink
            self._emit_locked(event)

    @property
    def ended(self) -> bool:
        return self._ended

    # -- internals ---------------------------------------------------------
    def _arm_timer_locked(self, delay: float) -> None:
        if self._timer is not None:
            return
        timer = threading.Timer(max(0.0, delay), self._on_timer)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _cancel_timer_locked(self) -> None:
        timer = self._timer
        self._timer = None
        if timer is not None:
            timer.cancel()

    def _on_timer(self) -> None:
        with self._lock:
            self._timer = None
            if not self._ended:
                self._flush_locked()

    def _flush_locked(self) -> None:
        self._cancel_timer_locked()
        if not self._pending_text:
            return
        text = "".join(self._pending_text)
        channel = self._pending_channel or "content"
        self._pending_text = []
        self._pending_channel = None
        self._last_emit = self._clock()
        self._emit_locked(
            {
                "kind": "llm.delta",
                "run_id": self.run_id,
                "parent_run_id": self.parent_run_id,
                "node_id": self.node_id,
                "call_id": self.call_id,
                "seq": self._seq,
                "text": text,
                "channel": channel,
            }
        )
        self.emitted_deltas += 1

    def _emit_locked(self, event: Dict[str, Any]) -> None:
        # seq advances even when the sink is gone: the numbering describes the
        # call, not the delivery.
        self._seq += 1
        sink = self._sink
        if sink is None:
            return
        try:
            sink(event)
        except Exception as exc:
            self._failed_sink = self._sink if self._failed_sink is None else None
            self._sink = None
            if self.unavailable_detail is None:
                self.unavailable_detail = "sink_error"
            logger.warning(
                "live delta sink raised; disabled for the rest of call %s (run %s): %r",
                self.call_id,
                self.run_id,
                exc,
            )


__all__ = [
    "UNAVAILABLE_DETAILS",
    "DEFAULT_FLUSH_INTERVAL_S",
    "DELTA_CHANNELS",
    "END_REASONS",
    "LiveDeltaEmitter",
    "LiveDeltaSink",
]
