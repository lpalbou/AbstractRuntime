"""abstractruntime.core.health

RuntimeHealth — counters, not folklore (backlog 0054).

The runtime's self-knowledge used to be `logger.warning` at ~25 sites plus
per-feature stats objects nothing aggregates. "Is the runtime healthy?" was
answered by reading logs or sampling a live process (the H7c starvation was
diagnosed with macOS `sample`). A 24/7 fleet needs a pulse, and the gateway
needs something to serve.

Shape, deliberately minimal:
- monotonic COUNTERS incremented at the existing warning/decision sites
  (an `int += 1` under one lock — no new hot-path work);
- coarse TICK-DURATION buckets (a histogram nobody has to configure);
- a small LAST-ERRORS ring (the first diagnostic is "what broke last",
  not a log grep);
- gauges for the 0053 vars-size watch (last/max serialized bytes).

`snapshot()` returns a plain JSON-safe dict — no metrics framework, no
export dependency; hosts serve it however they like (the gateway's ops
surface is the first consumer). Counters are process-lifetime: persistence
would imply a database for what is honestly a pulse.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Optional

__all__ = ["RuntimeHealth", "TICK_DURATION_BUCKETS_MS"]

# Upper bounds (ms) of the coarse tick-duration histogram; the last bucket
# is open-ended. Coarse on purpose: the question is "are ticks degrading",
# not "what is p99.9".
TICK_DURATION_BUCKETS_MS = (10.0, 50.0, 250.0, 1000.0, 5000.0)

_LAST_ERRORS_RING_SIZE = 16


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RuntimeHealth:
    """Thread-safe counter surface for one Runtime instance.

    All mutators are cheap and exception-free by construction (observability
    must never fail the thing it observes); `snapshot()` is the only read
    and copies everything so callers can never alias live state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {}
        self._tick_buckets = [0] * (len(TICK_DURATION_BUCKETS_MS) + 1)
        self._last_errors: deque = deque(maxlen=_LAST_ERRORS_RING_SIZE)
        self._gauges: Dict[str, float] = {}
        self._started_at = _utc_now_iso()

    # -- mutators (call sites: existing warning/decision points) ----------

    def increment(self, name: str, amount: int = 1) -> None:
        try:
            with self._lock:
                self._counters[name] = self._counters.get(name, 0) + int(amount)
        except Exception:  # pragma: no cover - never fail the observed path
            pass

    def observe_tick(self, duration_ms: float) -> None:
        try:
            idx = len(TICK_DURATION_BUCKETS_MS)
            for i, bound in enumerate(TICK_DURATION_BUCKETS_MS):
                if duration_ms < bound:
                    idx = i
                    break
            with self._lock:
                self._counters["ticks_total"] = self._counters.get("ticks_total", 0) + 1
                self._tick_buckets[idx] += 1
        except Exception:  # pragma: no cover
            pass

    def record_error(self, site: str, message: str) -> None:
        """Ring entry + per-site counter. `site` is a stable short name
        (e.g. "steer_drain", "terminal_append") — the ring answers "what
        broke last" without a log grep."""
        try:
            with self._lock:
                self._counters[f"errors_{site}_total"] = self._counters.get(f"errors_{site}_total", 0) + 1
                self._last_errors.append(
                    {"ts": _utc_now_iso(), "site": str(site), "message": str(message)[:500]}
                )
        except Exception:  # pragma: no cover
            pass

    def set_gauge(self, name: str, value: float) -> None:
        """Last-value gauge; `<name>_max` is maintained automatically."""
        try:
            with self._lock:
                self._gauges[name] = float(value)
                max_key = f"{name}_max"
                if float(value) > self._gauges.get(max_key, float("-inf")):
                    self._gauges[max_key] = float(value)
        except Exception:  # pragma: no cover
            pass

    # -- the one read ------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            buckets: Dict[str, int] = {}
            for i, bound in enumerate(TICK_DURATION_BUCKETS_MS):
                buckets[f"lt_{int(bound)}ms"] = self._tick_buckets[i]
            buckets[f"ge_{int(TICK_DURATION_BUCKETS_MS[-1])}ms"] = self._tick_buckets[-1]
            return {
                "started_at": self._started_at,
                "counters": dict(self._counters),
                "tick_duration_buckets": buckets,
                "gauges": dict(self._gauges),
                "last_errors": list(self._last_errors),
            }
