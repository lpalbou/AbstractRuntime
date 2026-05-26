"""Small process-level execution metrics for VisualFlow node runs."""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, Optional


def process_rss_mb() -> Optional[float]:
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if rss <= 0:
        return None
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(rss / divisor, 3)


def capture_execution_start() -> Dict[str, Optional[float]]:
    return {
        "wall": time.perf_counter(),
        "cpu": time.process_time(),
        "rss_mb": process_rss_mb(),
    }


def finish_execution_metrics(start: Dict[str, Optional[float]]) -> Dict[str, Any]:
    wall_started = float(start.get("wall") or 0.0)
    cpu_started = float(start.get("cpu") or 0.0)
    wall_finished = time.perf_counter()
    cpu_finished = time.process_time()
    duration_ms = max(0.0, (wall_finished - wall_started) * 1000.0)
    cpu_time_ms = max(0.0, (cpu_finished - cpu_started) * 1000.0)

    metrics: Dict[str, Any] = {
        "duration_ms": round(duration_ms, 3),
        "cpu_time_ms": round(cpu_time_ms, 3),
    }
    if duration_ms > 0:
        metrics["cpu_percent"] = round((cpu_time_ms / duration_ms) * 100.0, 3)

    rss_started = start.get("rss_mb")
    rss_finished = process_rss_mb()
    if rss_finished is not None:
        metrics["memory_rss_mb"] = rss_finished
    if rss_started is not None and rss_finished is not None:
        metrics["memory_rss_delta_mb"] = round(rss_finished - rss_started, 3)
    return metrics
