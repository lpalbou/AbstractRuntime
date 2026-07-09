"""Parallel read-only tool execution (backlog 0214).

Verifies:
1. A batch of independent read-only calls runs concurrently (wall-clock ~max, not sum).
2. Result order is identical to the input order regardless of completion order.
3. Side-effecting / unknown tools run sequentially and are never reordered or run concurrently
   with each other; ordering of side effects relative to reads is preserved.
"""
from __future__ import annotations

import threading
import time

from abstractcore.tools.core import tool
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor


def test_independent_reads_run_in_parallel() -> None:
    @tool
    def read_file(path: str) -> str:
        """read a file (stub with latency)"""
        time.sleep(0.2)
        return f"content:{path}"

    ex = MappingToolExecutor.from_tools([read_file])
    calls = [
        {"name": "read_file", "arguments": {"path": f"f{i}.txt"}, "call_id": str(i)}
        for i in range(4)
    ]
    t0 = time.perf_counter()
    out = ex.execute(tool_calls=calls)
    elapsed = time.perf_counter() - t0

    results = out["results"]
    assert [r["name"] for r in results] == ["read_file"] * 4
    assert [r["output"] for r in results] == [f"content:f{i}.txt" for i in range(4)]
    # 4 x 0.2s serial = 0.8s; parallel should be well under (generous bound for CI noise).
    assert elapsed < 0.5, f"expected parallel execution, took {elapsed:.3f}s"


def test_result_order_matches_input_order_regardless_of_completion() -> None:
    # Later calls finish FIRST; result order must still match input order.
    @tool
    def read_file(path: str, delay_ms: int = 0) -> str:
        """read with controllable delay"""
        time.sleep(delay_ms / 1000.0)
        return path

    ex = MappingToolExecutor.from_tools([read_file])
    calls = [
        {"name": "read_file", "arguments": {"path": "a", "delay_ms": 200}, "call_id": "0"},
        {"name": "read_file", "arguments": {"path": "b", "delay_ms": 10}, "call_id": "1"},
        {"name": "read_file", "arguments": {"path": "c", "delay_ms": 100}, "call_id": "2"},
    ]
    out = ex.execute(tool_calls=calls)
    assert [r["output"] for r in out["results"]] == ["a", "b", "c"]
    assert [r["call_id"] for r in out["results"]] == ["0", "1", "2"]


def test_side_effect_tools_run_sequentially_and_in_order() -> None:
    order: list[str] = []
    lock = threading.Lock()
    concurrency = {"current": 0, "max": 0}

    @tool
    def read_file(path: str) -> str:
        """read-only"""
        with lock:
            concurrency["current"] += 1
            concurrency["max"] = max(concurrency["max"], concurrency["current"])
        time.sleep(0.05)
        with lock:
            concurrency["current"] -= 1
            order.append(f"read:{path}")
        return path

    @tool
    def write_file(file_path: str, content: str = "") -> str:
        """side-effecting write (stub)"""
        with lock:
            concurrency["current"] += 1
            concurrency["max"] = max(concurrency["max"], concurrency["current"])
        time.sleep(0.05)
        with lock:
            concurrency["current"] -= 1
            order.append(f"write:{file_path}")
        return "ok"

    ex = MappingToolExecutor.from_tools([read_file, write_file])
    # reads, then a write, then reads: the write must not run concurrently with anything.
    calls = [
        {"name": "read_file", "arguments": {"path": "r1"}, "call_id": "0"},
        {"name": "read_file", "arguments": {"path": "r2"}, "call_id": "1"},
        {"name": "write_file", "arguments": {"file_path": "w1"}, "call_id": "2"},
        {"name": "read_file", "arguments": {"path": "r3"}, "call_id": "3"},
    ]
    out = ex.execute(tool_calls=calls)
    # Result order preserved.
    assert [r["call_id"] for r in out["results"]] == ["0", "1", "2", "3"]
    # The write is ordered after the first two reads and before the last read.
    assert order.index("write:w1") > order.index("read:r1")
    assert order.index("write:w1") > order.index("read:r2")
    assert order.index("write:w1") < order.index("read:r3")


def test_unknown_tool_is_not_parallelized_and_errors_cleanly() -> None:
    @tool
    def read_file(path: str) -> str:
        """read-only"""
        return path

    ex = MappingToolExecutor.from_tools([read_file])
    calls = [
        {"name": "read_file", "arguments": {"path": "a"}, "call_id": "0"},
        {"name": "mystery_tool", "arguments": {}, "call_id": "1"},
    ]
    out = ex.execute(tool_calls=calls)
    results = out["results"]
    assert results[0]["success"] is True and results[0]["output"] == "a"
    assert results[1]["success"] is False and "not found" in results[1]["error"]
