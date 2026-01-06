from __future__ import annotations

from pathlib import Path

from abstractcore.tools.common_tools import read_file
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor


def test_mapping_tool_executor_maps_read_file_start_end_aliases(tmp_path: Path) -> None:
    target = tmp_path / "demo.txt"
    target.write_text("a\nb\nc\n", encoding="utf-8")

    executor = MappingToolExecutor.from_tools([read_file])
    result = executor.execute(
        tool_calls=[
            {
                "call_id": "1",
                "name": "read_file",
                "arguments": {"file_path": str(target), "start_line": 2, "end_line": 2},
            }
        ]
    )

    assert isinstance(result, dict)
    results = result.get("results")
    assert isinstance(results, list) and results
    first = results[0]
    assert isinstance(first, dict)
    assert first.get("success") is True
    out = first.get("output")
    assert isinstance(out, str)

    # Should contain only the requested line 2.
    assert "2: b" in out
    assert "1: a" not in out
    assert "3: c" not in out


def test_mapping_tool_executor_maps_read_file_legacy_range_aliases(tmp_path: Path) -> None:
    target = tmp_path / "demo.txt"
    target.write_text("a\nb\nc\n", encoding="utf-8")

    executor = MappingToolExecutor.from_tools([read_file])
    result = executor.execute(
        tool_calls=[
            {
                "call_id": "1",
                "name": "read_file",
                "arguments": {"file_path": str(target), "start_line_one_indexed": 2, "end_line_one_indexed_inclusive": 2},
            }
        ]
    )

    assert isinstance(result, dict)
    results = result.get("results")
    assert isinstance(results, list) and results
    first = results[0]
    assert isinstance(first, dict)
    assert first.get("success") is True
    out = first.get("output")
    assert isinstance(out, str)

    assert "2: b" in out
    assert "1: a" not in out
    assert "3: c" not in out
