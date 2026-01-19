from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def test_default_tool_specs_include_skim_files_and_executor_runs(tmp_path: Path) -> None:
    from abstractruntime.integrations.abstractcore.default_tools import get_default_tools, list_default_tool_specs
    from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor

    specs = list_default_tool_specs()
    names = {s.get("name") for s in specs if isinstance(s, dict)}
    assert "skim_files" in names

    target = tmp_path / "demo.txt"
    target.write_text("one.\n\ntwo.\n\nthree.\n", encoding="utf-8")

    executor = MappingToolExecutor.from_tools(get_default_tools())
    result = executor.execute(
        tool_calls=[
            {
                "call_id": "1",
                "name": "skim_files",
                "arguments": {"paths": [str(target)], "target_percent": 8.0},
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
    assert "File:" in out
    assert "1:" in out
