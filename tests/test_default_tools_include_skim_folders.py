from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def test_default_tool_specs_include_skim_folders_and_executor_runs(tmp_path: Path) -> None:
    from abstractruntime.integrations.abstractcore.default_tools import get_default_tools, list_default_tool_specs
    from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor

    specs = list_default_tool_specs()
    names = {s.get("name") for s in specs if isinstance(s, dict)}
    assert "skim_folders" in names

    skim = next((s for s in specs if isinstance(s, dict) and s.get("name") == "skim_folders"), None)
    assert isinstance(skim, dict)
    params = skim.get("parameters") or {}
    assert isinstance(params, dict)
    paths_schema = params.get("paths") or {}
    assert isinstance(paths_schema, dict)
    assert paths_schema.get("type") == "array"

    root = tmp_path / "root"
    root.mkdir()
    (root / "docs").mkdir()
    (root / "docs" / "README.md").write_text("x\n", encoding="utf-8")

    executor = MappingToolExecutor.from_tools(get_default_tools())
    result = executor.execute(
        tool_calls=[
            {
                "call_id": "1",
                "name": "skim_folders",
                "arguments": {"paths": [str(root)], "max_depth": 2},
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
    assert "Folder:" in out
    assert "docs/ (" in out

