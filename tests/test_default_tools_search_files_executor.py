from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def test_default_tool_executor_search_files_respects_max_hits_and_head_limit(tmp_path: Path) -> None:
    from abstractruntime.integrations.abstractcore.default_tools import get_default_tools, list_default_tool_specs
    from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor

    specs = list_default_tool_specs()
    search = next((s for s in specs if isinstance(s, dict) and s.get("name") == "search_files"), None)
    assert isinstance(search, dict)
    params = search.get("parameters") or {}
    assert isinstance(params, dict)
    assert "max_hits" in params
    assert "output_mode" not in params
    assert "context_lines" not in params
    assert "case_sensitive" not in params
    assert "ignore_dirs" not in params

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    a.write_text("TODO a1\nTODO a2\n", encoding="utf-8")
    b.write_text("TODO b1\nTODO b2\n", encoding="utf-8")
    c.write_text("TODO c1\nTODO c2\n", encoding="utf-8")

    executor = MappingToolExecutor.from_tools(get_default_tools())
    result = executor.execute(
        tool_calls=[
            {
                "call_id": "1",
                "name": "search_files",
                "arguments": {
                    "pattern": "TODO",
                    "path": str(tmp_path),
                    "file_pattern": "*.txt",
                    "head_limit": 1,
                    "max_hits": 2,
                },
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
    assert out.count("\n📄 ") == 2
    assert sum(int(str(p) in out) for p in (a, b, c)) == 2
    assert "TODO a2" not in out
    assert "TODO b2" not in out
    assert "TODO c2" not in out
