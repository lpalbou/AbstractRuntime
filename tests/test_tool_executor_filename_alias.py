from __future__ import annotations

from pathlib import Path

from abstractcore.tools.common_tools import read_file
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor


def test_mapping_tool_executor_maps_filename_to_file_path_for_read_file(tmp_path: Path) -> None:
    target = tmp_path / "demo.txt"
    target.write_text("hello\n", encoding="utf-8")

    exe = MappingToolExecutor.from_tools([read_file])
    result = exe.execute(
        tool_calls=[
            {
                "call_id": "1",
                "name": "read_file",
                "arguments": {"filename": str(target)},
            }
        ]
    )

    assert result["results"][0]["success"] is True
    out = result["results"][0]["output"]
    assert isinstance(out, str)
    assert "hello" in out



