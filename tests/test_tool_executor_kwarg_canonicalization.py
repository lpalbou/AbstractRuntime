from __future__ import annotations

from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor


def _echo_file_path(*, file_path: str) -> str:
    return file_path


def _echo_search(*, pattern: str, path: str = ".") -> str:
    return f"{pattern}@{path}"


def test_mapping_tool_executor_normalizes_keys_by_separators_and_case() -> None:
    exe = MappingToolExecutor.from_tools([_echo_file_path])
    result = exe.execute(
        tool_calls=[
            {
                "name": "_echo_file_path",
                "arguments": {"filePath": "demo.txt"},
                "call_id": "1",
            }
        ]
    )
    assert result["results"][0]["success"] is True
    assert result["results"][0]["output"] == "demo.txt"


def test_mapping_tool_executor_applies_small_synonym_table() -> None:
    exe = MappingToolExecutor.from_tools([_echo_search])
    result = exe.execute(
        tool_calls=[
            {
                "name": "_echo_search",
                "arguments": {"query": "pygame", "directory": "/tmp"},
                "call_id": "1",
            }
        ]
    )
    assert result["results"][0]["success"] is True
    assert result["results"][0]["output"] == "pygame@/tmp"


def test_mapping_tool_executor_parses_json_string_arguments() -> None:
    exe = MappingToolExecutor.from_tools([_echo_file_path])
    result = exe.execute(
        tool_calls=[
            {
                "name": "_echo_file_path",
                "arguments": "{\"file-path\": \"demo.txt\"}",
                "call_id": "1",
            }
        ]
    )
    assert result["results"][0]["success"] is True
    assert result["results"][0]["output"] == "demo.txt"


