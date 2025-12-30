from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor


def test_mapping_tool_executor_marks_error_string_outputs_as_failure() -> None:
    def tool_returns_error() -> str:
        return "Error: Something went wrong"

    executor = MappingToolExecutor.from_tools([tool_returns_error])
    result = executor.execute(tool_calls=[{"name": "tool_returns_error", "arguments": {}, "call_id": "c1"}])

    assert result["mode"] == "executed"
    assert result["results"][0]["success"] is False
    assert result["results"][0]["output"] is None
    assert result["results"][0]["error"] == "Something went wrong"


def test_mapping_tool_executor_marks_cross_mark_outputs_as_failure() -> None:
    def tool_returns_cross() -> str:
        return "❌ Permission denied: Cannot write"

    executor = MappingToolExecutor.from_tools([tool_returns_cross])
    result = executor.execute(tool_calls=[{"name": "tool_returns_cross", "arguments": {}, "call_id": "c1"}])

    assert result["mode"] == "executed"
    assert result["results"][0]["success"] is False
    assert result["results"][0]["output"] is None
    assert result["results"][0]["error"] == "Permission denied: Cannot write"

