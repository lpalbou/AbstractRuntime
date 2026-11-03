import pytest

from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor

pytestmark = pytest.mark.basic


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


def test_mapping_tool_executor_marks_json_error_outputs_as_failure() -> None:
    def tool_returns_json_error() -> str:
        return '{"success":false,"status_hint":"error","error":"requests is not installed","results":[]}'

    executor = MappingToolExecutor.from_tools([tool_returns_json_error])
    result = executor.execute(tool_calls=[{"name": "tool_returns_json_error", "arguments": {}, "call_id": "c1"}])

    assert result["mode"] == "executed"
    assert result["results"][0]["success"] is False
    # Backlog 1005: the tool's own structured explanation is kept for the agent and the ledger.
    assert result["results"][0]["output"] == {"success": False, "status_hint": "error", "error": "requests is not installed", "results": []}
    assert result["results"][0]["error"] == "requests is not installed"


def test_a_failure_reported_as_a_json_object_string_keeps_its_diagnostics() -> None:
    """Backlog 1005: a web search blocked by a bot challenge reported its backends, hint and warnings; the
    runtime used to keep only the error line, so neither the agent nor the person could tell "blocked" from
    "nothing exists"."""
    payload = ('{"success": false, "error": "No results found", "hint": "retry fewer searches at once", '
               '"backend_attempts": [{"backend": "ddgs", "error": "No results found."}, {"backend": "html", "status": 202}]}')

    def web_search() -> str:
        return payload

    result = MappingToolExecutor.from_tools([web_search]).execute(tool_calls=[{"name": "web_search", "arguments": {}, "call_id": "c1"}])
    row = result["results"][0]
    assert row["success"] is False and row["error"] == "No results found"
    assert row["output"]["hint"] == "retry fewer searches at once"
    assert [a["backend"] for a in row["output"]["backend_attempts"]] == ["ddgs", "html"]


def test_a_failure_reported_as_a_dict_is_kept_as_before() -> None:
    def tool_returns_dict() -> dict:
        return {"ok": False, "error": "quota reached", "retry_after_s": 30}

    row = MappingToolExecutor.from_tools([tool_returns_dict]).execute(
        tool_calls=[{"name": "tool_returns_dict", "arguments": {}, "call_id": "c1"}])["results"][0]
    assert row["success"] is False and row["output"] == {"ok": False, "error": "quota reached", "retry_after_s": 30}


def test_a_braced_error_string_that_is_not_json_keeps_an_empty_output() -> None:
    def tool_returns_braced_error() -> str:
        return "Error: {not json}"

    row = MappingToolExecutor.from_tools([tool_returns_braced_error]).execute(
        tool_calls=[{"name": "tool_returns_braced_error", "arguments": {}, "call_id": "c1"}])["results"][0]
    assert row["success"] is False and row["output"] is None and row["error"] == "{not json}"
