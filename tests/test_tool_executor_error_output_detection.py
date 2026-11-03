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


def _agent_observation_for(row: dict) -> str:
    """The tool observation the visual-flow Agent node writes into the run context for this result row."""
    from abstractruntime.core.models import RunState
    from abstractruntime.visualflow_compiler.compiler import compile_visualflow

    spec = compile_visualflow({"id": "f", "name": "f", "nodes": [{"id": "agent", "type": "agent", "data": {"agentConfig": {}}}],
                               "edges": [], "entryNode": "agent"})
    run = RunState.new(workflow_id=spec.workflow_id, entry_node="agent", vars={"context": {}})
    run.vars["_temp"] = {"agent": {"agent": {
        "phase": "subworkflow",
        "resolved_inputs": {"task": "Search the web.", "context": {}, "provider": "lmstudio", "model": "dummy",
                            "include_context": True, "tools": [row["name"]]},
        "sub": {"sub_run_id": "sub-1", "output": {"answer": "done", "iterations": 1}, "node_traces": {"act": {"steps": [{
            "ts": "2026-11-03T00:00:00+00:00", "node_id": "act", "status": "completed",
            "effect": {"type": "tool_calls", "payload": {"tool_calls": [{"name": row["name"], "arguments": {}, "call_id": row["call_id"]}]}},
            "result": {"results": [row]},
        }]}}},
    }}}
    spec.nodes["agent"](run, None)
    observations = [m for m in run.vars["context"]["messages"] if (m.get("metadata") or {}).get("kind") == "tool_observation"]
    assert len(observations) == 1
    return str(observations[0]["content"])


def test_the_agent_node_shows_the_model_a_failed_call_s_explanation_after_its_error() -> None:
    """Backlog 1005: the kept object reaches the model, not only the ledger. The visual-flow Agent node used to
    render a failed call as its error line alone ("[web_search]: Error: No results found")."""
    payload = ('{"success": false, "error": "No results found", "hint": "retry fewer searches at once", '
               '"backend_attempts": [{"backend": "html", "status": 202}]}')

    def web_search() -> str:
        return payload

    row = MappingToolExecutor.from_tools([web_search]).execute(
        tool_calls=[{"name": "web_search", "arguments": {}, "call_id": "c1"}])["results"][0]
    content = _agent_observation_for(row)
    assert content.startswith("[web_search]: Error: No results found\n")
    assert "retry fewer searches at once" in content and '"status": 202' in content


def test_the_agent_node_still_shows_only_the_error_for_a_plain_text_failure() -> None:
    def tool_returns_error() -> str:
        return "Error: Something went wrong"

    row = MappingToolExecutor.from_tools([tool_returns_error]).execute(
        tool_calls=[{"name": "tool_returns_error", "arguments": {}, "call_id": "c1"}])["results"][0]
    assert _agent_observation_for(row) == "[tool_returns_error]: Error: Something went wrong"
