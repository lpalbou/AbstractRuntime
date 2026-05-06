from __future__ import annotations

from typing import Any, Dict, List

from abstractruntime import Effect, EffectType, InMemoryLedgerStore, InMemoryRunStore, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.core.runtime import Runtime
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_executor import ApprovalToolExecutor, MappingToolExecutor, ToolApprovalPolicy


def test_tool_approval_resume_executes_tools_in_runtime() -> None:
    def write_file(*, path: str, content: str) -> Dict[str, Any]:
        return {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}

    delegate = MappingToolExecutor({"write_file": write_file})
    tools = ApprovalToolExecutor(delegate=delegate, policy=ToolApprovalPolicy())

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)},
    )
    runtime.set_tool_executor_for_resume(tools)

    tool_calls: List[Dict[str, Any]] = [{"name": "write_file", "arguments": {"path": "a.txt", "content": "hello"}, "call_id": "c1"}]

    def tools_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="tools",
            effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": tool_calls}, result_key="tool_results"),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(node_id="done", complete_output={"tool_results": run.vars.get("tool_results")})

    workflow = WorkflowSpec(workflow_id="tool_approval_resume_executes_test", entry_node="tools", nodes={"tools": tools_node, "done": done_node})

    run_id = runtime.start(workflow=workflow)
    state = runtime.tick(workflow=workflow, run_id=run_id)

    assert state.status == RunStatus.WAITING
    assert state.waiting is not None
    assert isinstance(state.waiting.details, dict)
    assert state.waiting.details.get("mode") == "approval_required"
    assert isinstance(state.waiting.details.get("executor"), dict)
    assert state.waiting.details["executor"].get("kind") == "tool_approval"

    resumed = runtime.resume(workflow=workflow, run_id=run_id, wait_key=state.waiting.wait_key, payload={"approved": True}, max_steps=10)

    assert resumed.status == RunStatus.COMPLETED
    out = resumed.output or {}
    tool_results = out.get("tool_results")
    assert isinstance(tool_results, dict)
    assert tool_results.get("mode") == "executed"
    results = tool_results.get("results")
    assert isinstance(results, list)
    assert results and results[0].get("success") is True
    assert results[0].get("name") == "write_file"


def test_tool_approval_resume_denied_returns_tool_errors() -> None:
    def write_file(*, path: str, content: str) -> Dict[str, Any]:
        return {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}

    delegate = MappingToolExecutor({"write_file": write_file})
    tools = ApprovalToolExecutor(delegate=delegate, policy=ToolApprovalPolicy())

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)},
    )
    runtime.set_tool_executor_for_resume(tools)

    tool_calls: List[Dict[str, Any]] = [{"name": "write_file", "arguments": {"path": "a.txt", "content": "hello"}, "call_id": "c1"}]

    def tools_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="tools",
            effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": tool_calls}, result_key="tool_results"),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(node_id="done", complete_output={"tool_results": run.vars.get("tool_results")})

    workflow = WorkflowSpec(workflow_id="tool_approval_resume_denied_test", entry_node="tools", nodes={"tools": tools_node, "done": done_node})

    run_id = runtime.start(workflow=workflow)
    state = runtime.tick(workflow=workflow, run_id=run_id)
    assert state.status == RunStatus.WAITING
    assert state.waiting is not None

    resumed = runtime.resume(
        workflow=workflow,
        run_id=run_id,
        wait_key=state.waiting.wait_key,
        payload={"approved": False, "reason": "Denied by user"},
        max_steps=10,
    )

    assert resumed.status == RunStatus.COMPLETED
    out = resumed.output or {}
    tool_results = out.get("tool_results")
    assert isinstance(tool_results, dict)
    results = tool_results.get("results")
    assert isinstance(results, list)
    assert results and results[0].get("success") is False
    assert "Denied by user" in str(results[0].get("error") or "")
