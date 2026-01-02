from __future__ import annotations

from typing import Any, Dict, List

from abstractruntime import (
    Effect,
    EffectType,
    InMemoryLedgerStore,
    InMemoryRunStore,
    RunStatus,
    StepPlan,
    WaitReason,
    WorkflowSpec,
)
from abstractruntime.core.runtime import Runtime
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor, PassthroughToolExecutor, ToolExecutor


def _make_runtime(*, tool_executor: ToolExecutor) -> Runtime:
    return Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tool_executor)},
    )


def test_passthrough_tool_wait_filters_disallowed_calls_and_resume_merges_blocked_results() -> None:
    def allowed_tool(*, x: int) -> Dict[str, Any]:
        return {"ok": True, "x2": x * 2}

    runtime = _make_runtime(tool_executor=PassthroughToolExecutor(mode="approval_required"))

    tool_calls: List[Dict[str, Any]] = [
        {"name": "allowed_tool", "arguments": {"x": 2}, "call_id": "c1"},
        {"name": "blocked_tool", "arguments": {"y": 123}, "call_id": "c2"},
    ]

    def tools_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="tools",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": tool_calls, "allowed_tools": ["allowed_tool"]},
                result_key="tool_results",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="done",
            complete_output={"tool_results": run.vars.get("tool_results")},
        )

    workflow = WorkflowSpec(
        workflow_id="tool_wait_allowlist_test",
        entry_node="tools",
        nodes={"tools": tools_node, "done": done_node},
    )

    run_id = runtime.start(workflow=workflow)
    state = runtime.tick(workflow=workflow, run_id=run_id)

    assert state.status == RunStatus.WAITING
    assert state.waiting is not None
    assert state.waiting.reason == WaitReason.EVENT
    assert isinstance(state.waiting.details, dict)

    # Only allowlist-safe tool calls are exposed to the host for execution.
    wait_calls = state.waiting.details.get("tool_calls")
    assert isinstance(wait_calls, list)
    assert [c.get("name") for c in wait_calls] == ["allowed_tool"]

    # Blocked entries are tracked for deterministic merge on resume.
    assert state.waiting.details.get("original_call_count") == 2
    blocked = state.waiting.details.get("blocked_by_index")
    assert isinstance(blocked, dict)
    assert isinstance(blocked.get("1"), dict)
    assert blocked["1"].get("name") == "blocked_tool"

    # Host executes only the allowed call(s) and resumes with results for that subset.
    host_runner = MappingToolExecutor({"allowed_tool": allowed_tool})
    tool_results = host_runner.execute(tool_calls=wait_calls)

    resumed = runtime.resume(
        workflow=workflow,
        run_id=run_id,
        wait_key=state.waiting.wait_key,
        payload=tool_results,
    )

    assert resumed.status == RunStatus.COMPLETED
    out = resumed.output or {}
    tool_results2 = out.get("tool_results")
    assert isinstance(tool_results2, dict)
    results = tool_results2.get("results")
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].get("name") == "allowed_tool"
    assert results[1].get("name") == "blocked_tool"
    assert "not allowed" in str(results[1].get("error") or "")


def test_delegated_tool_wait_sets_wait_reason_job_and_uses_executor_wait_key() -> None:
    class DelegatedExecutor:
        def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {
                "mode": "delegated",
                "wait_key": "job_123",
                "wait_reason": "job",
                "details": {"worker": "w1"},
            }

    runtime = _make_runtime(tool_executor=DelegatedExecutor())

    def tools_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="tools",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": [{"name": "noop", "arguments": {}, "call_id": "c1"}]},
                result_key="tool_results",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(node_id="done", complete_output={"ok": True})

    workflow = WorkflowSpec(
        workflow_id="tool_wait_job_test",
        entry_node="tools",
        nodes={"tools": tools_node, "done": done_node},
    )

    run_id = runtime.start(workflow=workflow)
    state = runtime.tick(workflow=workflow, run_id=run_id)

    assert state.status == RunStatus.WAITING
    assert state.waiting is not None
    assert state.waiting.reason == WaitReason.JOB
    assert state.waiting.wait_key == "job_123"
    assert isinstance(state.waiting.details, dict)
    assert state.waiting.details.get("mode") == "delegated"
    assert isinstance(state.waiting.details.get("executor"), dict)
    assert state.waiting.details["executor"].get("worker") == "w1"

