from __future__ import annotations

from pathlib import Path

from abstractruntime import (
    Effect,
    EffectType,
    RunStatus,
    StepPlan,
    WorkflowSpec,
)
from abstractruntime.core.runtime import Runtime
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor, PassthroughToolExecutor
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore


def test_workspace_policy_tool_calls_persist_across_restart(tmp_path: Path) -> None:
    base = tmp_path / "stores"
    run_store = JsonFileRunStore(base)
    ledger_store = JsonlLedgerStore(base)

    rt1 = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers={
            EffectType.TOOL_CALLS: make_tool_calls_handler(tools=PassthroughToolExecutor(mode="approval_required")),
        },
    )

    workspace_root = tmp_path / "ws"
    vars0 = {
        "workspace_root": str(workspace_root),
        "workspace_access_mode": "workspace_only",
    }

    tool_calls = [
        {"name": "write_file", "arguments": {"file_path": "hello.txt", "content": "hi"}, "call_id": "c1"},
    ]

    def tools_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="TOOLS",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": tool_calls},
                result_key="tool_results",
            ),
            next_node="DONE",
        )

    def done_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(node_id="DONE", complete_output={"tool_results": run.vars.get("tool_results")})

    wf = WorkflowSpec(
        workflow_id="workspace_policy_restart_test",
        entry_node="TOOLS",
        nodes={"TOOLS": tools_node, "DONE": done_node},
    )

    run_id = rt1.start(workflow=wf, vars=vars0)
    state1 = rt1.tick(workflow=wf, run_id=run_id)
    assert state1.status == RunStatus.WAITING
    assert state1.waiting is not None
    assert isinstance(state1.waiting.details, dict)

    wait_calls = state1.waiting.details.get("tool_calls")
    assert isinstance(wait_calls, list) and len(wait_calls) == 1
    args0 = wait_calls[0].get("arguments")
    assert isinstance(args0, dict)
    fp = args0.get("file_path")
    assert isinstance(fp, str) and fp
    resolved = Path(fp)
    assert resolved.is_absolute()
    assert resolved.parent == workspace_root

    # Simulate a restart: new runtime instance, same file-backed stores.
    rt2 = Runtime(
        run_store=JsonFileRunStore(base),
        ledger_store=JsonlLedgerStore(base),
        effect_handlers={
            EffectType.TOOL_CALLS: make_tool_calls_handler(tools=PassthroughToolExecutor(mode="approval_required")),
        },
    )

    state2 = rt2.get_state(run_id)
    assert state2.status == RunStatus.WAITING
    assert state2.waiting is not None
    wait_calls2 = state2.waiting.details.get("tool_calls") if isinstance(state2.waiting.details, dict) else None
    assert wait_calls2 == wait_calls

    # Host executes the tool calls and resumes.
    from abstractcore.tools.common_tools import write_file

    host = MappingToolExecutor.from_tools([write_file])
    tool_results = host.execute(tool_calls=wait_calls2)
    resumed = rt2.resume(workflow=wf, run_id=run_id, wait_key=state2.waiting.wait_key, payload=tool_results)

    assert resumed.status == RunStatus.COMPLETED
    assert (workspace_root / "hello.txt").read_text(encoding="utf-8") == "hi"

