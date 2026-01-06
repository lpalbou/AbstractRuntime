from __future__ import annotations

from abstractruntime.core.config import RuntimeConfig
from abstractruntime.core.models import RunState, StepPlan
from abstractruntime.core.runtime import Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_runtime_start_seeds_tool_support_and_supports_native_tools_from_model_capabilities() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    rt = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        config=RuntimeConfig(
            provider="lmstudio",
            model="qwen/qwen3-next-80b",
            model_capabilities={"tool_support": "native"},
        ),
    )

    def done_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"ok": True})

    wf = WorkflowSpec(workflow_id="seed_tool_support", entry_node="done", nodes={"done": done_node})
    run_id = rt.start(workflow=wf, vars={"context": {}, "_runtime": {}})

    state = run_store.load(run_id)
    assert state is not None
    runtime_ns = state.vars.get("_runtime")
    assert isinstance(runtime_ns, dict)
    assert runtime_ns.get("tool_support") == "native"
    assert runtime_ns.get("supports_native_tools") is True



