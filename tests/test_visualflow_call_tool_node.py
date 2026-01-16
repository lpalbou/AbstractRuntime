from __future__ import annotations

from typing import Any, Dict

from abstractruntime.core.models import EffectType, RunStatus
from abstractruntime.core.runtime import Runtime
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow


def _make_runtime(*, tool_executor: MappingToolExecutor) -> Runtime:
    return Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tool_executor)},
    )


def test_visualflow_call_tool_node_maps_success_output() -> None:
    def ok_tool(*, x: int) -> Dict[str, Any]:
        return {"ok": True, "x": x}

    runtime = _make_runtime(tool_executor=MappingToolExecutor.from_tools([ok_tool]))

    flow = {
        "id": "call-tool-ok",
        "name": "call-tool-ok",
        "entryNode": "node-1",
        "nodes": [
            {
                "id": "node-1",
                "type": "on_flow_start",
                "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "node-2",
                "type": "call_tool",
                "data": {
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "tool_call", "label": "tool_call", "type": "object"},
                        {"id": "allowed_tools", "label": "allowed_tools", "type": "array"},
                    ],
                    "outputs": [
                        {"id": "exec-out", "label": "", "type": "execution"},
                        {"id": "result", "label": "result", "type": "any"},
                        {"id": "success", "label": "success", "type": "boolean"},
                    ],
                    "pinDefaults": {
                        "tool_call": {"name": "ok_tool", "arguments": {"x": 7}, "call_id": "c1"},
                        "allowed_tools": ["ok_tool"],
                    },
                },
            },
            {
                "id": "node-3",
                "type": "on_flow_end",
                "data": {
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "tool_result", "label": "tool_result", "type": "any"},
                        {"id": "tool_success", "label": "tool_success", "type": "boolean"},
                    ],
                    "outputs": [],
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "node-1", "target": "node-2", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e2", "source": "node-2", "target": "node-3", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e3", "source": "node-2", "target": "node-3", "sourceHandle": "result", "targetHandle": "tool_result"},
            {"id": "e4", "source": "node-2", "target": "node-3", "sourceHandle": "success", "targetHandle": "tool_success"},
        ],
    }

    spec = compile_visualflow(flow)
    run_id = runtime.start(workflow=spec, vars={})
    state = runtime.tick(workflow=spec, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    assert isinstance(state.output, dict)
    assert state.output.get("success") is True

    assert "result" not in state.output
    assert state.output.get("tool_success") is True
    assert state.output.get("tool_result") == {"ok": True, "x": 7}


def test_visualflow_call_tool_node_maps_error_output() -> None:
    def boom_tool() -> str:
        raise RuntimeError("boom")

    runtime = _make_runtime(tool_executor=MappingToolExecutor.from_tools([boom_tool]))

    flow = {
        "id": "call-tool-boom",
        "name": "call-tool-boom",
        "entryNode": "node-1",
        "nodes": [
            {
                "id": "node-1",
                "type": "on_flow_start",
                "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "node-2",
                "type": "call_tool",
                "data": {
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "tool_call", "label": "tool_call", "type": "object"},
                        {"id": "allowed_tools", "label": "allowed_tools", "type": "array"},
                    ],
                    "outputs": [
                        {"id": "exec-out", "label": "", "type": "execution"},
                        {"id": "result", "label": "result", "type": "any"},
                        {"id": "success", "label": "success", "type": "boolean"},
                    ],
                    "pinDefaults": {
                        "tool_call": {"name": "boom_tool", "arguments": {}, "call_id": "c1"},
                        "allowed_tools": ["boom_tool"],
                    },
                },
            },
            {
                "id": "node-3",
                "type": "on_flow_end",
                "data": {
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "tool_result", "label": "tool_result", "type": "any"},
                        {"id": "tool_success", "label": "tool_success", "type": "boolean"},
                    ],
                    "outputs": [],
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "node-1", "target": "node-2", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e2", "source": "node-2", "target": "node-3", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e3", "source": "node-2", "target": "node-3", "sourceHandle": "result", "targetHandle": "tool_result"},
            {"id": "e4", "source": "node-2", "target": "node-3", "sourceHandle": "success", "targetHandle": "tool_success"},
        ],
    }

    spec = compile_visualflow(flow)
    run_id = runtime.start(workflow=spec, vars={})
    state = runtime.tick(workflow=spec, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    assert isinstance(state.output, dict)
    assert state.output.get("success") is True

    assert "result" not in state.output
    assert state.output.get("tool_success") is False
    assert isinstance(state.output.get("tool_result"), str)
    assert "boom" in str(state.output.get("tool_result"))
