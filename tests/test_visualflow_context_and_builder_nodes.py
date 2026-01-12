from __future__ import annotations

from abstractruntime import Runtime
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow


def test_visualflow_get_context_and_builder_nodes() -> None:
    flow = {
        "id": "context-and-builders",
        "name": "context-and-builders",
        "entryNode": "node-1",
        "nodes": [
            {
                "id": "node-1",
                "type": "on_flow_start",
                "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "node-2",
                "type": "make_context",
                "data": {
                    "inputs": [
                        {"id": "task", "label": "task", "type": "string"},
                        {"id": "messages", "label": "messages", "type": "array"},
                        {"id": "extra", "label": "extra", "type": "object"},
                    ],
                    "outputs": [{"id": "context", "label": "context", "type": "object"}],
                    "pinDefaults": {
                        "task": "hello",
                        "messages": [{"role": "user", "content": "hi"}],
                        "extra": {"foo": 1},
                    },
                },
            },
            {
                "id": "node-3",
                "type": "make_meta",
                "data": {
                    "inputs": [
                        {"id": "schema", "label": "schema", "type": "string"},
                        {"id": "version", "label": "version", "type": "number"},
                        {"id": "provider", "label": "provider", "type": "provider"},
                        {"id": "model", "label": "model", "type": "model"},
                        {"id": "usage", "label": "usage", "type": "object"},
                        {"id": "trace_id", "label": "trace_id", "type": "string"},
                        {"id": "warnings", "label": "warnings", "type": "array"},
                        {"id": "debug", "label": "debug", "type": "object"},
                        {"id": "extra", "label": "extra", "type": "object"},
                    ],
                    "outputs": [{"id": "meta", "label": "meta", "type": "object"}],
                    "pinDefaults": {
                        "provider": "lmstudio",
                        "model": "qwen/qwen3-next-80b",
                        "usage": {"input_tokens": 1, "output_tokens": 2},
                        "trace_id": "trace-123",
                        "warnings": ["w1"],
                        "debug": {"a": 1},
                        "extra": {"schema": "should_be_overridden"},
                    },
                },
            },
            {
                "id": "node-4",
                "type": "make_scratchpad",
                "data": {
                    "inputs": [
                        {"id": "sub_run_id", "label": "sub_run_id", "type": "string"},
                        {"id": "workflow_id", "label": "workflow_id", "type": "string"},
                        {"id": "node_traces", "label": "node_traces", "type": "object"},
                        {"id": "steps", "label": "steps", "type": "array"},
                        {"id": "extra", "label": "extra", "type": "object"},
                    ],
                    "outputs": [{"id": "scratchpad", "label": "scratchpad", "type": "object"}],
                    "pinDefaults": {
                        "sub_run_id": "sub-1",
                        "workflow_id": "wf-1",
                        "node_traces": {"node-9": {"node_id": "node-9", "steps": []}},
                        "steps": [{"type": "llm_call"}],
                        "extra": {"foo": "bar"},
                    },
                },
            },
            {
                "id": "node-5",
                "type": "make_raw_result",
                "data": {
                    "inputs": [
                        {"id": "content", "label": "content", "type": "string"},
                        {"id": "data", "label": "data", "type": "object"},
                        {"id": "tool_calls", "label": "tool_calls", "type": "array"},
                        {"id": "usage", "label": "usage", "type": "object"},
                        {"id": "model", "label": "model", "type": "model"},
                        {"id": "finish_reason", "label": "finish_reason", "type": "string"},
                        {"id": "metadata", "label": "metadata", "type": "object"},
                        {"id": "trace_id", "label": "trace_id", "type": "string"},
                        {"id": "extra", "label": "extra", "type": "object"},
                    ],
                    "outputs": [{"id": "raw_result", "label": "raw_result", "type": "object"}],
                    "pinDefaults": {
                        "content": "ok",
                        "data": {"k": "v"},
                        "tool_calls": [],
                        "usage": {"total": 3},
                        "model": "m1",
                        "finish_reason": "stop",
                        "metadata": {"debug": True},
                        "trace_id": "trace-x",
                        "extra": {"model": "should_be_overridden"},
                    },
                },
            },
            {
                "id": "node-6",
                "type": "get_context",
                "data": {
                    "inputs": [],
                    "outputs": [
                        {"id": "context", "label": "context", "type": "object"},
                        {"id": "task", "label": "task", "type": "string"},
                        {"id": "messages", "label": "messages", "type": "array"},
                    ],
                },
            },
            {
                "id": "node-7",
                "type": "on_flow_end",
                "data": {
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "built_context", "label": "built_context", "type": "object"},
                        {"id": "built_meta", "label": "built_meta", "type": "object"},
                        {"id": "built_scratchpad", "label": "built_scratchpad", "type": "object"},
                        {"id": "built_raw_result", "label": "built_raw_result", "type": "object"},
                        {"id": "got_context", "label": "got_context", "type": "object"},
                        {"id": "got_task", "label": "got_task", "type": "string"},
                        {"id": "got_messages", "label": "got_messages", "type": "array"},
                    ],
                    "outputs": [],
                },
            },
        ],
        "edges": [
            {"id": "e-exec", "source": "node-1", "target": "node-7", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e-bc", "source": "node-2", "target": "node-7", "sourceHandle": "context", "targetHandle": "built_context"},
            {"id": "e-bm", "source": "node-3", "target": "node-7", "sourceHandle": "meta", "targetHandle": "built_meta"},
            {"id": "e-bs", "source": "node-4", "target": "node-7", "sourceHandle": "scratchpad", "targetHandle": "built_scratchpad"},
            {"id": "e-br", "source": "node-5", "target": "node-7", "sourceHandle": "raw_result", "targetHandle": "built_raw_result"},
            {"id": "e-gc", "source": "node-6", "target": "node-7", "sourceHandle": "context", "targetHandle": "got_context"},
            {"id": "e-gt", "source": "node-6", "target": "node-7", "sourceHandle": "task", "targetHandle": "got_task"},
            {"id": "e-gm", "source": "node-6", "target": "node-7", "sourceHandle": "messages", "targetHandle": "got_messages"},
        ],
    }

    spec = compile_visualflow(flow)
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

    run_vars = {"context": {"task": "from_run", "messages": [{"role": "user", "content": "hello"}], "other": True}}

    run_id = runtime.start(workflow=spec, vars=run_vars)
    state = runtime.tick(workflow=spec, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    assert isinstance(state.output, dict)
    assert state.output.get("success") is True
    assert isinstance(state.output.get("result"), dict)

    result = state.output["result"]

    # Get Context
    assert result["got_context"]["task"] == "from_run"
    assert result["got_task"] == "from_run"
    assert result["got_messages"] == [{"role": "user", "content": "hello"}]

    # Make Context
    assert result["built_context"]["task"] == "hello"
    assert result["built_context"]["messages"] == [{"role": "user", "content": "hi"}]
    assert result["built_context"]["foo"] == 1

    # Make Meta (extra.schema overridden by default schema)
    assert result["built_meta"]["schema"] == "abstractcode.agent.v1.meta"
    assert result["built_meta"]["version"] == 1
    assert result["built_meta"]["provider"] == "lmstudio"
    assert result["built_meta"]["model"] == "qwen/qwen3-next-80b"
    assert result["built_meta"]["usage"] == {"input_tokens": 1, "output_tokens": 2}
    assert result["built_meta"]["trace"]["trace_id"] == "trace-123"
    assert result["built_meta"]["warnings"] == ["w1"]
    assert result["built_meta"]["debug"] == {"a": 1}

    # Make Scratchpad
    assert result["built_scratchpad"]["sub_run_id"] == "sub-1"
    assert result["built_scratchpad"]["workflow_id"] == "wf-1"
    assert result["built_scratchpad"]["node_traces"]["node-9"]["node_id"] == "node-9"
    assert result["built_scratchpad"]["steps"][0]["type"] == "llm_call"
    assert result["built_scratchpad"]["foo"] == "bar"

    # Make Raw Result (extra.model overridden by pin model)
    assert result["built_raw_result"]["content"] == "ok"
    assert result["built_raw_result"]["data"] == {"k": "v"}
    assert result["built_raw_result"]["tool_calls"] == []
    assert result["built_raw_result"]["usage"] == {"total": 3}
    assert result["built_raw_result"]["model"] == "m1"
    assert result["built_raw_result"]["finish_reason"] == "stop"
    assert result["built_raw_result"]["metadata"] == {"debug": True}
    assert result["built_raw_result"]["trace_id"] == "trace-x"

