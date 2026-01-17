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
	                        {"id": "context_extra", "label": "context_extra", "type": "object"},
	                    ],
	                    "outputs": [{"id": "context", "label": "context", "type": "object"}],
	                    "pinDefaults": {
	                        "task": "hello",
	                        "messages": [{"role": "user", "content": "hi"}],
	                        "context_extra": {"foo": 1},
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
	                        {"id": "task", "label": "task", "type": "string"},
	                        {"id": "messages", "label": "messages", "type": "array"},
	                        {"id": "context_extra", "label": "context_extra", "type": "object"},
	                        {"id": "node_traces", "label": "node_traces", "type": "object"},
	                        {"id": "steps", "label": "steps", "type": "array"},
	                        {"id": "tool_calls", "label": "tool_calls", "type": "array"},
	                        {"id": "tool_results", "label": "tool_results", "type": "array"},
	                    ],
	                    "outputs": [{"id": "scratchpad", "label": "scratchpad", "type": "object"}],
	                    "pinDefaults": {
	                        "sub_run_id": "sub-1",
	                        "workflow_id": "wf-1",
	                        "task": "scratch-task",
	                        "messages": [{"role": "user", "content": "hi"}],
	                        "context_extra": {"foo": "bar"},
	                        "node_traces": {"node-9": {"node_id": "node-9", "steps": []}},
	                        "steps": [{"type": "llm_call"}],
	                        "tool_calls": [],
	                        "tool_results": [],
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
    assert "result" not in state.output

    result = state.output

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
    assert result["built_scratchpad"]["task"] == "scratch-task"
    assert result["built_scratchpad"]["messages"] == [{"role": "user", "content": "hi"}]
    assert result["built_scratchpad"]["context_extra"]["foo"] == "bar"
    assert result["built_scratchpad"]["node_traces"]["node-9"]["node_id"] == "node-9"
    assert result["built_scratchpad"]["steps"][0]["type"] == "llm_call"
