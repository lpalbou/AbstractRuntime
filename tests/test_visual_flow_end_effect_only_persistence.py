from __future__ import annotations

import json
from pathlib import Path

import pytest

from abstractruntime import Runtime
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.visualflow_compiler import compile_visualflow


def _node(node_id: str, node_type: str, **data) -> dict:
    return {"id": node_id, "type": node_type, "data": {"nodeType": node_type, **data}}


def _edge(source: str, target: str, source_handle="exec-out", target_handle="exec-in") -> dict:
    return {"source": source, "target": target, "sourceHandle": source_handle, "targetHandle": target_handle}


def _runtime(root: Path) -> Runtime:
    return Runtime(run_store=JsonFileRunStore(root), ledger_store=JsonlLedgerStore(root))


@pytest.mark.parametrize("event_entry", [False, True])
def test_effect_only_pinless_end_persists_public_output(tmp_path: Path, event_entry: bool) -> None:
    nodes = [
        _node("answer", "answer_user", pinDefaults={"message": "Event delivered", "level": "message"}),
        _node("end", "on_flow_end"),
    ]
    edges = [_edge("answer", "end")]
    if event_entry:
        nodes.insert(0, _node("listen", "on_event", eventConfig={"name": "fixture.ping", "scope": "session"}))
        edges.insert(0, _edge("listen", "answer"))
    flow = {
        "id": "effect-only-end", "name": "Effect only end", "nodes": nodes, "edges": edges,
        "entryNode": "listen" if event_entry else "answer",
    }
    workflow = compile_visualflow(flow)
    runtime = _runtime(tmp_path)
    run_id = runtime.start(workflow=workflow, vars={"public_input": {"value": 42}}, session_id="fixture-session")
    state = runtime.tick(workflow=workflow, run_id=run_id)
    if event_entry:
        assert state.status == RunStatus.WAITING
        assert state.waiting is not None
        assert state.waiting.wait_key.endswith(":fixture.ping")
        # Rehydrate both the workflow cache and the durable state before delivery.
        runtime = _runtime(tmp_path)
        workflow = compile_visualflow(flow)
        state = runtime.resume(workflow=workflow, run_id=run_id, wait_key=state.waiting.wait_key, payload={"ping": True})
    assert state.status == RunStatus.COMPLETED
    assert state.output == {"public_input": {"value": 42}, "success": True}
    fresh = _runtime(tmp_path).get_state(run_id)
    assert fresh.status == RunStatus.COMPLETED
    assert fresh.output == state.output
    json.dumps(fresh.vars)
    json.dumps(fresh.output)
    assert not any(key.startswith("_") for key in fresh.output)


@pytest.mark.parametrize("connected_result", [False, True])
def test_flow_end_keeps_authored_output_including_private_looking_keys(tmp_path: Path, connected_result: bool) -> None:
    end_data = {"inputs": [{"id": "exec-in", "type": "execution"}, {"id": "result", "type": "object"}]} if connected_result else {}
    edges = [_edge("build", "end")]
    if connected_result:
        edges.append(_edge("build", "end", "output", "result"))
    workflow = compile_visualflow({
        "id": "authored-output", "name": "Authored output", "entryNode": "build",
        "nodes": [
            _node("build", "code", codeBody="return {'_application_field': 'keep me', 'value': 42}"),
            _node("end", "on_flow_end", **end_data),
        ],
        "edges": edges,
    })
    runtime = _runtime(tmp_path)
    run_id = runtime.start(workflow=workflow, vars={})
    state = runtime.tick(workflow=workflow, run_id=run_id)
    value = {"_application_field": "keep me", "value": 42}
    expected = {"result": value, "success": True} if connected_result else {**value, "success": True}
    assert state.status == RunStatus.COMPLETED
    assert state.output == expected
    fresh = _runtime(tmp_path).get_state(run_id)
    assert fresh.status == RunStatus.COMPLETED
    assert fresh.output == expected
    json.dumps(fresh.vars)
