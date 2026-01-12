from __future__ import annotations

from abstractruntime.core.models import EffectType, RunState, RunStatus
from abstractruntime.visualflow_compiler import compile_visualflow


def test_visualflow_compiler_supports_memory_compact_node() -> None:
    flow = {
        "id": "vf_memory_compact",
        "name": "vf_memory_compact",
        "entryNode": "start",
        "nodes": [
            {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
            {"id": "compact", "type": "memory_compact", "data": {"nodeType": "memory_compact"}},
        ],
        "edges": [
            {"source": "start", "sourceHandle": "exec-out", "target": "compact", "targetHandle": "exec-in"},
        ],
    }

    spec = compile_visualflow(flow)
    assert "compact" in spec.nodes

    run = RunState(run_id="run", workflow_id=str(spec.workflow_id), status=RunStatus.RUNNING, current_node="compact", vars={"_temp": {}})
    plan = spec.nodes["compact"](run, None)
    assert plan.effect is not None
    assert plan.effect.type == EffectType.MEMORY_COMPACT


def test_visualflow_compiler_supports_memory_tag_node() -> None:
    flow = {
        "id": "vf_memory_tag",
        "name": "vf_memory_tag",
        "entryNode": "start",
        "nodes": [
            {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
            {"id": "tag", "type": "memory_tag", "data": {"nodeType": "memory_tag"}},
        ],
        "edges": [
            {"source": "start", "sourceHandle": "exec-out", "target": "tag", "targetHandle": "exec-in"},
        ],
    }

    spec = compile_visualflow(flow)
    assert "tag" in spec.nodes

    run = RunState(run_id="run", workflow_id=str(spec.workflow_id), status=RunStatus.RUNNING, current_node="tag", vars={"_temp": {}})
    plan = spec.nodes["tag"](run, None)
    assert plan.effect is not None
    assert plan.effect.type == EffectType.MEMORY_TAG


def test_visualflow_memory_note_maps_keep_in_context_and_location() -> None:
    flow = {
        "id": "vf_memory_note_mapping",
        "name": "vf_memory_note_mapping",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "data": {
                    "nodeType": "on_flow_start",
                    "outputs": [
                        {"id": "exec-out", "type": "execution"},
                        {"id": "content", "type": "string"},
                        {"id": "keep_in_context", "type": "boolean"},
                        {"id": "location", "type": "string"},
                    ],
                    "pinDefaults": {"content": "hello", "keep_in_context": True, "location": "flow:test"},
                },
            },
            {"id": "note", "type": "memory_note", "data": {"nodeType": "memory_note"}},
        ],
        "edges": [
            {"source": "start", "sourceHandle": "exec-out", "target": "note", "targetHandle": "exec-in"},
            {"source": "start", "sourceHandle": "content", "target": "note", "targetHandle": "content"},
            {"source": "start", "sourceHandle": "keep_in_context", "target": "note", "targetHandle": "keep_in_context"},
            {"source": "start", "sourceHandle": "location", "target": "note", "targetHandle": "location"},
        ],
    }

    spec = compile_visualflow(flow)
    assert "note" in spec.nodes

    run = RunState(
        run_id="run",
        workflow_id=str(spec.workflow_id),
        status=RunStatus.RUNNING,
        current_node="start",
        vars={"_temp": {}},
    )
    plan = spec.nodes["start"](run, None)
    assert plan.next_node == "note"
    run.current_node = "note"

    plan = spec.nodes["note"](run, None)
    assert plan.effect is not None
    assert plan.effect.type == EffectType.MEMORY_NOTE
    assert plan.effect.payload.get("keep_in_context") is True
    assert plan.effect.payload.get("location") == "flow:test"


def test_visualflow_memory_query_maps_tags_mode_usernames_and_locations() -> None:
    flow = {
        "id": "vf_memory_query_mapping",
        "name": "vf_memory_query_mapping",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "data": {
                    "nodeType": "on_flow_start",
                    "outputs": [
                        {"id": "exec-out", "type": "execution"},
                        {"id": "query", "type": "string"},
                        {"id": "limit", "type": "number"},
                        {"id": "tags_mode", "type": "string"},
                        {"id": "usernames", "type": "array"},
                        {"id": "locations", "type": "array"},
                    ],
                    "pinDefaults": {
                        "query": "who",
                        "limit": 3,
                        "tags_mode": "any",
                        "usernames": ["alice"],
                        "locations": ["flow:x"],
                    },
                },
            },
            {"id": "query", "type": "memory_query", "data": {"nodeType": "memory_query"}},
        ],
        "edges": [
            {"source": "start", "sourceHandle": "exec-out", "target": "query", "targetHandle": "exec-in"},
            {"source": "start", "sourceHandle": "query", "target": "query", "targetHandle": "query"},
            {"source": "start", "sourceHandle": "limit", "target": "query", "targetHandle": "limit"},
            {"source": "start", "sourceHandle": "tags_mode", "target": "query", "targetHandle": "tags_mode"},
            {"source": "start", "sourceHandle": "usernames", "target": "query", "targetHandle": "usernames"},
            {"source": "start", "sourceHandle": "locations", "target": "query", "targetHandle": "locations"},
        ],
    }

    spec = compile_visualflow(flow)
    assert "query" in spec.nodes

    run = RunState(
        run_id="run",
        workflow_id=str(spec.workflow_id),
        status=RunStatus.RUNNING,
        current_node="start",
        vars={"_temp": {}},
    )
    plan = spec.nodes["start"](run, None)
    assert plan.next_node == "query"
    run.current_node = "query"

    plan = spec.nodes["query"](run, None)
    assert plan.effect is not None
    assert plan.effect.type == EffectType.MEMORY_QUERY
    assert plan.effect.payload.get("tags_mode") == "any"
    assert plan.effect.payload.get("usernames") == ["alice"]
    assert plan.effect.payload.get("locations") == ["flow:x"]
