from __future__ import annotations

from abstractmemory import InMemoryTripleStore
from abstractruntime.core.runtime import Runtime
from abstractruntime.integrations.abstractmemory.effect_handlers import build_memory_kg_effect_handlers
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

from abstractflow import FlowRunner


def test_visual_memory_kg_resolve_propagates_candidates() -> None:
    store = InMemoryTripleStore()

    run_store = InMemoryRunStore()
    runtime = Runtime(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        effect_handlers=build_memory_kg_effect_handlers(store=store, run_store=run_store, now_iso=lambda: "2026-01-01T00:00:02Z"),
    )

    vf = load_visualflow_json(
        {
            "id": "test-kg-resolve-propagates-candidates",
            "name": "test-kg-resolve-propagates-candidates",
            "description": "",
            "interfaces": [],
            "nodes": [
                {
                    "id": "node-1",
                    "type": "on_flow_start",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "nodeType": "on_flow_start",
                        "label": "Start",
                        "inputs": [],
                        "outputs": [{"id": "exec-out", "label": "", "type": "execution"}],
                    },
                },
                {
                    "id": "node-2",
                    "type": "literal_array",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "nodeType": "literal_array",
                        "label": "Assertions",
                        "inputs": [],
                        "outputs": [{"id": "value", "label": "assertions", "type": "assertions"}],
                        "literalValue": [
                            {
                                "subject": "ex:person-noonien-soong",
                                "predicate": "schema:name",
                                "object": "doctor noonien soong",
                                "observed_at": "2026-01-01T00:00:00Z",
                                "provenance": {"span_id": "s1"},
                            },
                            {
                                "subject": "ex:person-noonien-soong",
                                "predicate": "rdf:type",
                                "object": "schema:Person",
                                "observed_at": "2026-01-01T00:00:00Z",
                                "provenance": {"span_id": "s1"},
                            },
                        ],
                    },
                },
                {
                    "id": "node-3",
                    "type": "memory_kg_assert",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "nodeType": "memory_kg_assert",
                        "label": "Assert",
                        "inputs": [
                            {"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "assertions", "label": "assertions", "type": "assertions"},
                            {"id": "scope", "label": "scope", "type": "string"},
                        ],
                        "outputs": [
                            {"id": "exec-out", "label": "", "type": "execution"},
                            {"id": "count", "label": "count", "type": "number"},
                            {"id": "ok", "label": "ok", "type": "boolean"},
                        ],
                        "pinDefaults": {"scope": "global"},
                    },
                },
                {
                    "id": "node-4",
                    "type": "memory_kg_resolve",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "nodeType": "memory_kg_resolve",
                        "label": "Resolve",
                        "inputs": [
                            {"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "label", "label": "label", "type": "string"},
                            {"id": "expected_type", "label": "expected_type", "type": "string"},
                            {"id": "scope", "label": "scope", "type": "string"},
                            {"id": "recall_level", "label": "recall_level", "type": "string"},
                            {"id": "max_candidates", "label": "max_candidates", "type": "number"},
                        ],
                        "outputs": [
                            {"id": "exec-out", "label": "", "type": "execution"},
                            {"id": "ok", "label": "ok", "type": "boolean"},
                            {"id": "count", "label": "count", "type": "number"},
                            {"id": "candidates", "label": "candidates", "type": "array"},
                        ],
                        "pinDefaults": {
                            "label": "doctor noonien soong",
                            "expected_type": "schema:person",
                            "scope": "global",
                            "recall_level": "urgent",
                            "max_candidates": 5,
                        },
                    },
                },
                {
                    "id": "node-5",
                    "type": "on_flow_end",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "nodeType": "on_flow_end",
                        "label": "End",
                        "inputs": [
                            {"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "ok", "label": "ok", "type": "boolean"},
                            {"id": "count", "label": "count", "type": "number"},
                            {"id": "candidates", "label": "candidates", "type": "array"},
                        ],
                        "outputs": [],
                    },
                },
            ],
            "edges": [
                {"id": "e1", "source": "node-1", "sourceHandle": "exec-out", "target": "node-3", "targetHandle": "exec-in", "animated": True},
                {"id": "e2", "source": "node-2", "sourceHandle": "value", "target": "node-3", "targetHandle": "assertions", "animated": False},
                {"id": "e3", "source": "node-3", "sourceHandle": "exec-out", "target": "node-4", "targetHandle": "exec-in", "animated": True},
                {"id": "e4", "source": "node-4", "sourceHandle": "exec-out", "target": "node-5", "targetHandle": "exec-in", "animated": True},
                {"id": "e5", "source": "node-4", "sourceHandle": "ok", "target": "node-5", "targetHandle": "ok", "animated": False},
                {"id": "e6", "source": "node-4", "sourceHandle": "count", "target": "node-5", "targetHandle": "count", "animated": False},
                {"id": "e7", "source": "node-4", "sourceHandle": "candidates", "target": "node-5", "targetHandle": "candidates", "animated": False},
            ],
            "entryNode": "node-1",
        }
    )

    flow = visual_to_flow(vf)
    runner = FlowRunner(flow, runtime=runtime)
    out = runner.run({})

    assert out.get("success") is True
    result = out.get("result")
    assert isinstance(result, dict)
    assert result.get("ok") is True
    assert int(result.get("count") or 0) == 1
    candidates = result.get("candidates")
    assert isinstance(candidates, list)
    assert candidates and isinstance(candidates[0], dict)
    assert candidates[0].get("id") == "ex:person-noonien-soong"

