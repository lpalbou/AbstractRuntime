from __future__ import annotations

from pathlib import Path

from abstractmemory import LanceDBTripleStore
from abstractruntime.core.runtime import Runtime
from abstractruntime.integrations.abstractmemory.effect_handlers import build_memory_kg_effect_handlers
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

from abstractflow import FlowRunner


class _DummyEmbedder:
    def embed_texts(self, texts):
        out = []
        for t in texts:
            t2 = str(t or "").lower()
            out.append([1.0 if "alice" in t2 else 0.0, 0.1, 0.2])
        return out


def test_visual_memact_compose_maps_kg_packets_into_memact_current_context(tmp_path) -> None:
    store_dir = Path(tmp_path) / "kg"
    store = LanceDBTripleStore(store_dir, embedder=_DummyEmbedder())

    run_store = InMemoryRunStore()
    runtime = Runtime(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        effect_handlers=build_memory_kg_effect_handlers(store=store, run_store=run_store, now_iso=lambda: "2026-01-01T00:00:02Z"),
    )

    vf = load_visualflow_json(
        {
            "id": "test-memact-compose",
            "name": "test-memact-compose",
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
                                "subject": "ex:person-alice",
                                "predicate": "schema:name",
                                "object": "Alice",
                                "observed_at": "2026-01-01T00:00:00Z",
                                "provenance": {"span_id": "s1"},
                            }
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
                            {"id": "owner_id", "label": "owner_id", "type": "string"},
                        ],
                        "outputs": [
                            {"id": "exec-out", "label": "", "type": "execution"},
                            {"id": "count", "label": "count", "type": "number"},
                            {"id": "ok", "label": "ok", "type": "boolean"},
                        ],
                        "pinDefaults": {"scope": "global", "owner_id": "global_memory"},
                    },
                },
                {
                    "id": "node-4",
                    "type": "memory_kg_query",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "nodeType": "memory_kg_query",
                        "label": "Query",
                        "inputs": [
                            {"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "query_text", "label": "query_text", "type": "string"},
                            {"id": "scope", "label": "scope", "type": "string"},
                            {"id": "owner_id", "label": "owner_id", "type": "string"},
                            {"id": "min_score", "label": "min_score", "type": "number"},
                            {"id": "limit", "label": "limit", "type": "number"},
                            {"id": "max_input_tokens", "label": "max_input_tokens", "type": "number"},
                            {"id": "model", "label": "model", "type": "model"},
                        ],
                        "outputs": [
                            {"id": "exec-out", "label": "", "type": "execution"},
                            {"id": "raw", "label": "raw", "type": "object"},
                        ],
                        "pinDefaults": {
                            "query_text": "alice",
                            "scope": "global",
                            "owner_id": "global_memory",
                            "min_score": 0.0,
                            "limit": 10,
                            "max_input_tokens": 120,
                            "model": "unit-test-model",
                        },
                    },
                },
                {
                    "id": "node-5",
                    "type": "memact_compose",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "nodeType": "memact_compose",
                        "label": "Compose",
                        "inputs": [
                            {"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "kg_result", "label": "kg_result", "type": "object"},
                            {"id": "stimulus", "label": "stimulus", "type": "string"},
                            {"id": "marker", "label": "marker", "type": "string"},
                            {"id": "max_items", "label": "max_items", "type": "number"},
                        ],
                        "outputs": [
                            {"id": "exec-out", "label": "", "type": "execution"},
                            {"id": "ok", "label": "ok", "type": "boolean"},
                            {"id": "memact_system_prompt", "label": "memact_system_prompt", "type": "string"},
                        ],
                        "pinDefaults": {"marker": "KG:", "stimulus": "who is alice?", "max_items": 5},
                    },
                },
                {
                    "id": "node-6",
                    "type": "on_flow_end",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "nodeType": "on_flow_end",
                        "label": "End",
                        "inputs": [
                            {"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "ok", "label": "ok", "type": "boolean"},
                            {"id": "memact_system_prompt", "label": "memact_system_prompt", "type": "string"},
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
                {"id": "e5", "source": "node-4", "sourceHandle": "raw", "target": "node-5", "targetHandle": "kg_result", "animated": False},
                {"id": "e6", "source": "node-5", "sourceHandle": "exec-out", "target": "node-6", "targetHandle": "exec-in", "animated": True},
                {"id": "e7", "source": "node-5", "sourceHandle": "ok", "target": "node-6", "targetHandle": "ok", "animated": False},
                {"id": "e8", "source": "node-5", "sourceHandle": "memact_system_prompt", "target": "node-6", "targetHandle": "memact_system_prompt", "animated": False},
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
    assert bool(result.get("ok")) is True
    sys_prompt = str(result.get("memact_system_prompt") or "")
    assert "## CURRENT CONTEXT" in sys_prompt
    # Keep this assertion resilient to unicode formatting differences (dash/arrow) and provenance suffixes.
    assert "KG: ex:person-alice" in sys_prompt
    assert "schema:name" in sys_prompt
