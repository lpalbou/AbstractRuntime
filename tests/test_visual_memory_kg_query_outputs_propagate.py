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
    """Deterministic tiny embedder for tests (no external services)."""

    def embed_texts(self, texts):
        out = []
        for t in texts:
            t2 = str(t or "").lower()
            out.append([1.0 if "android" in t2 else 0.0, 0.1, 0.2])
        return out


def test_visual_memory_kg_query_propagates_active_memory_outputs(tmp_path) -> None:
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
            "id": "test-kg-query-propagates-packing",
            "name": "test-kg-query-propagates-packing",
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
                                "subject": "Data",
                                "predicate": "dcterms:description",
                                "object": "Android officer",
                                "observed_at": "2026-01-01T00:00:00Z",
                                "attributes": {"evidence_quote": "android"},
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
                            {"id": "active_memory_text", "label": "active_memory_text", "type": "string"},
                            {"id": "estimated_tokens", "label": "estimated_tokens", "type": "number"},
                            {"id": "count", "label": "count", "type": "number"},
                        ],
                        "pinDefaults": {
                            "query_text": "android",
                            "scope": "global",
                            "owner_id": "global_memory",
                            "min_score": 0.0,
                            "limit": 10,
                            "max_input_tokens": 80,
                            "model": "unit-test-model",
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
                            {"id": "active_memory_text", "label": "active_memory_text", "type": "string"},
                            {"id": "estimated_tokens", "label": "estimated_tokens", "type": "number"},
                            {"id": "count", "label": "count", "type": "number"},
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
                {"id": "e5", "source": "node-4", "sourceHandle": "active_memory_text", "target": "node-5", "targetHandle": "active_memory_text", "animated": False},
                {"id": "e6", "source": "node-4", "sourceHandle": "estimated_tokens", "target": "node-5", "targetHandle": "estimated_tokens", "animated": False},
                {"id": "e7", "source": "node-4", "sourceHandle": "count", "target": "node-5", "targetHandle": "count", "animated": False},
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
    assert isinstance(result.get("active_memory_text"), str)
    assert result["active_memory_text"].startswith("## KG ACTIVE MEMORY")
    assert int(result.get("estimated_tokens") or 0) > 0
    assert int(result.get("count") or 0) == 1

