from __future__ import annotations

from pathlib import Path

from abstractmemory import LanceDBTripleStore
from abstractruntime.core.models import Effect, EffectType, RunState
from abstractruntime.integrations.abstractmemory.effect_handlers import build_memory_kg_effect_handlers
from abstractruntime.storage.in_memory import InMemoryRunStore


class _DummyEmbedder:
    """Deterministic tiny embedder for Level B tests (no external services)."""

    def embed_texts(self, texts):
        out = []
        for t in texts:
            t2 = str(t or "").lower()
            out.append(
                [
                    1.0 if "android" in t2 else 0.0,
                    1.0 if "bald" in t2 else 0.0,
                    0.1,
                ]
            )
        return out


def test_memory_kg_query_packetization_survives_restart(tmp_path, monkeypatch) -> None:
    # Make packet packing deterministic for the test (avoid model-specific TokenUtils estimates).
    monkeypatch.setattr(
        "abstractruntime.memory.kg_packets.estimate_tokens",
        lambda text, model=None: len(str(text or "").split()),
    )

    store_dir = Path(tmp_path) / "kg"
    store = LanceDBTripleStore(store_dir, embedder=_DummyEmbedder())

    run_store = InMemoryRunStore()
    run = RunState.new(workflow_id="wf", entry_node="start", session_id="sess-1", vars={"_temp": {}})
    run_store.save(run)

    handlers = build_memory_kg_effect_handlers(store=store, run_store=run_store, now_iso=lambda: "2026-01-01T00:00:02Z")
    assert_handler = handlers[EffectType.MEMORY_KG_ASSERT]
    query_handler = handlers[EffectType.MEMORY_KG_QUERY]

    out1 = assert_handler(
        run,
        Effect(
            type=EffectType.MEMORY_KG_ASSERT,
            payload={
                "scope": "global",
                "owner_id": "global_memory",
                "assertions": [
                    {
                        "subject": "Data",
                        "predicate": "dcterms:description",
                        "object": "Android officer",
                        "observed_at": "2026-01-01T00:00:00Z",
                        "attributes": {"evidence_quote": "android"},
                        "provenance": {"span_id": "s1"},
                    },
                    {
                        "subject": "Picard",
                        "predicate": "dcterms:description",
                        "object": "Bald captain",
                        "observed_at": "2026-01-01T00:00:01Z",
                        "attributes": {"evidence_quote": "bald"},
                        "provenance": {"span_id": "s2"},
                    },
                ],
            },
        ),
        None,
    )
    assert out1.status == "completed"
    assert isinstance(out1.result, dict)
    assert out1.result.get("ok") is True
    assert int(out1.result.get("count") or 0) == 2

    query_effect = Effect(
        type=EffectType.MEMORY_KG_QUERY,
        payload={
            "scope": "global",
            "owner_id": "global_memory",
            "query_text": "android",
            "min_score": 0.0,
            "limit": 10,
            "max_input_tokens": 80,
            "model": "unit-test-model",
        },
    )

    out2 = query_handler(run, query_effect, None)
    assert out2.status == "completed"
    assert isinstance(out2.result, dict)
    assert out2.result.get("ok") is True
    assert isinstance(out2.result.get("active_memory_text"), str)
    assert out2.result["active_memory_text"].startswith("## KG ACTIVE MEMORY")
    assert "score:" in out2.result["active_memory_text"]
    packets = out2.result.get("packets")
    assert isinstance(packets, list) and packets

    snapshot = out2.result["active_memory_text"]

    # Restart simulation: new store instance pointing at the same dir should see the same data.
    store2 = LanceDBTripleStore(store_dir, embedder=_DummyEmbedder())
    handlers2 = build_memory_kg_effect_handlers(store=store2, run_store=run_store, now_iso=lambda: "2026-01-01T00:00:02Z")
    query_handler2 = handlers2[EffectType.MEMORY_KG_QUERY]

    out3 = query_handler2(run, query_effect, None)
    assert out3.status == "completed"
    assert isinstance(out3.result, dict)
    assert out3.result.get("ok") is True
    assert out3.result.get("active_memory_text") == snapshot
