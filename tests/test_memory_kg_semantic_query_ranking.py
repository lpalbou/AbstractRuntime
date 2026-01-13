from __future__ import annotations

from abstractmemory import InMemoryTripleStore
from abstractruntime.core.models import Effect, EffectType, RunState
from abstractruntime.integrations.abstractmemory.effect_handlers import build_memory_kg_effect_handlers
from abstractruntime.storage.in_memory import InMemoryRunStore


def test_memory_kg_query_text_preserves_similarity_ranking_and_supports_min_score() -> None:
    """Level A: semantic queries must be similarity-ranked (not observed_at ranked)."""

    class _DummyEmbedder:
        def embed_texts(self, texts):
            out = []
            for t in texts:
                t2 = str(t or "").lower()
                if "android" in t2:
                    out.append([1.0, 0.0])
                elif "bald" in t2:
                    out.append([0.0, 1.0])
                else:
                    out.append([0.0, 0.0])
            return out

    store = InMemoryTripleStore(embedder=_DummyEmbedder())
    run_store = InMemoryRunStore()
    run = RunState.new(workflow_id="wf", entry_node="start", session_id="sess-1", vars={"_temp": {}})
    run_store.save(run)

    handlers = build_memory_kg_effect_handlers(store=store, run_store=run_store, now_iso=lambda: "2026-01-03T00:00:00+00:00")
    assert_handler = handlers[EffectType.MEMORY_KG_ASSERT]
    query_handler = handlers[EffectType.MEMORY_KG_QUERY]

    # Insert two assertions:
    # - picard is newer but irrelevant to "android"
    # - data is older but semantically relevant to "android"
    out1 = assert_handler(
        run,
        Effect(
            type=EffectType.MEMORY_KG_ASSERT,
            payload={
                "scope": "global",
                "assertions": [
                    {
                        "subject": "Picard",
                        "predicate": "has_attribute",
                        "object": "Bald",
                        "observed_at": "2026-01-02T00:00:00+00:00",
                        "attributes": {"evidence_quote": "bald"},
                    },
                    {
                        "subject": "Data",
                        "predicate": "is_a",
                        "object": "Android",
                        "observed_at": "2026-01-01T00:00:00+00:00",
                        "attributes": {"evidence_quote": "android"},
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

    out2 = query_handler(
        run,
        Effect(type=EffectType.MEMORY_KG_QUERY, payload={"scope": "global", "query_text": "android", "limit": 10}),
        None,
    )
    assert out2.status == "completed"
    assert isinstance(out2.result, dict)
    assert out2.result.get("ok") is True
    items = out2.result.get("items")
    assert isinstance(items, list)
    assert items and items[0].get("subject") == "data"
    attrs0 = items[0].get("attributes")
    assert isinstance(attrs0, dict)
    retrieval0 = attrs0.get("_retrieval")
    assert isinstance(retrieval0, dict)
    assert isinstance(retrieval0.get("score"), (int, float))

    # With an explicit min_score threshold, irrelevant assertions should be suppressed.
    out3 = query_handler(
        run,
        Effect(type=EffectType.MEMORY_KG_QUERY, payload={"scope": "global", "query_text": "android", "min_score": 0.5, "limit": 10}),
        None,
    )
    assert out3.status == "completed"
    assert isinstance(out3.result, dict)
    items2 = out3.result.get("items")
    assert isinstance(items2, list)
    assert len(items2) == 1
    assert items2[0].get("subject") == "data"

