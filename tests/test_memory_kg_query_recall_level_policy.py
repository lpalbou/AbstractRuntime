from __future__ import annotations

from abstractmemory import InMemoryTripleStore
from abstractruntime.core.models import Effect, EffectType, RunState
from abstractruntime.integrations.abstractmemory.effect_handlers import build_memory_kg_effect_handlers
from abstractruntime.storage.in_memory import InMemoryRunStore


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


def test_memory_kg_query_recall_level_applies_defaults_and_clamps() -> None:
    store = InMemoryTripleStore(embedder=_DummyEmbedder())
    run_store = InMemoryRunStore()
    run = RunState.new(workflow_id="wf", entry_node="start", session_id="sess-1", vars={"_temp": {}})
    run_store.save(run)

    handlers = build_memory_kg_effect_handlers(store=store, run_store=run_store, now_iso=lambda: "2026-01-03T00:00:00+00:00")
    assert_handler = handlers[EffectType.MEMORY_KG_ASSERT]
    query_handler = handlers[EffectType.MEMORY_KG_QUERY]

    out1 = assert_handler(
        run,
        Effect(
            type=EffectType.MEMORY_KG_ASSERT,
            payload={
                "scope": "global",
                "assertions": [
                    {"subject": "Data", "predicate": "dcterms:description", "object": "Android", "attributes": {"evidence_quote": "android"}},
                    {"subject": "Picard", "predicate": "dcterms:description", "object": "Bald", "attributes": {"evidence_quote": "bald"}},
                ],
            },
        ),
        None,
    )
    assert out1.status == "completed"

    # Defaults: urgent should apply strict min_score/limit and a small packing budget (so packets/active_memory_text exist).
    out2 = query_handler(
        run,
        Effect(type=EffectType.MEMORY_KG_QUERY, payload={"scope": "global", "query_text": "android", "recall_level": "urgent"}),
        None,
    )
    assert out2.status == "completed"
    assert isinstance(out2.result, dict)
    assert out2.result.get("ok") is True
    assert isinstance(out2.result.get("effort"), dict)
    assert out2.result["effort"].get("recall_level") == "urgent"
    assert isinstance(out2.result.get("items"), list) and out2.result["items"]
    assert isinstance(out2.result.get("active_memory_text"), str) and out2.result["active_memory_text"]
    assert isinstance(out2.result.get("packets"), list) and out2.result["packets"]

    # Overrides beyond the urgent envelope should be clamped and explained via warnings.
    out3 = query_handler(
        run,
        Effect(
            type=EffectType.MEMORY_KG_QUERY,
            payload={
                "scope": "global",
                "query_text": "android",
                "recall_level": "urgent",
                "limit": 9999,
                "min_score": 0.1,
                "max_input_tokens": 999999,
            },
        ),
        None,
    )
    assert out3.status == "completed"
    assert isinstance(out3.result, dict)
    assert out3.result.get("ok") is True
    warnings = out3.result.get("warnings")
    assert isinstance(warnings, list) and warnings
    joined = " | ".join([str(w) for w in warnings])
    assert "clamped limit" in joined
    assert "raised min_score" in joined
    assert "clamped max_input_tokens" in joined

