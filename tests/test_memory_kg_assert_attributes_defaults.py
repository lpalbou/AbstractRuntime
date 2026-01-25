from __future__ import annotations

from abstractmemory import InMemoryTripleStore
from abstractruntime.core.models import Effect, EffectType, RunState
from abstractruntime.integrations.abstractmemory.effect_handlers import build_memory_kg_effect_handlers
from abstractruntime.storage.in_memory import InMemoryRunStore


def test_memory_kg_assert_applies_attributes_defaults_without_overriding() -> None:
    store = InMemoryTripleStore(embedder=None)
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
                "attributes_defaults": {
                    "persona_id": "default",
                    "extractor_policy_version": "v1",
                    "evidence_quote": "should_not_override",
                },
                "assertions": [
                    {
                        "subject": "Data",
                        "predicate": "dcterms:description",
                        "object": "Android",
                        "attributes": {"evidence_quote": "android", "persona_id": "explicit"},
                    },
                    {"subject": "Picard", "predicate": "dcterms:description", "object": "Bald"},
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
        Effect(type=EffectType.MEMORY_KG_QUERY, payload={"scope": "global", "predicate": "dcterms:description", "limit": 10}),
        None,
    )
    assert out2.status == "completed"
    assert isinstance(out2.result, dict)
    items = out2.result.get("items")
    assert isinstance(items, list)

    def _norm(v) -> str:
        return str(v or "").strip().lower()

    item_data = next(x for x in items if _norm(x.get("object")) == "android")
    item_picard = next(x for x in items if _norm(x.get("object")) == "bald")

    attrs_data = item_data.get("attributes")
    assert isinstance(attrs_data, dict)
    assert attrs_data.get("persona_id") == "explicit"
    assert attrs_data.get("extractor_policy_version") == "v1"
    assert attrs_data.get("evidence_quote") == "android"

    attrs_picard = item_picard.get("attributes")
    assert isinstance(attrs_picard, dict)
    assert attrs_picard.get("persona_id") == "default"
    assert attrs_picard.get("extractor_policy_version") == "v1"
    assert attrs_picard.get("evidence_quote") == "should_not_override"

