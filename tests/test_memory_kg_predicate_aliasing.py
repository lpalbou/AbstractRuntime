from __future__ import annotations

from abstractmemory import InMemoryTripleStore
from abstractruntime.core.models import Effect, EffectType, RunState
from abstractruntime.integrations.abstractmemory.effect_handlers import build_memory_kg_effect_handlers
from abstractruntime.storage.in_memory import InMemoryRunStore


def test_memory_kg_predicate_aliases_are_normalized_on_assert_and_query() -> None:
    """Level A: common schema.org-ish predicate aliases must not break ingestion."""

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
                "assertions": [
                    {
                        "subject": "Data",
                        "predicate": "schema:description",
                        "object": "Android",
                        "attributes": {"evidence_quote": "android"},
                    },
                    {
                        "subject": "Data",
                        "predicate": "schema:awareness",
                        "object": "Android",
                        "attributes": {"evidence_quote": "awareness"},
                    },
                    {
                        "subject": "Jean-Luc Picard",
                        "predicate": "schema:hasParent",
                        "object": "Starfleet",
                        "attributes": {"evidence_quote": "starfleet"},
                    },
                ],
            },
        ),
        None,
    )
    assert out1.status == "completed"
    assert isinstance(out1.result, dict)
    assert out1.result.get("ok") is True
    assert int(out1.result.get("count") or 0) == 3

    # Query using canonical predicate.
    out2 = query_handler(
        run,
        Effect(type=EffectType.MEMORY_KG_QUERY, payload={"scope": "global", "predicate": "dcterms:description", "limit": 10}),
        None,
    )
    assert out2.status == "completed"
    assert isinstance(out2.result, dict)
    items = out2.result.get("items")
    assert isinstance(items, list)
    assert items and items[0].get("predicate") == "dcterms:description"

    # Query using canonical predicate for awareness/knowledge.
    out_aw1 = query_handler(
        run,
        Effect(type=EffectType.MEMORY_KG_QUERY, payload={"scope": "global", "predicate": "schema:knowsAbout", "limit": 10}),
        None,
    )
    assert out_aw1.status == "completed"
    assert isinstance(out_aw1.result, dict)
    items_aw1 = out_aw1.result.get("items")
    assert isinstance(items_aw1, list)
    assert any(x.get("predicate") == "schema:knowsabout" for x in items_aw1)

    # Query using alias predicate (should be normalized).
    out3 = query_handler(
        run,
        Effect(type=EffectType.MEMORY_KG_QUERY, payload={"scope": "global", "predicate": "schema:description", "limit": 10}),
        None,
    )
    assert out3.status == "completed"
    assert isinstance(out3.result, dict)
    items2 = out3.result.get("items")
    assert isinstance(items2, list)
    assert items2 and items2[0].get("predicate") == "dcterms:description"

    # Query using awareness alias (should be normalized to schema:knowsabout).
    out_aw2 = query_handler(
        run,
        Effect(type=EffectType.MEMORY_KG_QUERY, payload={"scope": "global", "predicate": "schema:awareness", "limit": 10}),
        None,
    )
    assert out_aw2.status == "completed"
    assert isinstance(out_aw2.result, dict)
    items_aw2 = out_aw2.result.get("items")
    assert isinstance(items_aw2, list)
    assert any(x.get("predicate") == "schema:knowsabout" for x in items_aw2)
