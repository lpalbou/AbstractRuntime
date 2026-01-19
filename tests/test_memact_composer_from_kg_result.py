from __future__ import annotations

from abstractruntime.memory.active_memory import apply_memact_envelope, get_memact_memory
from abstractruntime.memory.memact_composer import compose_memact_current_context_from_kg_result


def test_memact_composer_replaces_previous_kg_context_entries() -> None:
    vars: dict = {}

    _ = apply_memact_envelope(
        vars,
        envelope={
            "current_context": {"added": ["KG: old —schema:name→ stale"], "removed": []},
        },
        now_iso=lambda: "2026-01-19T12:00:00+00:00",
        now_compact=lambda: "26/01/19 12:00:00",
    )

    kg_result = {
        "ok": True,
        "packets": [
            {
                "statement": "ex:person-alice —schema:name→ Alice",
                "span_id": "s_demo",
                "observed_at": "2026-01-01T00:00:00Z",
            }
        ],
        "count": 1,
        "packed_count": 1,
        "estimated_tokens": 10,
        "dropped": 0,
    }

    out = compose_memact_current_context_from_kg_result(vars, kg_result=kg_result, stimulus="who is alice?", marker="KG:")
    assert out.get("ok") is True

    mem = get_memact_memory(vars)
    ctx = mem.get("current_context") or []
    assert isinstance(ctx, list) and len(ctx) == 1
    assert "old —schema:name→ stale" not in str(ctx[0].get("text") or "")
    assert str(ctx[0].get("text") or "").startswith("KG:")
    assert "ex:person-alice —schema:name→ Alice" in str(ctx[0].get("text") or "")
    assert "span:s_demo" in str(ctx[0].get("text") or "")


def test_memact_composer_falls_back_to_packetize_items_when_packets_missing() -> None:
    vars: dict = {}
    kg_result = {
        "ok": True,
        "items": [
            {
                "subject": "ex:person-bob",
                "predicate": "schema:name",
                "object": "Bob",
                "observed_at": "2026-01-01T00:00:00Z",
                "provenance": {"span_id": "s1"},
            },
            {
                "subject": "ex:person-bob",
                "predicate": "rdf:type",
                "object": "schema:Person",
                "observed_at": "2026-01-01T00:00:01Z",
                "provenance": {"span_id": "s2"},
            },
        ],
        "count": 2,
    }

    out = compose_memact_current_context_from_kg_result(vars, kg_result=kg_result, stimulus="bob", max_items=1)
    assert out.get("ok") is True

    mem = get_memact_memory(vars)
    ctx = mem.get("current_context") or []
    assert isinstance(ctx, list) and len(ctx) == 1
    assert "ex:person-bob —schema:name→ Bob" in str(ctx[0].get("text") or "")

