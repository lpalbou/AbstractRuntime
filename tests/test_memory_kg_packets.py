from __future__ import annotations

from abstractruntime.memory.kg_packets import pack_active_memory_text, packetize_assertions


def test_packetize_assertions_extracts_provenance_and_scores_and_dedups() -> None:
    items = [
        {
            "subject": "data",
            "predicate": "dcterms:description",
            "object": "android officer",
            "scope": "global",
            "owner_id": "global_memory",
            "observed_at": "2026-01-01T00:00:00Z",
            "provenance": {"span_id": "span-1", "writer_run_id": "r1", "writer_workflow_id": "wf1"},
            "attributes": {"_retrieval": {"score": 0.75}},
        },
        # Duplicate (case drift); should be deduped.
        {
            "subject": "Data",
            "predicate": "DCTERMS:DESCRIPTION",
            "object": "Android officer",
            "scope": "global",
            "owner_id": "global_memory",
            "observed_at": "2026-01-01T00:00:00Z",
            "provenance": {"span_id": "span-1"},
        },
    ]

    packets = packetize_assertions(items)
    assert len(packets) == 1
    pkt = packets[0]
    assert pkt.get("version") == 0
    assert pkt.get("subject") == "data"
    assert pkt.get("predicate") == "dcterms:description"
    assert pkt.get("object") == "android officer"
    assert "data —dcterms:description→ android officer" == pkt.get("statement")
    assert pkt.get("observed_at") == "2026-01-01T00:00:00Z"
    assert pkt.get("scope") == "global"
    assert pkt.get("owner_id") == "global_memory"
    assert pkt.get("span_id") == "span-1"
    assert pkt.get("writer_run_id") == "r1"
    assert pkt.get("writer_workflow_id") == "wf1"
    assert pkt.get("retrieval_score") == 0.75


def test_pack_active_memory_text_respects_budget(monkeypatch) -> None:
    # Deterministic token estimate for the test (avoid model-dependent TokenUtils heuristics).
    monkeypatch.setattr(
        "abstractruntime.memory.kg_packets.estimate_tokens",
        lambda text, model=None: len(str(text or "").split()),
    )

    packets = [
        {"statement": "data —dcterms:description→ android officer", "observed_at": "t1", "retrieval_score": 0.7},
        {"statement": "picard —dcterms:description→ bald captain", "observed_at": "t2", "retrieval_score": 0.6},
        {"statement": "x —dcterms:description→ y", "observed_at": "t3", "retrieval_score": 0.5},
    ]

    # Small budget: should only include a subset of packets.
    text, kept, estimated, dropped = pack_active_memory_text(
        packets,
        scope="session",
        max_input_tokens=24,
        include_scores=True,
    )

    assert isinstance(text, str)
    assert text.startswith("## KG ACTIVE MEMORY")
    assert estimated > 0
    assert 1 <= len(kept) < len(packets)
    assert dropped == len(packets) - len(kept)
    assert "score:" in text
