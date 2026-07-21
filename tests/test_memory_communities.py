"""Memory communities (identity/communities.py) — the topic-view compute.

Pins: determinism (same store -> same partition), structural dominance
(engraved relations beat keyword hints), bridge caps (a ubiquitous keyword
cannot fuse the life), label mechanics (distinctive keywords, exemplar by
internal degree), and the JSON contract the gateway route serves.
"""

from __future__ import annotations

from abstractruntime.identity.communities import (
    CommunityGraph,
    MemoryNode,
    _add_keyword_bridges,
    louvain_partition,
    memory_communities,
)


class _FakeRow:
    def __init__(self, subject: str, predicate: str, obj: str, attributes=None):
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.attributes = attributes or {}
        self.assertion_id = f"{subject}|{predicate}|{obj}"


class _FakeStore:
    """Serves TripleQuery(predicate=...) like the home store."""

    def __init__(self, rows):
        self._rows = rows

    def query(self, q):
        return [r for r in self._rows if q.predicate is None or r.predicate == q.predicate]


def _digest(gid: str, kind: str, title: str, keywords=()):
    return _FakeRow(
        gid,
        "dcterms:abstract",
        f"digest of {gid}",
        {"record_kind": kind, "title": title, "keywords": list(keywords)},
    )


def _two_cliques_store():
    """Two structural cliques joined by one weak keyword pair."""
    rows = []
    for i in range(4):
        rows.append(_digest(f"ex:a{i}", "episode", f"alpha topic {i}", ["dam", "verification", f"a{i}"]))
        rows.append(_digest(f"ex:b{i}", "episode", f"beta topic {i}", ["voyager", "coherence", f"b{i}"]))
    for i in range(3):
        rows.append(_FakeRow(f"ex:a{i}", "continues", f"ex:a{i+1}"))
        rows.append(_FakeRow(f"ex:b{i}", "continues", f"ex:b{i+1}"))
    rows.append(_FakeRow("ex:a0", "mentions", "ex:a2"))
    rows.append(_FakeRow("ex:b0", "mentions", "ex:b2"))
    return _FakeStore(rows)


def test_two_cliques_partition_into_two_communities() -> None:
    out = memory_communities(_two_cliques_store())
    assert out["node_count"] == 8
    assert len(out["communities"]) == 2
    sizes = sorted(c["size"] for c in out["communities"])
    assert sizes == [4, 4]
    members0 = set(out["communities"][0]["member_ids"])
    assert members0 == {f"ex:a{i}" for i in range(4)} or members0 == {f"ex:b{i}" for i in range(4)}


def test_deterministic_across_calls() -> None:
    a = memory_communities(_two_cliques_store())
    b = memory_communities(_two_cliques_store())
    assert a == b


def test_labels_are_distinctive_keywords() -> None:
    out = memory_communities(_two_cliques_store())
    labels = {c["label"] for c in out["communities"]}
    joined = " ".join(labels)
    assert "dam" in joined or "verification" in joined
    assert "voyager" in joined or "coherence" in joined


def test_ubiquitous_keyword_cannot_fuse_the_life() -> None:
    """Every record shares 'memory'+'thinking' — without the df cut and
    degree cap this would be one blob; the structural cliques must still
    separate."""
    rows = []
    for i in range(4):
        rows.append(_digest(f"ex:a{i}", "episode", f"alpha {i}", ["memory", "thinking", "dam"]))
        rows.append(_digest(f"ex:b{i}", "episode", f"beta {i}", ["memory", "thinking", "voyager"]))
    for i in range(3):
        rows.append(_FakeRow(f"ex:a{i}", "continues", f"ex:a{i+1}"))
        rows.append(_FakeRow(f"ex:b{i}", "continues", f"ex:b{i+1}"))
    out = memory_communities(_FakeStore(rows))
    assert len(out["communities"]) == 2


def test_bridge_degree_cap_and_no_double_edges() -> None:
    g = CommunityGraph()
    for i in range(10):
        g.nodes[f"ex:n{i}"] = MemoryNode(gid=f"ex:n{i}", kind="episode", title=f"n{i}", keywords=("kw1", "kw2"))
    g.add_edge("ex:n0", "ex:n1", 1.0)  # existing structural edge
    _add_keyword_bridges(g)
    # No bridge doubled the structural edge weight beyond one addition.
    assert g.adj["ex:n0"]["ex:n1"] == 1.0
    for gid in g.nodes:
        bridge_count = sum(1 for w in g.adj.get(gid, {}).values() if w == 0.35)
        assert bridge_count <= 4, f"{gid} accepted {bridge_count} bridges"


def test_singletons_fold_into_unclustered() -> None:
    rows = [
        _digest("ex:a0", "episode", "a0", ["x1", "y1"]),
        _digest("ex:a1", "episode", "a1", ["x2", "y2"]),
        _FakeRow("ex:a0", "continues", "ex:a1"),
        _digest("ex:lone", "interest", "a lone interest", ["z9", "z8"]),
    ]
    out = memory_communities(_FakeStore(rows))
    assert out["unclustered"] == ["ex:lone"]
    assert all("ex:lone" not in c["member_ids"] for c in out["communities"])


def test_empty_store_serves_empty_contract() -> None:
    out = memory_communities(_FakeStore([]))
    assert out["communities"] == [] and out["unclustered"] == [] and out["node_count"] == 0


def test_edges_to_unknown_nodes_are_ignored() -> None:
    rows = [
        _digest("ex:a0", "episode", "a0", []),
        _FakeRow("ex:a0", "mentions", "ex:ghost"),  # target has no digest
        _FakeRow("ex:ghost2", "mentions", "ex:a0"),
    ]
    out = memory_communities(_FakeStore(rows))
    assert out["node_count"] == 1
    assert out["edge_count"] == 0


def test_partition_covers_every_node_exactly_once() -> None:
    out = memory_communities(_two_cliques_store())
    seen = list(out["unclustered"])
    for c in out["communities"]:
        seen.extend(c["member_ids"])
    assert sorted(seen) == sorted({s for s in seen})
    assert len(seen) == out["node_count"]


def test_louvain_empty_graph() -> None:
    assert louvain_partition(CommunityGraph()) == {}


def test_ring_of_cliques_multipass_aggregation() -> None:
    """Adversary F1 (2026-07-17): dropping intra-community self-loops in
    the aggregation made every pass after the first maximize modularity of
    the WRONG graph — a 10x K5 ring collapsed to ONE community (Q=0.0).
    Correct aggregation keeps self-loops; the ring resolves its 10 planted
    cliques at resolution 1."""
    g = CommunityGraph()
    K, N = 5, 10
    for c in range(N):
        for i in range(K):
            gid = f"ex:c{c}n{i}"
            g.nodes[gid] = MemoryNode(gid=gid, kind="episode", title=gid, keywords=())
    for c in range(N):
        ids = [f"ex:c{c}n{i}" for i in range(K)]
        for i in range(K):
            for j in range(i + 1, K):
                g.add_edge(ids[i], ids[j], 1.0)
        # ring link to the next clique
        g.add_edge(ids[0], f"ex:c{(c + 1) % N}n0", 1.0)
    part = louvain_partition(g, resolution=1.0)
    n_communities = len(set(part.values()))
    assert n_communities == N, f"expected {N} planted cliques, got {n_communities}"


def test_duplicate_relation_rows_do_not_inflate_weight() -> None:
    """Adversary F4: the same (s,p,o) asserted twice must weigh once."""
    rows = [
        _digest("ex:a", "episode", "a", []),
        _digest("ex:b", "episode", "b", []),
        _FakeRow("ex:a", "mentions", "ex:b"),
        _FakeRow("ex:a", "mentions", "ex:b"),  # duplicate assertion
    ]
    from abstractruntime.identity.communities import collect_graph

    g = collect_graph(_FakeStore(rows))
    assert g.adj["ex:a"]["ex:b"] == 1.0


def test_non_string_keywords_never_become_labels() -> None:
    """Adversary F6: None/int keyword elements must not materialize."""
    rows = [
        _digest("ex:a", "episode", "a", ["mémoire", None, 42, "éphémère"]),
        _digest("ex:b", "episode", "b", ["mémoire", "éphémère", "x"]),
        _FakeRow("ex:a", "continues", "ex:b"),
    ]
    out = memory_communities(_FakeStore(rows))
    joined = " ".join(c["label"] for c in out["communities"])
    assert "none" not in joined and "42" not in joined
