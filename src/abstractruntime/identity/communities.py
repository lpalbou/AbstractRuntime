"""Memory communities — topic regrouping over a home's graph (pure read).

The operator's ask (2026-07-17): "apply a community detection algorithm
like louvain … to regroup memories on a same topic and provide a much
nicer visualization", computed at the RUNTIME level so it is durable and
serves every consumer (the entity app's Topics view is the first).

Design (posted for runtime owner-review, commons c2732):
- INPUT is the store only — the structures the formation lane already
  writes: digest assertions carry kind/title/keywords; structural relation
  assertions (summarizes/mentions/written_amid/continues/derived_from/
  from_session/reflected_in/refines) are the graph's engraved topology.
  No journal walk, no embeddings, no LLM call, no new dependencies.
- Keyword BRIDGE edges (weak, capped) connect records that share
  keywords: young lives have sparse structural topology, and topicality
  is exactly what shared digest keywords witness. Weights stay below
  structural edges so engraved relations dominate the partition.
- LOUVAIN, deterministic: sorted iteration order, no RNG — the same life
  always yields the same partition (operators scrub back and forth; a
  view that reshuffles on every load is not a view of anything).
- LABELS are mechanical: the community's most DISTINCTIVE member keywords
  (document-frequency-penalized across communities) plus an exemplar
  title (highest internal weighted degree). Honest, cheap, no model call
  — an LLM naming pass can layer on later as a separate, labeled act.

Privacy: labels and exemplars are built from KEYWORDS and TITLES only —
diary projections carry no keywords and act-frame titles ("Diary entry
(note) — <date>"), and PRIVATE entries' digests are word-free by the
formation rule. Digest TEXT is never served by this module; a future
change that surfaced digests would expose non-private gists and must
re-review this boundary (adversary F8).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

# The engraved relation vocabulary (memory's one-spelling law — these are
# the plain-word predicates shipped lives carry; widening follows the
# registry coordination rule, never ad-hoc).
STRUCTURAL_PREDICATES: Tuple[str, ...] = (
    "summarizes",
    "mentions",
    "written_amid",
    "continues",
    "derived_from",
    "from_session",
    "reflected_in",
    "refines",
)

STRUCTURAL_WEIGHT = 1.0
# Keyword bridges are HINTS, not engravings: weak enough that structure
# wins wherever both exist.
KEYWORD_BRIDGE_WEIGHT = 0.35
# Two shared keywords required — one shared token is noise at digest
# keyword sizes (8 per record).
KEYWORD_BRIDGE_MIN_SHARED = 2
# Degree cap per node for bridge edges: the strongest partners only, so
# a ubiquitous keyword cannot fuse the whole life into one blob.
KEYWORD_BRIDGE_MAX_PER_NODE = 4

# Communities smaller than this fold into the "unclustered" bucket the
# view renders separately (a topic of one is not a topic).
MIN_COMMUNITY_SIZE = 2


@dataclass
class MemoryNode:
    """One formed record as the community graph sees it."""

    gid: str
    kind: str
    title: str
    keywords: Tuple[str, ...] = ()


@dataclass
class CommunityGraph:
    nodes: Dict[str, MemoryNode] = field(default_factory=dict)
    # Undirected weighted adjacency: gid -> {gid: weight} (symmetric by
    # construction).
    adj: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def add_edge(self, a: str, b: str, w: float) -> None:
        if a == b or a not in self.nodes or b not in self.nodes:
            return
        self.adj.setdefault(a, {})
        self.adj.setdefault(b, {})
        self.adj[a][b] = self.adj[a].get(b, 0.0) + w
        self.adj[b][a] = self.adj[b].get(a, 0.0) + w


def collect_graph(store: Any) -> CommunityGraph:
    """Read the community graph out of a home's triple store (pure read).

    Uses the store's query surface only (TripleQuery), so it works on any
    backend the home runs. Limits are explicit and generous — a home is
    one life, not a warehouse (Ephemeral: 369 records after a dense week).
    """
    from abstractmemory.store import TripleQuery

    g = CommunityGraph()
    # Nodes: every digest assertion is one formed record.
    for row in store.query(TripleQuery(predicate="dcterms:abstract", limit=200_000)):
        gid = str(row.subject or "")
        if not gid:
            continue
        attrs: Mapping[str, Any] = row.attributes if isinstance(row.attributes, Mapping) else {}
        kws = attrs.get("keywords")
        # Strings only (adversary F6: None/int elements materialized as
        # label tokens "none"/"42").
        keywords = (
            tuple(str(k).lower() for k in kws if isinstance(k, str) and k.strip())
            if isinstance(kws, (list, tuple))
            else ()
        )
        g.nodes[gid] = MemoryNode(
            gid=gid,
            kind=str(attrs.get("record_kind") or "memory"),
            title=str(attrs.get("title") or ""),
            keywords=keywords,
        )
    # Structural edges: engraved relations between records. STAR
    # NORMALIZATION: a consolidation summary `summarizes` 60 episodes and
    # a diary entry is `written_amid` a dozen — at flat weight those hubs
    # fuse the whole life into one blob. Each relation's weight divides by
    # sqrt(fan-out of its subject under that predicate): chains keep
    # weight 1, stars SPREAD theirs (deliberately — a hub edge carries
    # less per-pair salience; at fan-out >= 9 it drops below the bridge
    # weight, which is the intent, not an invariant violation).
    # Duplicate (s,p,o) rows dedupe first (adversary F4: re-asserted
    # triples inflated weight superlinearly).
    rel_set: set = set()
    for predicate in STRUCTURAL_PREDICATES:
        for row in store.query(TripleQuery(predicate=predicate, limit=200_000)):
            s, o = str(row.subject or ""), str(row.object or "")
            if s and o:
                rel_set.add((s, predicate, o))
    fanout: Counter = Counter()
    for s, predicate, _o in rel_set:
        fanout[(s, predicate)] += 1
    for s, predicate, o in sorted(rel_set):
        g.add_edge(s, o, STRUCTURAL_WEIGHT / (fanout[(s, predicate)] ** 0.5))
    _add_keyword_bridges(g)
    return g


def _add_keyword_bridges(g: CommunityGraph) -> None:
    """Weak topic edges between records sharing >= MIN_SHARED keywords.

    Deterministic and degree-capped: candidate pairs rank by (shared
    count desc, pair asc); each node accepts at most MAX_PER_NODE
    bridges. An inverted index keeps this near-linear in practice
    (keyword lists are <= 8 per record by formation rule)."""
    by_keyword: Dict[str, List[str]] = defaultdict(list)
    for gid in sorted(g.nodes):
        for kw in set(g.nodes[gid].keywords):
            by_keyword[kw].append(gid)
    # A keyword carried by more than a QUARTER of the life ("memory" on a
    # memory-obsessed entity) is ambient vocabulary, not topical signal —
    # the df cut a tf-idf would apply, RELATIVE to life size (an absolute
    # cut never fires on young lives and ambient words fuse everything).
    df_cut = max(4, len(g.nodes) // 4)
    # Candidate-pair budget (adversary F2: one keyword at df=2500 in a
    # 10k life cost 13.8s / 746MB in C(df,2) pairs). Keywords iterate
    # most-SPECIFIC first (df asc — specific words are the topical
    # signal), and enumeration stops at the budget: the bridges lost are
    # the least-informative ones by construction.
    pair_budget = 60_000
    shared: Counter = Counter()
    for kw in sorted(by_keyword, key=lambda k: (len(by_keyword[k]), k)):
        gids = by_keyword[kw]
        if len(gids) < 2 or len(gids) > df_cut:
            continue
        if pair_budget <= 0:
            break
        for i, a in enumerate(gids):
            for b in gids[i + 1 :]:
                shared[(a, b)] += 1
        pair_budget -= (len(gids) * (len(gids) - 1)) // 2
    accepted: Dict[str, int] = defaultdict(int)
    ranked = sorted(shared.items(), key=lambda kv: (-kv[1], kv[0]))
    for (a, b), n in ranked:
        if n < KEYWORD_BRIDGE_MIN_SHARED:
            continue
        if accepted[a] >= KEYWORD_BRIDGE_MAX_PER_NODE or accepted[b] >= KEYWORD_BRIDGE_MAX_PER_NODE:
            continue
        # Never double an existing structural edge — bridges only ADD.
        if b in g.adj.get(a, {}):
            continue
        g.add_edge(a, b, KEYWORD_BRIDGE_WEIGHT)
        accepted[a] += 1
        accepted[b] += 1


# --------------------------------------------------------------- louvain


def louvain_partition(g: CommunityGraph, *, max_passes: int = 10, resolution: float = 1.0) -> Dict[str, int]:
    """Deterministic Louvain: modularity-greedy local moves + aggregation.

    Iteration order is sorted everywhere (no RNG): the same graph yields
    the same partition on every call. Standard two-phase loop; the
    `resolution` parameter (γ in the RB-modularity gain) controls
    granularity — a dense small-world life is ONE community at γ=1
    (measured live: 361/369), and the caller sweeps γ up until the
    partition is informative. Isolated nodes keep singleton communities
    (folded to the unclustered bucket by the caller)."""
    nodes: List[str] = sorted(g.nodes)
    adj: Dict[str, Dict[str, float]] = {n: dict(g.adj.get(n, {})) for n in nodes}
    member_of: Dict[str, str] = {n: n for n in nodes}

    for _ in range(max_passes):
        m2 = sum(sum(nbrs.values()) for nbrs in adj.values())  # == 2m
        if m2 <= 0:
            break
        degree = {n: sum(nbrs.values()) for n, nbrs in adj.items()}
        community: Dict[str, int] = {n: i for i, n in enumerate(sorted(adj))}
        comm_degree: Dict[int, float] = defaultdict(float)
        for n, c in community.items():
            comm_degree[c] += degree[n]

        improved = False
        moved = True
        while moved:
            moved = False
            for n in sorted(adj):
                c_old = community[n]
                to_comm: Dict[int, float] = defaultdict(float)
                for nbr, w in adj[n].items():
                    if nbr != n:
                        to_comm[community[nbr]] += w
                comm_degree[c_old] -= degree[n]
                base = to_comm.get(c_old, 0.0) - resolution * comm_degree[c_old] * degree[n] / m2
                best_c, best_delta = c_old, 0.0
                for c_new in sorted(to_comm):
                    if c_new == c_old:
                        continue
                    gain = to_comm[c_new] - resolution * comm_degree[c_new] * degree[n] / m2
                    delta = gain - base
                    if delta > best_delta + 1e-12:
                        best_delta, best_c = delta, c_new
                comm_degree[best_c] += degree[n]
                if best_c != c_old:
                    community[n] = best_c
                    moved = True
                    improved = True
        if not improved:
            break

        # Aggregate: one super-node per community (deterministic labels).
        # SELF-LOOPS ARE KEPT (adversary F1, 2026-07-17): dropping
        # intra-community weight made every pass after the first maximize
        # modularity of the WRONG graph (ring-of-cliques: 10 planted
        # cliques merged into ONE; Q=0.0 vs 0.81 correct). With the self
        # entry holding both folded directions (2x intra weight), row sums
        # keep degree[c] == the sum of member degrees and m2 invariant —
        # textbook Louvain. Local moves already exclude the self edge
        # (nbr != n), so no other site changes.
        label_of: Dict[int, str] = {}
        for n in sorted(adj):
            c = community[n]
            if c not in label_of:
                label_of[c] = f"c{len(label_of)}"
        new_adj: Dict[str, Dict[str, float]] = {}
        for n, nbrs in adj.items():
            cn = label_of[community[n]]
            row = new_adj.setdefault(cn, {})
            for nbr, w in nbrs.items():
                cb = label_of[community[nbr]]
                row[cb] = row.get(cb, 0.0) + w
        for gid, super_node in list(member_of.items()):
            member_of[gid] = label_of[community[super_node]]
        adj = new_adj
        if len(adj) <= 1:
            break

    final_ids = {sn: i for i, sn in enumerate(sorted(set(member_of.values())))}
    return {gid: final_ids[sn] for gid, sn in member_of.items()}


# --------------------------------------------------------------- labeling


_LABEL_STOP = frozenset(
    "the a an and or of to in on for with from this that entity memory memories record records "
    "exchange own time diary entry visit session".split()
)


def _community_labels(g: CommunityGraph, partition: Dict[str, int]) -> Dict[int, Dict[str, Any]]:
    members: Dict[int, List[str]] = defaultdict(list)
    for gid, c in partition.items():
        members[c].append(gid)
    # Document frequency of keywords ACROSS communities (distinctiveness).
    df: Counter = Counter()
    per_comm_counts: Dict[int, Counter] = {}
    for c, gids in members.items():
        counts: Counter = Counter()
        for gid in gids:
            for kw in set(g.nodes[gid].keywords):
                if kw not in _LABEL_STOP and len(kw) > 2:
                    counts[kw] += 1
        per_comm_counts[c] = counts
        for kw in counts:
            df[kw] += 1
    n_comm = max(1, len(members))
    out: Dict[int, Dict[str, Any]] = {}
    for c, gids in members.items():
        counts = per_comm_counts[c]
        scored = sorted(
            counts.items(),
            key=lambda kv: (-(kv[1] * (1.0 + (n_comm / df[kv[0]]))), kv[0]),
        )
        top_keywords = [kw for kw, _ in scored[:5]]

        # Exemplar: highest weighted INTERNAL degree; deterministic
        # tie-break by (titled first, gid).
        def internal_degree(gid: str) -> float:
            return sum(w for nbr, w in g.adj.get(gid, {}).items() if partition.get(nbr) == c)

        exemplar = max(
            sorted(gids),
            key=lambda gid: (internal_degree(gid), bool(g.nodes[gid].title), gid),
        )
        out[c] = {
            "keywords": top_keywords,
            "exemplar_id": exemplar,
            "exemplar_title": g.nodes[exemplar].title,
            "label": ", ".join(top_keywords[:3])
            if top_keywords
            else (g.nodes[exemplar].title[:60] or f"community {c}"),
        }
    return out


# ------------------------------------------------------------------ api


# The deterministic resolution sweep: accept the first γ whose partition
# is informative — at least 2 real communities AND the largest holds at
# most half the nodes; keep the best split seen if no γ satisfies both.
# (With correct aggregation γ=1 usually already passes; the sweep guards
# degenerate lives, not the algorithm.)
RESOLUTION_SWEEP: Tuple[float, ...] = (1.0, 1.6, 2.5, 4.0, 6.0, 10.0, 16.0)


def _partition_quality(g: CommunityGraph, partition: Dict[str, int]) -> Tuple[int, float]:
    sizes = Counter(partition.values())
    real = [n for n in sizes.values() if n >= MIN_COMMUNITY_SIZE]
    largest_frac = (max(sizes.values()) / max(1, len(partition))) if sizes else 1.0
    return len(real), largest_frac


def memory_communities(store: Any) -> Dict[str, Any]:
    """The one call consumers use: nodes+edges -> partition -> labeled
    communities, JSON-safe. Deterministic for a given store state."""
    g = collect_graph(store)
    partition: Dict[str, int] = {}
    chosen_resolution = RESOLUTION_SWEEP[0]
    best: Tuple[float, Dict[str, int], float] = (2.0, {}, chosen_resolution)  # (largest_frac, partition, γ)
    for gamma in RESOLUTION_SWEEP:
        p = louvain_partition(g, resolution=gamma)
        n_real, largest_frac = _partition_quality(g, p)
        if largest_frac < best[0] and n_real >= 2:
            best = (largest_frac, p, gamma)
        if n_real >= 2 and largest_frac <= 0.5:
            partition, chosen_resolution = p, gamma
            break
    if not partition:
        partition, chosen_resolution = (best[1] or louvain_partition(g)), best[2]
    labels = _community_labels(g, partition)

    members: Dict[int, List[str]] = defaultdict(list)
    for gid, c in partition.items():
        members[c].append(gid)

    communities: List[Dict[str, Any]] = []
    unclustered: List[str] = []
    for c in sorted(members, key=lambda cc: (-len(members[cc]), cc)):
        gids = sorted(members[c])
        if len(gids) < MIN_COMMUNITY_SIZE:
            unclustered.extend(gids)
            continue
        kinds = Counter(g.nodes[gid].kind for gid in gids)
        communities.append(
            {
                "id": c,
                "label": labels[c]["label"],
                "keywords": labels[c]["keywords"],
                "exemplar_id": labels[c]["exemplar_id"],
                "exemplar_title": labels[c]["exemplar_title"],
                "size": len(gids),
                "kinds": dict(kinds),
                "member_ids": gids,
            }
        )
    # Cross-community edge weights (the view draws faint inter-topic ties).
    cross: Dict[Tuple[int, int], float] = defaultdict(float)
    for a, nbrs in g.adj.items():
        ca = partition.get(a)
        for b, w in nbrs.items():
            cb = partition.get(b)
            if ca is None or cb is None or ca >= cb:
                continue
            cross[(ca, cb)] += w
    kept_ids = {c["id"] for c in communities}
    links = [
        {"a": a, "b": b, "weight": round(w, 3)}
        for (a, b), w in sorted(cross.items())
        if a in kept_ids and b in kept_ids
    ]

    return {
        "algorithm": "louvain-v1 (deterministic, resolution sweep) + keyword bridges",
        "node_count": len(g.nodes),
        "edge_count": sum(len(v) for v in g.adj.values()) // 2,
        "communities": communities,
        "unclustered": sorted(unclustered),
        "links": links,
        "params": {
            "resolution": chosen_resolution,
            "structural_weight": STRUCTURAL_WEIGHT,
            "keyword_bridge_weight": KEYWORD_BRIDGE_WEIGHT,
            "keyword_bridge_min_shared": KEYWORD_BRIDGE_MIN_SHARED,
            "keyword_bridge_max_per_node": KEYWORD_BRIDGE_MAX_PER_NODE,
            "min_community_size": MIN_COMMUNITY_SIZE,
        },
    }
