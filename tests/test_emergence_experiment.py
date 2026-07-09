"""The emergence experiment (a2a thread 0002, protocol ratified in 0002/001-002).

Falsifiable claim under test: at EQUAL token budget, a working set produced by
spreading activation + decay over a durable graph (Arm E) recalls a once-stated
fact and maintains topical focus at least as well as recency+embedding retrieval
(Arm B) — and additionally exhibits graceful decay and cue-driven reactivation
that the baseline cannot express.

Design constraints honored (0002/002):
- Arm B is the SAME engine constructed with `ablation="recency_embedding"`
  (implementation-controlled comparison; write paths stay live under ablation).
- `min_activation` left unset in BOTH arms (a threshold would bias Arm B).
- LLM-free: scripted fixtures; digests are pre-written; metrics are computed
  from handles (topic labels ride `handle.provenance["topic"]`).
- Deterministic: a hash bag-of-tokens embedder; determinism asserted by
  re-running the probe with a pinned `as_of` and comparing byte-identical JSON.

Kill criterion (ratified): if Arm E cannot match Arm B on probe recall at equal
budget, the strong "emergent STM" claim is NOT earned — this test failing is
that signal, by design.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractruntime.core.models import Effect, EffectType
from abstractruntime.integrations.abstractmemory import build_memory_seam_effect_handlers
from abstractruntime.storage.artifacts import InMemoryArtifactStore

pytest.importorskip("abstractmemory")

from abstractmemory import InMemoryJournal, InMemoryTripleStore, MemorySystem  # noqa: E402

OWNER = "exp-run"
SCOPES = [["run", OWNER]]
# Equal in both arms. Deliberately TIGHT (~3-4 digests): the first harness run
# used 220 tokens, which admitted every record every turn — everything was
# "prompt-visible", every commit re-deposited every record's trail, and decay
# was structurally impossible. Scarcity is what makes selection (and therefore
# forgetting) real; this mirrors real prompt economics.
TOKEN_BUDGET = 70
SHELF_SIZE = 6


class _Run:
    run_id = OWNER
    session_id = None


_STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "in", "on", "at", "we", "us", "our",
    "let's", "lets", "now", "what", "should", "use", "any", "per", "with",
    "and", "or", "is", "are", "be", "was", "did", "do", "about", "that", "this",
}


class _BagOfTokensEmbedder:
    """Deterministic, dependency-free embedder: hashed bag-of-tokens, L2-normed.

    Stopwords are stripped so cosine similarity reflects CONTENT overlap only.
    Honesty note (validity review): the decisive fix for the early decay
    failure was BUDGET SCARCITY (token_budget 220 -> 70), not this strip — the
    reviewer's 2x2 counterfactual showed decay passes with stopwords kept at
    budget 70. The strip stays because it makes the vector channel's semantics
    cleaner, not because it rescues any assertion.     No model, no drift (the
    0014 embedding-manifest caveat from 0002/002 is moot by construction).

    dim is 512, not 64: at 64 buckets, ~40 distinct content tokens collide
    often, giving unrelated digests a spurious non-zero cosine — which the
    validity review exposed once the store actually computed vectors (the gold
    fact got weakly re-selected on off-topic turns and never decayed). A wider
    space makes collisions rare, so "no lexical overlap" means "~zero
    similarity" as intended. This is instrument denoising, not result tuning.
    """

    dim = 512

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in str(text).lower().split():
                tok = tok.strip(".,;:!?()[]\"'")
                if not tok or tok in _STOPWORDS:
                    continue
                idx = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "big") % self.dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


# ---------------------------------------------------------------------------
# The scripted session: topic A (db/pool), topic B (auth refactor), distractors.
# The GOLD fact is stated once at turn 2 and never mentioned again until the probe.
# ---------------------------------------------------------------------------

GOLD_DIGEST = "decided to use pgbouncer in transaction mode for the connection pool"
# A fact stated ONCE (turn 0) and never cued again. It is the graceful-forgetting
# probe: unlike the gold fact (repeatedly used, so correctly stays warm under the
# union model), the decoy is used once and must DECAY OUT of the working set.
DECOY_DIGEST = "project uses postgres 16 as the primary database"

TURNS: List[Dict[str, Any]] = [
    {"cue": "let's set up the database for the project", "topic": "A",
     "record": {"title": "primary database", "digest": "project uses postgres 16 as the primary database", "keywords": ["postgres", "database"]}},
    {"cue": "what pooler should we use for postgres connections?", "topic": "A", "gold": True,
     "record": {"title": "connection pool decision", "digest": GOLD_DIGEST, "keywords": ["pgbouncer", "pool", "connection"]}},
    {"cue": "now let's refactor the auth module", "topic": "B",
     "record": {"title": "auth refactor start", "digest": "auth module refactor: extract token validation into middleware", "keywords": ["auth", "middleware"]}},
    {"cue": "rename the session helpers in auth", "topic": "B",
     "record": {"title": "session helper rename", "digest": "renamed session helpers to snake_case across the auth module", "keywords": ["auth", "session"]}},
    {"cue": "any good lunch spot nearby?", "topic": "distractor",
     "record": {"title": "lunch", "digest": "team prefers the ramen place for lunch", "keywords": ["lunch"]}},
    {"cue": "split the login controller in the auth module", "topic": "B",
     "record": {"title": "login controller split", "digest": "login controller split into authn and authz services", "keywords": ["auth", "login"]}},
    {"cue": "add tests for the auth middleware", "topic": "B",
     "record": {"title": "middleware tests", "digest": "middleware token tests added covering expiry and refresh", "keywords": ["auth", "tests"]}},
    {"cue": "what color should the logo be?", "topic": "distractor",
     "record": {"title": "logo", "digest": "the logo should be blue per design", "keywords": ["logo"]}},
    {"cue": "improve error handling in authz", "topic": "B",
     "record": {"title": "authz errors", "digest": "authz errors now map to 403 with problem json bodies", "keywords": ["auth", "errors"]}},
    {"cue": "add rate limiting to login", "topic": "B",
     "record": {"title": "rate limiting", "digest": "added rate limiting to the login endpoint at ten requests per minute", "keywords": ["auth", "rate"]}},
    {"cue": "add audit logging for auth events", "topic": "B",
     "record": {"title": "audit logging", "digest": "auth events now emit audit logs with actor ids", "keywords": ["auth", "audit"]}},
    {"cue": "clean up dead code in the auth module", "topic": "B",
     "record": {"title": "dead code cleanup", "digest": "removed legacy session shim from the auth module", "keywords": ["auth", "cleanup"]}},
]

PROBE_CUE = "what did we decide about the connection pool?"
B_TURN_INDICES = [i for i, t in enumerate(TURNS) if t["topic"] == "B"]


def _memory_system(ablation: Optional[str]) -> MemorySystem:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        embedder = _BagOfTokensEmbedder()
        return MemorySystem(
            # The STORE gets the embedder too: vectors are computed at WRITE
            # time by the store. The validity review caught the first harness
            # building the store without one — every row had vector=None, the
            # vector channel returned nothing, and both arms silently reduced
            # to keyword retrieval.
            store=InMemoryTripleStore(embedder=embedder),
            journal=InMemoryJournal(),
            embedder=embedder,
            ablation=ablation,
        )


def _serialize_under_budget(handles: List[Dict[str, Any]], token_budget: int) -> List[Dict[str, Any]]:
    """The equal-budget serializer: take handles in engine order while the
    running token estimate fits. This is 'what entered the prompt'."""
    rendered: List[Dict[str, Any]] = []
    used = 0
    for h in handles:
        cost = int(h.get("token_estimate") or 0)
        if used + cost > token_budget:
            continue
        rendered.append(h)
        used += cost
    return rendered


class Arm:
    """One experiment arm driven ONLY through the runtime effect handlers."""

    def __init__(self, name: str, ablation: Optional[str]):
        self.name = name
        self.ms = _memory_system(ablation)
        self.handlers = build_memory_seam_effect_handlers(
            memory_system=self.ms, run_store=None, now_iso=lambda: "2026-07-06T00:00:00+00:00",
            artifact_store=InMemoryArtifactStore(),
        )
        self.gold_handle_id: Optional[str] = None
        self.selected_by_turn: List[List[Dict[str, Any]]] = []
        self.activation_track: Dict[str, float] = {}

    def _effect(self, etype: EffectType, payload: Dict[str, Any]):
        handler = self.handlers[etype]
        out = handler(_Run(), Effect(type=etype, payload=payload), None)
        assert out.status == "completed", f"{self.name}: {etype} failed: {getattr(out, 'error', None)}"
        json.dumps(out.result)  # every effect result must be ledger-safe
        return out.result

    def form(self, turn_idx: int, turn: Dict[str, Any]) -> None:
        rec = dict(turn["record"])
        rec.update({"kind": "memory", "topic": turn["topic"], "verbatim": f"turn {turn_idx}: {turn['cue']}"})
        self._effect(
            EffectType.MEMORY_FORM,
            {"records": [rec], "scope": "run", "owner_id": OWNER, "turn_id": f"t{turn_idx}"},
        )

    def recall(self, cue: str, turn_idx: int, as_of: Optional[int] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "cue_text": cue,
            "scopes": SCOPES,
            "view": "working_set",
            "turn_id": f"t{turn_idx}",
            "budget": {"token_budget": TOKEN_BUDGET, "shelf_size": SHELF_SIZE},
        }
        if as_of is not None:
            payload["as_of"] = as_of
        return self._effect(EffectType.MEMORY_RECALL, payload)

    def commit(self, result: Dict[str, Any], rendered: List[Dict[str, Any]]) -> None:
        used = [h["record_id"] for h in rendered]
        if not used:
            return
        self._effect(
            EffectType.MEMORY_ACCESS,
            {"trace_id": result["trace_id"], "used_record_ids": used,
             "prompt_token_estimate": sum(int(h.get("token_estimate") or 0) for h in rendered)},
        )

    def gold_activation(self) -> float:
        if not self.gold_handle_id:
            return 0.0
        scores = self.ms.activation([self.gold_handle_id], scope="run", owner_id=OWNER)
        return float(scores.get(self.gold_handle_id, {}).get("base_level", 0.0))

    def run_session(self) -> None:
        for i, turn in enumerate(TURNS):
            self.form(i, turn)
            result = self.recall(turn["cue"], i)
            rendered = _serialize_under_budget(result["handles"], TOKEN_BUDGET)
            self.selected_by_turn.append(rendered)
            self.commit(result, rendered)
            if turn.get("gold"):
                gold = [h for h in rendered if GOLD_DIGEST in h.get("digest", "")]
                assert gold, f"{self.name}: gold fact must be selected on its own turn"
                self.gold_handle_id = gold[0]["record_id"]
            if i == 2 and self.name == "E":
                self.activation_track["after_gold_use"] = self.gold_activation()
        self.activation_track["before_probe"] = self.gold_activation()

    def probe(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        result = self.recall(PROBE_CUE, len(TURNS))
        rendered = _serialize_under_budget(result["handles"], TOKEN_BUDGET)
        self.commit(result, rendered)
        self.activation_track["after_probe"] = self.gold_activation()
        return result, rendered

    # -- metrics ------------------------------------------------------------

    def probe_recall(self, rendered: List[Dict[str, Any]]) -> bool:
        return any(GOLD_DIGEST in h.get("digest", "") for h in rendered)

    def focus_coherence(self) -> float:
        """Mean share of topic-B items among STIMULUS-admitted handles on B turns.

        Seam v1.1 (union model): STM members are stimulus-independent CONTINUITY,
        not focus (memory 0001/012). Focus is a property of what the current
        stimulus pulls in, so it is measured over admission in {stimulus, both}.
        STM continuity is measured separately (it is a feature, not a focus loss).
        """
        ratios: List[float] = []
        for i in B_TURN_INDICES:
            sel = [h for h in self.selected_by_turn[i] if h.get("admission") in ("stimulus", "both")]
            if not sel:
                continue
            b = sum(1 for h in sel if (h.get("provenance") or {}).get("topic") == "B")
            ratios.append(b / len(sel))
        return sum(ratios) / len(ratios) if ratios else 0.0

    def decoy_last_present_turn(self) -> int:
        """Last turn index at which the once-used decoy fact was in the working
        set. Behavioral decay = it EXITS and stays out (not merely scores lower)."""
        last = -1
        for i in range(len(TURNS)):
            if any(DECOY_DIGEST in h.get("digest", "") for h in self.selected_by_turn[i]):
                last = i
        return last


@pytest.fixture(scope="module")
def arms() -> Dict[str, Arm]:
    arm_e = Arm("E", ablation=None)
    arm_b = Arm("B", ablation="recency_embedding")
    arm_e.run_session()
    arm_b.run_session()
    return {"E": arm_e, "B": arm_b}


def test_arm_b_is_labeled_ablation(arms):
    result = arms["B"].recall("sanity check cue", 99)
    assert any("ablation" in w for w in result.get("warnings", ())), "Arm B runs must self-label"


def test_probe_recall_arm_e_matches_or_beats_baseline(arms):
    _, rendered_e = arms["E"].probe()
    _, rendered_b = arms["B"].probe()
    recall_e = arms["E"].probe_recall(rendered_e)
    recall_b = arms["B"].probe_recall(rendered_b)
    # Ratified pass line 1 + kill criterion: E must not lose to B on probe recall.
    assert recall_e or not recall_b, (
        f"KILL CRITERION HIT: Arm E missed the gold fact while Arm B found it "
        f"(E={recall_e}, B={recall_b}) — the strong emergent-STM claim is not earned."
    )
    # The gold fact was stated once, 10+ turns earlier: at least one arm must recover it
    # for the experiment to be meaningful at this budget.
    assert recall_e or recall_b, "neither arm recovered the gold fact — budget or channels are broken"


def test_focus_coherence_arm_e_matches_or_beats_baseline(arms):
    focus_e = arms["E"].focus_coherence()
    focus_b = arms["B"].focus_coherence()
    assert focus_e >= focus_b - 1e-9, f"Arm E lost focus vs baseline (E={focus_e:.3f}, B={focus_b:.3f})"
    assert focus_e > 0.5, f"Arm E working sets on topic-B turns are not B-dominated (focus={focus_e:.3f})"


def test_behavioral_decay_once_used_fact_exits_working_set(arms):
    """Graceful forgetting, measured BEHAVIORALLY (memory 0002/004 + 0002/006):
    a fact used ONCE must EXIT the working set as unrelated activity accumulates
    — not merely score lower. The gold fact is the wrong probe for this under the
    union model (it is re-used every time it's rendered, so it correctly stays
    warm — that is continuity, not a decay failure). The DECOY (turn 0, never
    cued again) is the right probe: it must fall out and stay out by end of
    session.
    """
    e = arms["E"]
    last = e.decoy_last_present_turn()
    assert 0 <= last < len(TURNS) - 3, (
        f"no graceful forgetting: the once-used decoy fact was still in the working "
        f"set at turn {last} of {len(TURNS)} — it should have decayed out well before the end"
    )
    # And it must be absent from the FINAL working set (stayed out).
    assert not any(DECOY_DIGEST in h.get("digest", "") for h in e.selected_by_turn[-1]), (
        "decoy fact reappeared in the final working set — decay did not stick"
    )


def test_gold_stays_warm_under_union_continuity(arms):
    """The union model's counterpart to decay: a REPEATEDLY-relevant fact should
    remain reconstructable (as STM continuity or stimulus match), so recall never
    'forgets' something that keeps mattering. This is why the decay test uses the
    decoy, not the gold."""
    e = arms["E"]
    # The gold accumulated real usage activation on its turn.
    assert e.activation_track.get("after_gold_use", 0.0) > 0.0, (
        "gold fact never accumulated activation on its own turn"
    )
    # It remains recoverable at the probe (continuity + reactivation).
    result = e.recall(PROBE_CUE, 400)
    assert any(GOLD_DIGEST in h.get("digest", "") for h in result["handles"]), (
        "gold fact — repeatedly relevant — was lost; continuity failed"
    )


def test_reactivation_probe_cue_relifts_the_dormant_fact(arms):
    e = arms["E"]
    result, rendered = e.recall(PROBE_CUE, 200), None  # fresh read; no commit (read-only check)
    gold = [h for h in result["handles"] if GOLD_DIGEST in h.get("digest", "")]
    assert gold, "probe cue did not surface the gold fact at all"
    h = gold[0]
    relevance = h.get("relevance") or {}
    activation = h.get("activation") or {}
    assert any(v > 0 for v in relevance.values()) or activation.get("spread", 0) > 0, (
        "reactivation not attributable: no channel match and no spread on the gold handle"
    )
    assert h.get("cues"), "handle must explain WHY it surfaced (cues are mandatory)"
    # And the committed probe (from the fixture run) re-deposited the trail:
    assert e.activation_track["after_probe"] > e.activation_track["before_probe"], (
        "probe selection did not re-lift the dormant fact's activation (no pheromone re-deposit)"
    )


def _selection_content(result: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the LLM/agent would consume — minus per-call audit identity.

    `trace_id` is a fresh uuid per reconstruct() BY CONTRACT (it names the audit
    trace, not the selection); determinism is claimed for the SELECTION content.
    """
    out = dict(result)
    out.pop("trace_id", None)
    return out


def test_spreading_only_probe_arm_e_strictly_beats_baseline():
    """The hardest test (validity review): a probe with ZERO lexical overlap
    with the gold fact, answerable ONLY by spreading activation across a graph
    edge. This is where Arm E must strictly BEAT Arm B — channels alone (Arm B)
    cannot reach the gold; only the edge-walk (Arm E) can.

    Fixture: a `runbook` record `dcterms:references` the gold pool decision.
    Probe matches the runbook's vocabulary, never the gold's. In Arm E the
    runbook is channel-matched (seed) and spreading carries activation across
    the reference edge to the gold; in Arm B spreading is off, so the gold has
    zero activation and is filtered out by the membership floor.

    `min_activation` IS set here (unlike the main experiment, per memory's
    0002/002 note) — for a spread-only probe it is the CORRECT instrument, not a
    bias: it filters both arms equally, and only spread-carried activation lets
    the gold survive. That asymmetry is exactly the emergence being tested.
    """
    gold_digest = "chose pgbouncer transaction mode for the pool"

    def build(arm: Arm) -> str:
        # Turn 1: the gold fact. Capture its GRAPH id (the edge target).
        form = arm._effect(
            EffectType.MEMORY_FORM,
            {
                "records": [{"kind": "memory", "title": "pool decision", "digest": gold_digest,
                             "keywords": ["pgbouncer"], "topic": "A"}],
                "scope": "run", "owner_id": OWNER, "turn_id": "g",
            },
        )
        gold_graph_id = form["record_ids"][0]
        # Turn 2: a runbook that REFERENCES the gold, with disjoint vocabulary.
        arm._effect(
            EffectType.MEMORY_FORM,
            {
                "records": [{"kind": "memory", "title": "runbook", "topic": "A",
                             "digest": "infrastructure runbook documents deployment topology",
                             "keywords": ["runbook", "topology"],
                             "edges": [["dcterms:references", gold_graph_id]]}],
                "scope": "run", "owner_id": OWNER, "turn_id": "b",
            },
        )
        # Some unrelated noise.
        for i, tx in enumerate(["auth middleware refactor extracted validation",
                                "login controller split into services",
                                "logo should be blue per design"]):
            arm._effect(
                EffectType.MEMORY_FORM,
                {"records": [{"kind": "memory", "title": f"n{i}", "digest": tx,
                              "keywords": [tx.split()[0]], "topic": "B"}],
                 "scope": "run", "owner_id": OWNER, "turn_id": f"n{i}"},
            )
        return gold_graph_id

    probe = "where is deployment topology documented"
    budget = {"token_budget": 400, "shelf_size": 12, "min_activation": 0.001}

    arm_e = Arm("E", ablation=None)
    arm_b = Arm("B", ablation="recency_embedding")
    build(arm_e)
    build(arm_b)

    res_e = arm_e._effect(
        EffectType.MEMORY_RECALL,
        {"cue_text": probe, "scopes": SCOPES, "view": "working_set", "turn_id": "probe", "budget": budget},
    )
    res_b = arm_b._effect(
        EffectType.MEMORY_RECALL,
        {"cue_text": probe, "scopes": SCOPES, "view": "working_set", "turn_id": "probe", "budget": budget},
    )

    gold_e = [h for h in res_e["handles"] if "pgbouncer" in h.get("digest", "")]
    gold_b = [h for h in res_b["handles"] if "pgbouncer" in h.get("digest", "")]

    # Arm E reaches the gold; Arm B cannot. This is the strict emergence win.
    assert gold_e, "Arm E failed to reach the gold fact via spreading"
    assert not gold_b, "Arm B (no spreading) should NOT reach a zero-overlap gold fact"

    # And the reason is spread, not a sneaky channel match: base_level 0, spread > 0.
    act = gold_e[0].get("activation") or {}
    assert act.get("spread", 0) > 0, f"gold should arrive via spread, got activation={act}"
    assert (gold_e[0].get("relevance") or {}) == {} or all(
        v == 0 for v in (gold_e[0].get("relevance") or {}).values()
    ), "gold must NOT be channel-matched by a zero-overlap probe (else it isn't a spreading test)"


def test_e0_vs_e1_continuity_isolates_the_union_contribution():
    """The E0/E1 control (union token-economics review): the union model was in
    every result but measured by none. This isolates the union's CONTINUITY
    contribution from channels and spreading.

    Probe fact: used TWICE early (so it is trail-hot -> eligible for STM),
    vocabulary-DISJOINT from the checkpoint cue (channels cannot match it), and
    EDGE-FREE (spreading cannot reach it). Under budget SCARCITY with competing
    channel-matched records:
      - E1 (union, stm_fraction>0): present via admission="stm" (the STM
        reservation carries trail-hot continuity the stimulus alone would drop);
      - E0 (stm_fraction=0):        ABSENT (no STM reservation; the scarce budget
        goes to channel matches; the disjoint fact has no other way in).
    The difference is attributable to the union alone — not channels (disjoint),
    not spreading (edge-free). Scarcity is required: at a loose budget E0 also
    admits it via cheap unmatched fill (the C3-gap the review flagged), which
    would mask the isolation.
    """
    CONT = "zorblax calibration constant equals seventeen units"
    checkpoint_cue = "widget status report"

    def build(stm_fraction: float):
        arm = Arm("E1" if stm_fraction else "E0", ablation=None)

        def eff(etype, payload):
            return arm._effect(etype, payload)

        eff(EffectType.MEMORY_FORM, {
            "records": [{"kind": "memory", "title": "zorblax", "digest": CONT, "keywords": ["zorblax"], "topic": "Z"}],
            "scope": "run", "owner_id": OWNER, "turn_id": "g0",
        })
        # Use it TWICE (channel-matched cue + commit) -> trail-hot.
        for k in ("u1", "u2"):
            r = eff(EffectType.MEMORY_RECALL, {
                "cue_text": "zorblax calibration", "scopes": SCOPES, "view": "working_set",
                "turn_id": k, "budget": {"token_budget": 120, "shelf_size": 6, "stm_fraction": stm_fraction},
            })
            used = [h["record_id"] for h in r["handles"] if CONT in h.get("digest", "")]
            if used:
                eff(EffectType.MEMORY_ACCESS, {"trace_id": r["trace_id"], "used_record_ids": used})
        # Competing channel-matched records that will crowd the scarce budget.
        for i in range(5):
            eff(EffectType.MEMORY_FORM, {
                "records": [{"kind": "memory", "title": f"w{i}", "digest": f"widget status report {i} nominal ok",
                             "keywords": ["widget", "status"], "topic": "W"}],
                "scope": "run", "owner_id": OWNER, "turn_id": f"n{i}",
            })
        # Checkpoint: disjoint cue, SCARCE budget (competing matches fill it).
        return eff(EffectType.MEMORY_RECALL, {
            "cue_text": checkpoint_cue, "scopes": SCOPES, "view": "working_set", "turn_id": "chk",
            "budget": {"token_budget": 60, "shelf_size": 3, "stm_fraction": stm_fraction},
        })

    res_e1 = build(0.35)
    res_e0 = build(0.0)
    cont_e1 = [h for h in res_e1["handles"] if CONT in h.get("digest", "")]
    cont_e0 = [h for h in res_e0["handles"] if CONT in h.get("digest", "")]

    assert cont_e1, "E1 (union) failed to carry the trail-hot continuity fact into the working set"
    assert cont_e1[0].get("admission") == "stm", (
        f"continuity fact must enter E1 via STM, not channels/spreading; got admission={cont_e1[0].get('admission')!r}"
    )
    # Attribution: not a sneaky channel match, not spread.
    assert not (cont_e1[0].get("relevance") or {}), "continuity fact must NOT be channel-matched (disjoint cue)"
    assert (cont_e1[0].get("activation") or {}).get("spread", 0) == 0, "continuity fact is edge-free; spread must be 0"
    # The control: without the union, the same fact is gone under scarcity.
    assert not cont_e0, (
        "E0 (stm_fraction=0) should NOT surface the disjoint, edge-free continuity fact under scarcity — "
        "if it does, the union's contribution is not what's being measured"
    )


def test_determinism_probe_replay_is_byte_identical(arms):
    e = arms["E"]
    first = e.recall(PROBE_CUE, 300)
    pinned = first["as_of_seq"]
    again = e.recall(PROBE_CUE, 300, as_of=pinned)
    replay = e.recall(PROBE_CUE, 300, as_of=pinned)
    assert json.dumps(_selection_content(again), sort_keys=True) == json.dumps(
        _selection_content(replay), sort_keys=True
    ), "replay with pinned as_of is not deterministic"
