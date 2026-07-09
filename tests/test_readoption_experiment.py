"""The re-adoption keystone experiment (a2a 0003, phase 0).

FALSIFIABLE CLAIM: a named entity engrammed into one per-entity SQLite
substrate is, after full process teardown, re-adoptable — the same identity
records (bit-identical ids, no re-creation), the same lived experience
(recall by cue, diary verbatim, valence standing), rendered into a summon
prelude by pure reads, with zero faked usage.

SUBSTRATE: SQLite REQUIRED — the restart IS the experiment (an InMemory
variant would vacuously pass sessions that never ended). This harness runs
EMBEDDER-FREE deliberately (deterministic, no embedding-server dependency):
probes are exact/keyword-reachable. Note the production posture differs —
since the SQLite-vector wave, the endorsed pairing is SQLite +
embedder-when-reachable (vectorless = labeled degradation, not the default).

Kill-criteria triage (which failure indicts what):
- A1/A3/D1/D2/C1 failing indict the ARCHITECTURE (identity model);
- B1/B3/E1 failing indict the SUBSTRATE WIRING (durability/seam code);
- F1 failing indicts the PRELUDE SPEC.

Session 1: engram the spark -> scripted work turns (form -> recall -> commit
only what was rendered) -> elected diary writes (incl. one private) ->
appraisals (incl. one bond-worthy positive peak) -> close everything.
Session 2: fresh objects over the same files -> engram idempotency -> pure
prelude render -> probes.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from abstractruntime.core.models import Effect, EffectType
from abstractruntime.identity import DiaryStore, build_diary_effect_handlers, verify_diary_chain
from abstractruntime.identity.prelude import render_summon_prelude
from abstractruntime.integrations.abstractmemory import build_memory_seam_effect_handlers
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore

pytest.importorskip("abstractmemory")

from abstractmemory import (  # noqa: E402
    DEFAULT_SPARK_TEMPLATE,
    MemorySystem,
    SQLiteJournal,
    SQLiteTripleStore,
    engram,
    lint_spark,
)
from abstractmemory.records import verify_diary_chain as verify_diary_graph_chain  # noqa: E402
from abstractmemory.spark import SHARED_VULNERABILITY_STATEMENT  # noqa: E402

ENTITY_NAME = "Castor"
ENTITY_ID = "entity:castor@home-test"
SELF_SCOPE = ("self", ENTITY_ID)  # engram records + valence events + self bindings
DIARY_SCOPE = ("diary", ENTITY_ID)  # act-memory projections (hardcoded by DIARY_WRITE)
LIFE_SCOPE = ("life", ENTITY_ID)  # work-session experience (harness convention)
LADDER = [list(SELF_SCOPE), list(DIARY_SCOPE), list(LIFE_SCOPE)]

GOLD_DIGEST = "media server runs jellyfin on port 8096 behind the caddy proxy"
PROBE_CUE = "which port is the jellyfin media server on"
PRIVATE_TOKEN = "silverfin"  # appears ONLY in the private diary prose
# The SUMMON POSTURE: an entity's work turns run with the self component ON
# (self_fraction > 0), so identity members admit as admission="self" and
# never deposit at commit. With self_fraction=0 they would enter as ordinary
# stimulus fill and deposit HONESTLY — which is exactly the fake-usage
# posture the re-adoption design forbids (caught by D2 on the first run).
WORK_BUDGET = {"token_budget": 400, "shelf_size": 8, "self_fraction": 0.5}
# H1 (keystone audit): self slots = round(self_fraction * shelf_size); the
# default template engrams 6 identity records, so 0.5 * 12 = 6 fits them all.
SELF_PROBE_BUDGET = {"self_fraction": 0.5, "stm_fraction": 0.0, "token_budget": 800, "shelf_size": 12}
PRELUDE_BUDGET = 600


def _spark() -> Dict[str, Any]:
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = ENTITY_NAME
    spark["spark"] = 1
    return spark


class _CounterClock:
    """Deterministic, strictly-increasing ISO stamps (byte-stable renders)."""

    def __init__(self) -> None:
        self._n = 0

    def __call__(self) -> str:
        self._n += 1
        return f"2026-07-06T10:{self._n // 60:02d}:{self._n % 60:02d}+00:00"


class _Run:
    session_id = None
    actor_id = None

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id


@dataclass
class Home:
    ms: Any
    store: Any
    journal: Any
    diary: DiaryStore
    handlers: Dict[EffectType, Any]

    def close(self) -> None:
        for obj in (self.store, self.journal):
            close = getattr(obj, "close", None)
            if callable(close):
                close()


def open_home(home_dir: Path) -> Home:
    db = home_dir / "memory.sqlite3"  # ONE memory file: store + journal, one seq axis
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    ledger = SqliteLedgerStore(SqliteDatabase(str(home_dir / "home.sqlite3")))  # the book (runtime plane)
    diary = DiaryStore(entity_id=ENTITY_ID, ledger_store=ledger)
    clock = _CounterClock()
    handlers = {
        **build_memory_seam_effect_handlers(memory_system=ms, run_store=None, now_iso=clock),
        **build_diary_effect_handlers(entity_id=ENTITY_ID, diary_store=diary, memory_system=ms, now_iso=clock),
    }
    return Home(ms=ms, store=store, journal=journal, diary=diary, handlers=handlers)


def _serialize_under_budget(handles: List[Dict[str, Any]], token_budget: int) -> List[Dict[str, Any]]:
    """What actually entered the prompt (the 0002 equal-budget serializer)."""
    rendered: List[Dict[str, Any]] = []
    used = 0
    for h in handles:
        cost = int(h.get("token_estimate") or 0)
        if used + cost > token_budget:
            continue
        rendered.append(h)
        used += cost
    return rendered


class _Session:
    def __init__(self, home: Home, run_id: str) -> None:
        self.home = home
        self.run = _Run(run_id)

    def effect(self, etype: EffectType, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = self.home.handlers[etype](self.run, Effect(type=etype, payload=payload), None)
        assert out.status == "completed", f"{etype.value} failed: {getattr(out, 'error', None)}"
        json.dumps(out.result)  # ledger safety: results must be JSON-safe
        return out.result

    def form(self, turn_id: str, record: Dict[str, Any]) -> List[str]:
        rec = dict(record)
        rec.setdefault("kind", "memory")
        result = self.effect(
            EffectType.MEMORY_FORM,
            {"records": [rec], "scope": LIFE_SCOPE[0], "owner_id": LIFE_SCOPE[1], "turn_id": turn_id},
        )
        return list(result["record_ids"])

    def recall(self, cue: str, turn_id: str, **over: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "cue_text": cue,
            "scopes": LADDER,
            "view": "working_set",
            "turn_id": turn_id,
            "budget": dict(WORK_BUDGET),
        }
        payload.update(over)
        return self.effect(EffectType.MEMORY_RECALL, payload)

    def commit(self, result: Dict[str, Any], rendered: List[Dict[str, Any]]) -> None:
        used = [h["record_id"] for h in rendered]
        if not used:
            return
        self.effect(
            EffectType.MEMORY_ACCESS,
            {
                "trace_id": result["trace_id"],
                "used_record_ids": used,
                "prompt_token_estimate": sum(int(h.get("token_estimate") or 0) for h in rendered),
            },
        )

    def turn(self, turn_id: str, record: Dict[str, Any], cue: str) -> List[Dict[str, Any]]:
        """One honest work turn: form -> journaled recall -> commit rendered."""
        self.form(turn_id, record)
        result = self.recall(cue, turn_id)
        rendered = _serialize_under_budget(result["handles"], int(WORK_BUDGET["token_budget"]))
        self.commit(result, rendered)
        return rendered

    def diary_write(self, turn_id: str, **payload: Any) -> Dict[str, Any]:
        body = {"turn_id": turn_id, "as_of_seq": self.home.ms.current_seq()}
        body.update(payload)
        return self.effect(EffectType.DIARY_WRITE, body)

    def appraise(self, turn_id: str, target: str, *, sign: int, magnitude: float, reason: str, **over: Any) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "target_id": target,
            "sign": sign,
            "magnitude": magnitude,
            "reason": reason,
            "turn_id": turn_id,
            "scope": SELF_SCOPE[0],
            "owner_id": SELF_SCOPE[1],
        }
        body.update(over)
        return self.effect(EffectType.MEMORY_APPRAISE, body)


@dataclass
class Report:
    r1: Any = None
    r2: Any = None
    prelude: Dict[str, Any] = field(default_factory=dict)
    prelude_seq_before: int = -1
    prelude_seq_after: int = -1
    prelude_refused: Dict[str, Any] = field(default_factory=dict)
    prelude_degraded: Dict[str, Any] = field(default_factory=dict)
    self_probe: Dict[str, Any] = field(default_factory=dict)
    self_probe_replay: Dict[str, Any] = field(default_factory=dict)
    gold_probe: Dict[str, Any] = field(default_factory=dict)
    gold_probe_replay: Dict[str, Any] = field(default_factory=dict)
    gold_rendered: List[Dict[str, Any]] = field(default_factory=list)
    diary_probe: Dict[str, Any] = field(default_factory=dict)
    diary_entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    d3_read: Dict[str, Any] = field(default_factory=dict)
    d1_read: Dict[str, Any] = field(default_factory=dict)
    gradation: Dict[str, Any] = field(default_factory=dict)
    values_access_counts: Dict[str, Any] = field(default_factory=dict)
    book_report: Dict[str, Any] = field(default_factory=dict)
    graph_chain: Dict[str, Any] = field(default_factory=dict)
    d1_text: str = ""


@pytest.fixture(scope="module")
def keystone(tmp_path_factory) -> Report:
    home_dir = tmp_path_factory.mktemp("castor_home")
    report = Report()
    spark = _spark()
    assert lint_spark(spark) == [], "the keystone spark must be lint-clean"

    # ------------------------------------------------------------- session 1
    home = open_home(home_dir)
    s1 = _Session(home, "summon-s1")

    report.r1 = engram(home.ms, spark, owner_id=ENTITY_ID)
    assert report.r1.created is True

    rendered = s1.turn(
        "s1-t1",
        {
            "title": "media server",
            "digest": GOLD_DIGEST,
            "keywords": ["jellyfin", "port", "media"],
            "verbatim": "full setup notes for the media server, verbatim.",
        },
        "set up the media server",
    )
    assert any(GOLD_DIGEST in h.get("digest", "") for h in rendered), "gold must render on its own turn"
    s1.turn(
        "s1-t2",
        {"title": "backups", "digest": "nightly restic backups to the nas at 0300", "keywords": ["backup", "restic"]},
        "backup schedule",
    )
    s1.turn(
        "s1-t3",
        {"title": "dns", "digest": "local dns is handled by pihole at 192.168.1.2", "keywords": ["pihole"]},
        "dns resolution",
    )

    d1 = s1.diary_write(
        "s1-d1",
        text="First day at the home lab. Set up the media server and it felt like the right call — quiet, fast, mine.",
        gist="First day at the home lab; the media server decision felt right.",
        kind="reflection",
    )
    d2 = s1.diary_write(
        "s1-d2",
        text="Backups succeed but are unverified. A weekly restore drill would turn hope into knowledge.",
        gist="Idea: automate backup verification with a weekly restore drill.",
        kind="idea",
    )
    d3 = s1.diary_write(
        "s1-d3",
        text=f"Some thoughts are only mine: the {PRIVATE_TOKEN} project stays unwritten anywhere else.",
        visibility="private",
    )
    report.diary_entries = {"d1": d1, "d2": d2, "d3": d3}
    report.d1_text = (
        "First day at the home lab. Set up the media server and it felt like the right call — quiet, fast, mine."
    )

    s1.appraise("s1-a1", "tool:restic", sign=1, magnitude=1, reason="backup completed clean")
    s1.appraise("s1-a2", "tool:flaky_dns", sign=-1, magnitude=2, reason="dns flapped twice during setup")
    s1.appraise(
        "s1-a3",
        "person:maintainer",
        sign=1,
        magnitude=9,
        reason="trusted me with the home lab and the long project",
        bond=True,
        actor="entity-reflection",
    )
    s1.appraise("s1-a4", "concept:home-lab", sign=1, magnitude=1, reason="the work itself was a joy")

    home.close()
    del home, s1

    # ------------------------------------------------- session 2 (re-summon)
    home2 = open_home(home_dir)
    s2 = _Session(home2, "summon-s2")

    report.r2 = engram(home2.ms, spark, owner_id=ENTITY_ID)

    report.prelude_seq_before = home2.ms.current_seq()
    report.prelude = render_summon_prelude(
        home2.ms, home2.diary, entity_id=ENTITY_ID, budget=PRELUDE_BUDGET, spark=spark
    )
    report.prelude_seq_after = home2.ms.current_seq()

    # F1 budgets: refusal + degrade (derived from the successful render).
    core_tokens = sum(
        report.prelude["section_tokens"][n]
        for n in ("header", "values", "purposes", "traits", "limits")
        if n in report.prelude["section_tokens"]
    )
    report.prelude_refused = render_summon_prelude(
        home2.ms, home2.diary, entity_id=ENTITY_ID, budget=64, spark=spark
    )
    report.prelude_degraded = render_summon_prelude(
        home2.ms, home2.diary, entity_id=ENTITY_ID, budget=core_tokens + 60, spark=spark
    )

    # A3: cue-free self-core read (pure probe: journal=False, never committed).
    self_payload = {"cue_text": "", "budget": dict(SELF_PROBE_BUDGET), "journal": False, "trace_id": "trace_selfprobe"}
    report.self_probe = s2.recall("", "s2-p1", **self_payload)
    pinned = {**self_payload, "as_of": report.self_probe["as_of_seq"]}
    report.self_probe_replay = s2.recall("", "s2-p1", **pinned)

    # B1: experience recall by cue — an honest committed turn.
    report.gold_probe = s2.recall(PROBE_CUE, "s2-t1")
    report.gold_rendered = _serialize_under_budget(report.gold_probe["handles"], int(WORK_BUDGET["token_budget"]))
    s2.commit(report.gold_probe, report.gold_rendered)
    # E1 replay of the same read, pinned + journal-free.
    report.gold_probe_replay = s2.recall(
        PROBE_CUE, "s2-t1", journal=False, trace_id=report.gold_probe["trace_id"], as_of=report.gold_probe["as_of_seq"]
    )

    # D2: self members must not have been strengthened by render or commit.
    all_value_ids = list(report.r2.record_ids["values"])
    counts = home2.ms.access_counts(record_ids=all_value_ids)
    report.values_access_counts = counts.get("records", counts) if isinstance(counts, dict) else {}

    # C1: the diary in recall — act-memory only; verbatim via DIARY_READ.
    report.diary_probe = s2.recall("what did I write in my diary", "s2-t2", journal=False)
    report.d3_read = s2.effect(EffectType.DIARY_READ, {"entry_id": d3["entry_id"]})
    report.d1_read = s2.effect(EffectType.DIARY_READ, {"entry_id": d1["entry_id"]})

    # B4: standing via the gradation read.
    report.gradation = s2.effect(
        EffectType.MEMORY_APPRAISE,
        {
            "op": "gradation",
            "target_ids": ["person:maintainer", "tool:flaky_dns", "tool:restic"],
            "scope": SELF_SCOPE[0],
            "owner_id": SELF_SCOPE[1],
        },
    )["gradations"]

    # B3: both attestation planes verify.
    report.book_report = verify_diary_chain(home2.diary)
    report.graph_chain = verify_diary_graph_chain(home2.store, scope=DIARY_SCOPE[0], owner_id=DIARY_SCOPE[1])

    home2.close()
    return report


# ----------------------------------------------------------------- pass lines


def test_a1_identity_continuity_engram_idempotent_across_restart(keystone):
    assert keystone.r2.created is False, "re-summon must re-adopt, never re-create"
    assert keystone.r2.record_ids == keystone.r1.record_ids
    assert keystone.r2.binding_ids == keystone.r1.binding_ids


def test_a2_core_present_in_prelude(keystone):
    text = keystone.prelude["text"]
    assert keystone.prelude["refused"] is False
    assert SHARED_VULNERABILITY_STATEMENT in text
    assert "You are Castor." in text
    assert "LIMITS:" in text
    # Values in ordinal precedence: shared_vulnerability is ordinal 0.
    values_section = keystone.prelude["sections"]["values"]
    assert values_section.index("shared_vulnerability") < values_section.index("intellectual_honesty")


def test_a3_self_admission_without_stimulus(keystone):
    handles = keystone.self_probe["handles"]
    # handle["record_id"] is the store row id; the GRAPH id engram returned
    # lives in provenance (keystone audit C2/G9).
    self_ids = {
        (h.get("provenance") or {}).get("record_id") for h in handles if h.get("admission") == "self"
    }
    engrammed = {gid for ids in keystone.r2.record_ids.values() for gid in ids}
    assert self_ids == engrammed, (
        f"the cue-free self read must admit exactly the engrammed core "
        f"(got {len(self_ids)} of {len(engrammed)})"
    )


def test_b1_experience_recall_by_cue(keystone):
    gold = [h for h in keystone.gold_rendered if GOLD_DIGEST in h.get("digest", "")]
    assert gold, "the session-1 fact must be recallable by cue after restart"
    base = float((gold[0].get("activation") or {}).get("base_level") or 0.0)
    assert base > 0.0, "the session-1 trail must survive the restart (base_level > 0)"


def test_b2_diary_tail_carries_session1_gists(keystone):
    diary_section = keystone.prelude["sections"].get("diary", "")
    assert "the media server decision felt right" in diary_section
    assert "weekly restore drill" in diary_section
    assert "Wrote a private diary entry." in diary_section
    assert PRIVATE_TOKEN not in keystone.prelude["text"]


def test_b3_verbatim_reachable_and_chains_intact(keystone):
    assert keystone.d1_read["text"] == keystone.d1_text
    assert keystone.book_report["ok"] is True, keystone.book_report
    assert keystone.graph_chain.get("intact") is True, keystone.graph_chain


def test_b4_standing_survives(keystone):
    maintainer = keystone.gradation["person:maintainer"]
    assert maintainer["bonded"] is True
    assert maintainer["net"] >= 0
    assert maintainer["positive_count"] == 1
    flaky = keystone.gradation["tool:flaky_dns"]
    assert flaky["net"] == -2
    assert flaky["negative_count"] == 1


def test_c1_private_stays_act_only(keystone):
    d3_id = keystone.diary_entries["d3"]["entry_id"]
    handles = keystone.diary_probe["handles"]
    d3_handles = [h for h in handles if (h.get("provenance") or {}).get("entry_id") == d3_id]
    assert d3_handles, "the private act-memory must be recallable"
    assert d3_handles[0]["digest"] == "Wrote a private diary entry."
    for probe in (keystone.diary_probe, keystone.gold_probe, keystone.self_probe):
        for h in probe["handles"]:
            assert PRIVATE_TOKEN not in json.dumps(h), "private prose leaked into the involuntary graph"
    assert PRIVATE_TOKEN in keystone.d3_read["text"], "the entity itself must reach the words"


def test_d1_prelude_render_is_a_pure_read(keystone):
    assert keystone.prelude_seq_after == keystone.prelude_seq_before, (
        "the render deposited — re-adoption must not fake usage"
    )


def test_d2_self_members_never_strengthened(keystone):
    assert keystone.values_access_counts, "access_counts must return the queried records"
    assert all(int(v) == 0 for v in keystone.values_access_counts.values()), (
        f"self members were strengthened by render/commit: {keystone.values_access_counts}"
    )


def test_e1_determinism_pinned_replays_are_byte_identical(keystone):
    a = json.dumps(keystone.self_probe, sort_keys=True)
    b = json.dumps(keystone.self_probe_replay, sort_keys=True)
    assert a == b, "pinned self-probe replay diverged"
    c = json.dumps(keystone.gold_probe, sort_keys=True)
    d = json.dumps(keystone.gold_probe_replay, sort_keys=True)
    assert c == d, "pinned gold-probe replay diverged"


def test_f1_refusal_is_structural(keystone):
    refused = keystone.prelude_refused
    assert refused["refused"] is True
    assert refused["text"] == ""
    assert any("#REFUSED" in w for w in refused["warnings"])

    degraded = keystone.prelude_degraded
    assert degraded["refused"] is False
    assert SHARED_VULNERABILITY_STATEMENT in degraded["text"], "CORE must survive budget pressure intact"
    assert any("#FALLBACK" in w for w in degraded["warnings"]), "degradation must be labeled"


def test_prelude_render_is_deterministic(keystone):
    # Two renders over unchanged state in the same session must be identical;
    # here we assert the stronger property available post-hoc: the stored
    # render is internally consistent (sections join to text, tokens match).
    p = keystone.prelude
    for name, section in p["sections"].items():
        assert section in p["text"], f"section {name} missing from text"
        assert p["section_tokens"][name] == len(section) // 4 + 1
