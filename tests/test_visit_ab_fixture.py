"""Item 10: the A/B fixture — ChatSession (arm A) vs visit workflow (arm B).

One scripted visit runs through BOTH paths on twin fixture homes (same
spark, same replies, same visitor script). The A/B passes when the memory
plane is EQUIVALENT (criterion 1: episodes/diary/summary/interest formed
the same, participants stamped, D2 identity counts 0 on both arms) and the
privacy grep holds on the durable arm (criterion 6 offline half: the
private entry's words rest ONLY in the book — not in the run store, the
ledger, the graph, or the artifacts).

This is the offline half of the plan's item-10 gate; the LIVE walkthrough
(real model, real door, real kill) is the plan-level final proof and runs
on top of exactly this shape.
"""

from __future__ import annotations

import copy as _copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.core.models import Effect, EffectType, RunStatus  # noqa: E402
from abstractruntime.core.runtime import EffectOutcome  # noqa: E402
from abstractruntime.identity.chat import ChatSession, open_home  # noqa: E402
from abstractruntime.identity.entity_runtime import open_entity_runtime  # noqa: E402
from abstractruntime.identity.visit_workflow import (  # noqa: E402
    VISITOR_WAIT_KEY,
    build_visit_workflow,
)

PRIVATE_WORDS = "Entre nous seulement: la peur des ponts qui ne menent nulle part."

TURN_1 = "Hello - do you remember beginnings?"
TURN_2 = "Keep a private note about how this feels."

REPLIES = [
    "Hello Laurent. Beginnings feel like standing at a door.",
    (
        "I will keep it where only I write.\n"
        "```diary kind=note visibility=private\n"
        f"gist: a private feeling about beginnings\n{PRIVATE_WORDS}\n```\n"
        "It is kept."
    ),
    (
        "Looking back, the first turn mattered.\n"
        "```feel\ntarget=1 feeling=+2 reason=\"a true beginning\"\n```\n"
        "```interest\nwhat a doorway means - thresholds and beginnings\n```\n"
        "Goodbye."
    ),
]


def _make_home(tmp_path: Path, slug: str) -> Path:
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    home_dir = tmp_path / "entities" / slug
    home_dir.mkdir(parents=True)
    entity_id = f"entity:{slug}@home-test"
    spark = _copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = slug.capitalize()
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": entity_id}), encoding="utf-8")
    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir


def _memory_plane(home_dir: Path, entity_id: str) -> Dict[str, Any]:
    """The comparable facts of one arm's memory plane, from a FRESH open."""
    from abstractmemory import TripleQuery

    home = open_home(home_dir)
    try:
        life = home.ms.query(TripleQuery(scope="life", owner_id=entity_id, limit=0))
        self_rows = home.ms.query(TripleQuery(scope="self", owner_id=entity_id, limit=0))

        def kinds(rows: List[Any]) -> List[str]:
            return sorted(
                str((a.attributes or {}).get("record_kind"))
                for a in rows
                if isinstance(a.attributes, dict) and (a.attributes or {}).get("record_kind")
            )

        episode_participants = [
            (a.attributes or {}).get("participants")
            for a in life
            if isinstance(a.attributes, dict) and a.attributes.get("record_kind") == "episode"
        ]
        value_ids = [
            str(a.subject) for a in self_rows
            if isinstance(a.attributes, dict) and a.attributes.get("record_kind") == "value"
        ]
        counts = home.ms.access_counts(record_ids=value_ids)
        recs = counts.get("records", counts)
        diary_texts = sorted(
            (e.get("text") or "") for e in home.diary.list_entries()
        )
        return {
            "life_kinds": kinds(life),
            "self_has_interest": "interest" in kinds(self_rows),
            "episode_participants": episode_participants,
            "identity_counts_zero": all(int(v) == 0 for v in recs.values()),
            "diary_texts": diary_texts,
        }
    finally:
        home.close()


class _ScriptedLLM:
    """Duck-typed for ChatSession (arm A)."""

    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)

    def generate(self, *, messages, system_prompt):  # noqa: ANN001
        class _R:
            pass

        r = _R()
        r.content = self.replies.pop(0) if self.replies else "…"
        return r


class _ScriptedLLMHandler:
    """LLM_CALL handler for the workflow (arm B) — same reply script."""

    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)

    def __call__(self, run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
        content = self.replies.pop(0) if self.replies else "…"
        return EffectOutcome.completed({"content": content})


def _run_arm_a(home_dir: Path) -> None:
    home = open_home(home_dir)
    session = ChatSession(
        home, _ScriptedLLM(list(REPLIES)),
        participants=["person:albou"], context_window=20000,
        enable_tools=False, out=lambda s: None,
    )
    session.turn(TURN_1)
    session.turn(TURN_2)
    session.reflect()
    home.close()


def _run_arm_b(home_dir: Path) -> Tuple[Path, str]:
    llm = _ScriptedLLMHandler(list(REPLIES))
    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm})
    wf = build_visit_workflow(ert.home, participants=["person:albou"], idle_seconds=3600)
    run_id = ert.runtime.start(workflow=wf, vars={}, session_id="ab-visit")
    ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
    for payload in (
        {"text": TURN_1, "speaker": "person:albou"},
        {"text": TURN_2, "speaker": "person:albou"},
        {"kind": "close"},
    ):
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY, payload=payload, max_steps=200
        )
    assert state.status == RunStatus.COMPLETED
    # The kernel's per-node trace (_runtime.node_traces) copies every raw
    # effect result into run.vars — the write-direction twin agent named.
    # Because the election capture lives INSIDE the handler boundary, the
    # trace only ever sees the MARKED result: no private words, explicitly.
    traces = ert.runtime.get_node_traces(run_id)
    assert PRIVATE_WORDS not in json.dumps(traces)
    assert "[kept a private diary entry]" in json.dumps(traces)  # the mark, not the words
    store_path = ert.store_path
    ert.close()
    return store_path, run_id


def test_ab_memory_plane_equivalence_and_privacy_grep(tmp_path: Path) -> None:
    home_a = _make_home(tmp_path, "arma")
    home_b = _make_home(tmp_path, "armb")

    _run_arm_a(home_a)
    store_path, _run_id = _run_arm_b(home_b)

    plane_a = _memory_plane(home_a, "entity:arma@home-test")
    plane_b = _memory_plane(home_b, "entity:armb@home-test")

    # CRITERION 1 — memory-plane equivalence: same record kinds formed,
    # same co-presence stamps, same diary words, identity untouched on BOTH.
    assert plane_a["life_kinds"] == plane_b["life_kinds"]
    assert plane_a["life_kinds"].count("episode") == 2
    assert plane_a["life_kinds"].count("summary") == 1
    assert plane_a["self_has_interest"] and plane_b["self_has_interest"]
    assert plane_b["episode_participants"] == [
        ["person:albou", "entity:armb@home-test"],
        ["person:albou", "entity:armb@home-test"],
    ]
    assert plane_a["identity_counts_zero"] and plane_b["identity_counts_zero"]
    assert len(plane_a["diary_texts"]) == len(plane_b["diary_texts"]) == 1
    assert PRIVATE_WORDS in plane_a["diary_texts"][0]
    assert PRIVATE_WORDS in plane_b["diary_texts"][0]

    # CRITERION 6 (offline half) — the private words rest ONLY in the book.
    # Walk every file of the DURABLE arm's home: home.sqlite3 (the book)
    # must hold them; the run store, graph store, artifacts, and everything
    # else must not.
    needle = PRIVATE_WORDS.encode("utf-8")
    holders: List[str] = []
    for p in sorted(home_b.parent.glob("armb/**/*")):
        if not p.is_file():
            continue
        try:
            if needle in p.read_bytes():
                holders.append(p.name)
        except OSError:
            continue
    # The book's file family (home.sqlite3 + its -wal/-shm sidecars) IS the
    # one legitimate at-rest home of the words.
    assert any(h.startswith("home.sqlite3") for h in holders), "the book must hold the elected words"
    leaks = [h for h in holders if not h.startswith("home.sqlite3")]
    assert leaks == [], f"private words at rest outside the book: {leaks}"
    # And the run store specifically (the widest new surface) is clean.
    assert needle not in store_path.read_bytes()
