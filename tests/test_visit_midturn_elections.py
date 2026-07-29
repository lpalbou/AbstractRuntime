"""Mid-turn identity elections on the durable visit lane (Veya deep check).

The operator-ordered deep check found a well-formed ```realize fence on
turn 7 of a real visit that formed NOTHING - no record, no notice, no
error - because this lane parsed election fences only in the CLOSE
reflection. The chat lane parses them mid-turn. Now the turn node parses
all fence kinds, resolves realization evidence against what the entity
saw THAT turn, stages everything on the visit (run vars are durable, so
staging survives crashes), and the close stages form them. The turn's
notices say so; nothing is silent.

Also pinned here: the speaker-attribution fix (deep check P2) - a
payload-claimed speaker name that matches no verified participant is
recorded as a claim in attributes, never engraved into digest prose.
"""

from __future__ import annotations

import copy as _copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.core.models import Effect, EffectType, RunStatus  # noqa: E402
from abstractruntime.core.runtime import EffectOutcome  # noqa: E402
from abstractruntime.identity.entity_runtime import open_entity_runtime  # noqa: E402
from abstractruntime.identity.visit_workflow import (  # noqa: E402
    VISITOR_WAIT_KEY,
    build_visit_workflow,
)


def _make_home(tmp_path: Path, slug: str = "electra") -> Path:
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
    store, journal = SQLiteTripleStore(db), SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir


class _ScriptedHandler:
    """Scripted replies; a callable reply builds itself from the run (so a
    fence can cite a record id the test cannot know in advance)."""

    def __init__(self, replies: List[Any]) -> None:
        self.replies = list(replies)

    def __call__(self, run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
        nxt = self.replies.pop(0) if self.replies else "…"
        content = nxt(run) if callable(nxt) else str(nxt)
        return EffectOutcome.completed({"content": content})


def _drive_visit(home_dir: Path, replies: List[str], turns: List[Dict[str, Any]]):
    """Open a visit, run the given turns, close it. Returns (ert, run_id)."""
    ert = open_entity_runtime(
        home_dir, extra_handlers={EffectType.LLM_CALL: _ScriptedHandler(replies)}
    )
    wf = build_visit_workflow(
        ert.home,
        participants=["person:admin"],
        idle_seconds=3600,
        model_info={"provider": "test", "model": "scripted"},
        visit_id="visit-elections",
    )
    run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-elections")
    state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
    assert state.status == RunStatus.WAITING
    for payload in turns:
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload=payload, max_steps=200,
        )
    return ert, wf, run_id, state


def _abstract_rows(home: Any) -> List[Any]:
    """Every formed record's digest is the object of its dcterms:abstract
    triple (the pinned record-shape rule); attributes ride the assertion."""
    from abstractmemory import TripleQuery

    return list(home.ms.store.query(TripleQuery(predicate="dcterms:abstract", limit=1000)))


def test_midturn_realize_and_interest_form_at_close(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    def reply_with_fences(run: Any) -> str:
        # Cite a record from the session sheet - turn 1's formed episode,
        # something the entity genuinely has in front of it this visit.
        # (The virgin fixture home's recall serves nothing, so the sheet is
        # the evidence space here; live homes also resolve against the
        # turn's recalled records.) The evidence grammar is evidence=#tag.
        sheet = list((run.vars.get("_visit") or {}).get("sheet") or [])
        assert sheet, "the session sheet is empty - turn 1 formed nothing"
        rid = str(sheet[0][0])
        return (
            "I see it now.\n"
            "```realize\nI keep returning to rivers when I explain change.\n"
            f"evidence={rid}\n```\n"
            "```interest\nhow rivers carve their own maps\n```\n"
            "That is my answer."
        )

    ert, wf, run_id, state = _drive_visit(
        home_dir,
        replies=["A plain first reply.", reply_with_fences, "Goodbye reflection: a good visit."],
        turns=[
            {"text": "hello", "speaker": "person:admin"},
            {"text": "What did you notice about yourself?", "speaker": "person:admin"},
        ],
    )
    try:
        # Mid-turn: staged on the visit, noticed loudly, fences stripped
        # from the delivered reply.
        run = ert.run_store_load(run_id) if hasattr(ert, "run_store_load") else None
        vars_raw = json.dumps(
            (ert.runtime.get_state(run_id).vars if run is None else run.vars), default=str
        )
        assert "pending_elections" in vars_raw
        assert "rivers carve" in vars_raw  # interest staged
        turn_state = ert.runtime.get_state(run_id)
        answered = str(((turn_state.vars.get("_turn") or {}).get("marked_reply")) or "")
        assert "```realize" not in answered and "```interest" not in answered
        notices = list((turn_state.vars.get("_turn") or {}).get("notices") or [])
        assert any("form at visit close" in n for n in notices)

        # Close: the staged elections FORM.
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"kind": "close"}, max_steps=400,
        )
        assert state.status == RunStatus.COMPLETED
        digests = [str(a.object) for a in _abstract_rows(ert.home)]
        assert any("rivers when I explain change" in d for d in digests), (
            "the mid-turn realization never formed"
        )
        assert any("rivers carve their own maps" in d for d in digests), (
            "the mid-turn interest never formed"
        )
    finally:
        ert.close()


def test_midturn_realize_without_evidence_refuses_loudly(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path, slug="electrb")
    reply = (
        "A thought.\n```realize\nSomething about nothing.\nevidence=#zzzzzz\n```\nDone."
    )
    ert, wf, run_id, state = _drive_visit(
        home_dir,
        replies=[reply, "Goodbye."],
        turns=[{"text": "hello", "speaker": "person:admin"}],
    )
    try:
        turn_state = ert.runtime.get_state(run_id)
        notices = list((turn_state.vars.get("_turn") or {}).get("notices") or [])
        assert any("realization refused" in n for n in notices), (
            "an unresolvable realization must refuse loudly, never vanish"
        )
    finally:
        ert.close()


def test_claimed_speaker_is_recorded_as_claim_not_engraved(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path, slug="electrc")
    ert, wf, run_id, state = _drive_visit(
        home_dir,
        replies=["Nice to meet you.", "Goodbye."],
        turns=[{"text": "hi there", "speaker": "fable5-guest"}],
    )
    try:
        rows = _abstract_rows(ert.home)
        assert rows, "the turn formed no records"
        digest_blob = " | ".join(str(a.object) for a in rows)
        # The verified participant is the prose speaker; the claimed label
        # is recorded as a claim on the assertion's attributes.
        assert "fable5-guest:" not in digest_blob, "claimed name engraved into prose"
        attr_blob = json.dumps([a.attributes for a in rows], default=str)
        assert "speaker_label_claimed" in attr_blob and "fable5-guest" in attr_blob
    finally:
        ert.close()
