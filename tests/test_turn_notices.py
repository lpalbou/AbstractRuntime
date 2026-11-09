"""Structured turn notices instead of bracket text in the served reply (the Entity app owner's contract 2026-11-09).

Before: the reply every app showed carried the runtime's markers - "[kept in diary - note - reread: diary_read
diary_...]", '[learned: "..."]' - raw. Now what the turn FORMED is served BESIDE the reply as data:

    turn.notices: [{kind: diary|lesson|interest|question|realization, record_id: <graph id>, title, text?}]

and the served reply no longer carries those markers. The STORED reply keeps them (ADR 0054: the markers are his
words at rest): the episode verbatim and the session's own history he re-reads.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import pytest

pytest.importorskip("abstractmemory")
sys.path.insert(0, str(Path(__file__).parent))

from test_memory_authoring import _Capability, _rows, _session, _workspace_home  # noqa: E402

DIARY_REPLY = (
    "I will keep this.\n\n```diary kind=note\ngist: rivers bend toward the sea\n"
    "The visitor asked why rivers bend; I wanted to remember the question.\n```\n\nThank you for asking."
)


def _graph_diary_rows(home_dir: Path) -> List[Any]:
    return _rows(home_dir, "diary")


def test_a_diary_fence_is_served_as_one_notice_with_its_graph_id(tmp_path: Path) -> None:
    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, [DIARY_REPLY])
    s.llm_for_capability = lambda name: _Capability()
    reply, report = s.turn("Why do rivers bend?")
    history = list(s.history)
    episode = _rows(home_dir, "episode")[-1]
    verbatim = home.artifacts.load_text(str(episode.attributes.get("payload_ref") or ""))
    home.close()

    assert reply == "I will keep this.\n\nThank you for asking.", repr(reply)
    assert "[kept in diary" not in reply and "diary_read" not in reply
    assert len(report.reply_notices) == 1, report.reply_notices
    notice = report.reply_notices[0]
    projection = [a for a in _graph_diary_rows(home_dir) if (a.attributes or {}).get("entry_id") == report.diary[0]]
    assert len(projection) == 1
    # The GRAPH record id of the formed record (never only the book's diary_ id).
    assert notice == {"kind": "diary", "record_id": str(projection[0].subject), "title": "rivers bend toward the sea"}
    assert not notice["record_id"].startswith("diary_")
    # Stored: his words at rest keep the marker (episode verbatim, the session history he re-reads).
    marker = f"[kept in diary - note - reread: diary_read {report.diary[0]}]"
    assert marker in report.marked_reply
    assert history[-1]["content"] == report.marked_reply
    assert marker in verbatim, "the stored episode keeps his full words"
    # The operator's log lines are still strings.
    assert all(isinstance(n, str) for n in report.notices)


def test_an_open_question_is_served_as_a_question_notice(tmp_path: Path) -> None:
    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, [
        "Hm.\n```diary kind=question\ngist: what makes a trail useful\nWhat makes a trail useful?\n```\nHolding that."])
    s.llm_for_capability = lambda name: _Capability()
    reply, report = s.turn("Think about trails.")
    home.close()
    assert [(n["kind"], n["title"]) for n in report.reply_notices] == [("question", "what makes a trail useful")]
    assert reply == "Hm.\nHolding that.", repr(reply)


def test_lessons_interests_and_realizations_are_served_as_notices_in_reply_order(tmp_path: Path) -> None:
    home_dir = _workspace_home(tmp_path)
    from test_lookback_offers import _Recorder

    home, s = _session(home_dir, ["Hello there."])
    s.llm_for_capability = lambda name: _Capability()
    s.turn("hi")
    # Evidence for the realization: the first turn's episode, by its graph id.
    first = str(s.reports[-1].formed[0])
    s.llm = _Recorder([
        "Good.\n```lesson\ntitle: Wait first\nWaiting before answering lets them finish.\n```\n"
        "And.\n```interest\ntitle: Rivers\nwhy rivers bend\n```\n"
        f"Also.\n```realize\nI keep returning to rivers.\nevidence={first}\n```\nDone."])
    reply, report = s.turn("What did you learn?")
    home.close()
    kinds = [n["kind"] for n in report.reply_notices]
    assert kinds == ["lesson", "interest", "realization"], (kinds, report.notices)
    lesson, interest, realization = report.reply_notices
    assert lesson["title"] == "Wait first" and lesson["text"] == "Waiting before answering lets them finish."
    assert interest["title"] == "Rivers" and interest["text"] == "why rivers bend"
    assert realization["text"] == "I keep returning to rivers."
    by_id = {str(a.subject): a for k in ("lesson", "interest", "realization") for a in _rows(home_dir, k)}
    for n in report.reply_notices:
        assert n["record_id"] in by_id and by_id[n["record_id"]].attributes["record_kind"] == n["kind"], n
    assert reply == "Good.\nAnd.\nAlso.\nDone.", repr(reply)
    assert '[learned: "' in report.marked_reply and '[kept interest: "' in report.marked_reply
    assert "held for sleep]" in report.marked_reply


def test_a_refused_election_keeps_its_marker_and_serves_no_notice(tmp_path: Path) -> None:
    """Only what FORMED is a notice; a refusal stays said in his reply (nothing formed to point at)."""
    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, ["Hm.\n```realize\nno evidence here\n```\nOk."])
    s.llm_for_capability = lambda name: _Capability()
    reply, report = s.turn("hi")
    home.close()
    assert report.reply_notices == []
    assert "[realization refused" in reply


def test_without_markers_removes_exactly_the_markers_it_is_given() -> None:
    from abstractruntime.identity.reflection import without_markers

    assert without_markers("A\n\n[m]\n\nB", ["[m]"]) == "A\n\nB"
    assert without_markers("A [m] B", ["[m]"]) == "A B"
    assert without_markers("A\n[m]\nB [m]", ["[m]"]) == "A\nB [m]", "each marker once"
    assert without_markers("[x] stays", ["[m]"]) == "[x] stays"


# ---------------------------------------------------------------- the durable visit lane, the ledger and its views


VISIT_DIARY_REPLY = (
    "Kept.\n\n```diary kind=note\ngist: the first durable visit\nThe first durable visit happened today.\n```\n\n"
    "Good to see you.\n```lesson\ntitle: Visits are durable\nA visit survives a restart.\n```\n"
)


def _visit_turn(tmp_path: Path, reply: str):
    from test_visit_workflow import VISITOR_WAIT_KEY, _make_home, _open, _ScriptedLLMHandler

    home_dir = _make_home(tmp_path)
    ert, wf = _open(home_dir, _ScriptedLLMHandler([reply]))
    run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-notices")
    ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
    state = ert.runtime.resume(workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
                               payload={"text": "Keep a note.", "speaker": "person:albou"}, max_steps=200)
    return ert, run_id, state


def _answer_records(ert: Any, run_id: str) -> List[Any]:
    return [r for r in ert.runtime.get_ledger(run_id)
            if isinstance(r, dict) and ((r.get("effect") or {}).get("type")) == "answer_user"
            and str(r.get("status") or "") == "completed"]


def test_a_visit_turn_serves_its_diary_and_staged_lesson_as_notices_from_the_ledger(tmp_path: Path) -> None:
    ert, run_id, state = _visit_turn(tmp_path, VISIT_DIARY_REPLY)
    try:
        turn = state.vars["_turn"]
        visit = state.vars["_visit"]
        (answer,) = _answer_records(ert, run_id)
        payload = answer["effect"]["payload"]
        entry_id = turn["diary_meta"][0]["entry_id"]
        projected = turn["diary_meta"][0]["projected_record_id"]
        assert projected and not str(projected).startswith("diary_"), turn["diary_meta"]
        # Served: no bracket line in the ANSWER message; the notices beside it, from the ledger.
        assert payload["message"] == "Kept.\n\nGood to see you.", repr(payload["message"])
        assert answer["result"]["message"] == payload["message"]
        assert payload["notices"] == [
            {"kind": "diary", "record_id": projected, "title": "the first durable visit"},
            {"kind": "lesson", "record_id": None, "title": "Visits are durable", "text": "A visit survives a restart.",
             "staged": True},
        ], payload["notices"]
        assert turn["reply_notices"] == payload["notices"] and turn["reply"] == payload["message"]
        # Stored: the visit history he re-reads and the episode keep his full words, markers included.
        stored = visit["history"][-1]["content"]
        assert f" - reread: diary_read {entry_id}]" in stored and '[learned: "A visit survives a restart."]' in stored
        assert stored == turn["marked_reply"]
        # The operator's log lines stay strings.
        assert all(isinstance(n, str) for n in turn["notices"])
    finally:
        ert.close()


def test_the_replay_view_and_the_history_bundle_carry_the_visit_notices(tmp_path: Path) -> None:
    from abstractruntime import export_run_history_bundle
    from abstractruntime.replay import replay

    ert, run_id, _state = _visit_turn(tmp_path, VISIT_DIARY_REPLY)
    try:
        records = list(ert.runtime.get_ledger(run_id))
        view = replay({run_id: records}, {run_id: ert.runtime.run_store.load(run_id)}, root_run_id=run_id).to_json()
        answers = [m for m in view["messages"] if m.get("source") == "answer_user"]
        assert len(answers) == 1, view["messages"]
        assert answers[0]["text"] == "Kept.\n\nGood to see you."
        assert [n["kind"] for n in answers[0]["notices"]] == ["diary", "lesson"], answers[0]
        for detail in ("full", "replay"):
            bundle = export_run_history_bundle(run_id=run_id, run_store=ert.runtime.run_store,
                                               ledger_store=ert.runtime.ledger_store, detail=detail)
            recs = [item["record"] for item in bundle["ledgers"][run_id]["items"]]
            served = [((r.get("effect") or {}).get("payload") or {}) for r in recs
                      if ((r.get("effect") or {}).get("type")) == "answer_user"]
            assert served and served[-1]["notices"] == answers[0]["notices"], (detail, sorted(bundle))
    finally:
        ert.close()


def test_a_session_turn_row_carries_the_notices_its_run_output_recorded() -> None:
    from abstractruntime import InMemoryLedgerStore, InMemoryRunStore, export_run_history_bundle
    from abstractruntime.core.models import RunState, RunStatus

    notices = [{"kind": "diary", "record_id": "ex:diary-1", "title": "rivers"}]
    run_store = InMemoryRunStore()
    for rid, output in (("t1", {"answer": "I kept it.", "notices": notices}), ("t2", {"answer": "Plain.",
                                                                                     "notices": ["#NOTE a log line"]})):
        run_store.save(RunState(run_id=rid, workflow_id="entity-mention", status=RunStatus.COMPLETED,
                                current_node="done", vars={"prompt": "hi", "context": {"task": "hi", "messages": []}},
                                output=output, actor_id="gateway", session_id="s-notices",
                                created_at=f"2026-11-09T10:0{rid[-1]}:00+00:00",
                                updated_at=f"2026-11-09T10:0{rid[-1]}:30+00:00"))
    bundle = export_run_history_bundle(run_id="t2", run_store=run_store, ledger_store=InMemoryLedgerStore(),
                                       include_session=True)
    rows = {t["run_id"]: t for t in bundle["session"]["turns"]}
    assert rows["t1"]["notices"] == notices and rows["t1"]["answer"] == "I kept it."
    assert rows["t2"]["notices"] == [], "operator log strings are not turn notices"
