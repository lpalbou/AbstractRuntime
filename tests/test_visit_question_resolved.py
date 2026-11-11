"""A question the entity asked is marked resolved when the visit answers it (operator 2026-11-11 02:00: an entity
asked him a question in a visit, he answered, nothing marked it resolved, no badge, the question stayed under OPEN
QUESTIONS: "investigate and fix"; root backlog 1206).

Memory already resolves a question when a later diary entry carries `resolves=<its entry id>` (the DIARY_WRITE
handler checks the claim: `resolves_status`). The visit lane never asked for it: its look-back did not list the open
questions nor invite the resolution, and its diary metadata did not carry the verdict, so no notice could say it.

Pinned through the real durable visit workflow on a real home:

- the look-back lists the open questions the visit asked or touched (entry id + title) and offers a ```resolve
  election; a scripted look-back that resolves the question takes it off memory's open questions, links the entry to
  the moment that answered it, and its capture carries the `question_resolved` notice (the question's graph id and
  title); a look-back without a resolution leaves the question open;
- a turn whose diary entry resolves the question serves `question_resolved` beside the reply, and the look-back then
  names the resolution (the resolution->lesson bridge) instead of listing the question as open;
- the act-only capture's metadata carries `resolves` + `resolves_status` (a refused claim too, with no notice);
- the chat lane's look-back prompt is unchanged (no open-question list, no ```resolve offer).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.core.models import Effect, EffectType, RunStatus  # noqa: E402
from abstractruntime.core.runtime import EffectOutcome  # noqa: E402
from abstractruntime.identity.chat import question_resolved_notice  # noqa: E402
from abstractruntime.identity.entity_runtime import open_entity_runtime  # noqa: E402
from abstractruntime.identity.lifecycle_fence import LifecycleFence  # noqa: E402
from abstractruntime.identity.reflection import build_reflection_prompt, parse_resolve_blocks  # noqa: E402
from abstractruntime.identity.visit_workflow import VISITOR_WAIT_KEY, build_visit_workflow  # noqa: E402
from test_visit_workflow import _make_home  # noqa: E402

VISITOR = "person:albou"
QUESTION_TITLE = "What makes a trail worth following"
ASKS = (
    f"```diary kind=question\ngist: {QUESTION_TITLE}\nWhat makes a trail worth following, for a mind like mine?\n```\n"
    "Can I ask you something: what makes a trail worth following?"
)
ANSWER = "Time order: a trail is worth following when each step was written after the one before it."

Reply = Union[str, Callable[[Dict[str, Any]], str]]


class _Scripted:
    """The provider stub: queued replies, each a string or a function of the call's payload (the scripted model
    names the question's entry id, which only exists once the visit wrote it)."""

    def __init__(self, replies: List[Reply]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
        payload = dict(effect.payload or {})
        self.calls.append(payload)
        reply = self.replies.pop(0) if self.replies else "…"
        return EffectOutcome.completed({"content": reply(payload) if callable(reply) else reply})


def _open(home_dir: Path, llm: _Scripted):
    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm}, fence=LifecycleFence.admit(home_dir))
    wf = build_visit_workflow(ert.home, participants=[VISITOR], idle_seconds=3600,
                              model_info={"provider": "test", "model": "scripted"}, visit_id="visit-q1")
    return ert, wf


def _say(ert: Any, wf: Any, run_id: str, text: str) -> Any:
    state = ert.runtime.resume(workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
                               payload={"text": text, "speaker": VISITOR}, max_steps=100)
    assert state.status == RunStatus.WAITING
    return state


def _close(ert: Any, wf: Any, run_id: str) -> Any:
    state = ert.runtime.resume(workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
                               payload={"kind": "close"}, max_steps=300)
    assert state.status == RunStatus.COMPLETED
    return state


def _question_entry(ert: Any) -> Dict[str, Any]:
    questions = [e for e in ert.home.diary.list_entries() if e.get("kind") == "question"]
    assert len(questions) == 1, questions
    return questions[0]


def _projection(ert: Any, entry_id: str) -> Any:
    from abstractmemory import TripleQuery

    rows = [a for a in ert.home.ms.store.query(TripleQuery(predicate="dcterms:abstract", scope="diary",
                                                           owner_id=ert.entity_id, limit=0))
            if isinstance(a.attributes, dict) and a.attributes.get("entry_id") == entry_id]
    assert len(rows) == 1, f"one projection of {entry_id}"
    return rows[0]


def _open_question_ids(ert: Any) -> List[str]:
    from abstractmemory.diary import open_questions

    return [str(a.subject) for a in open_questions(ert.home.ms.store, scope="diary", owner_id=ert.entity_id,
                                                    journal=ert.home.ms.journal, limit=0)]


def _lookback_prompt(llm: _Scripted) -> str:
    calls = [c for c in llm.calls if c.get("turn_id") == "t-reflect"]
    assert len(calls) == 1, "one look-back model call"
    return str(calls[0]["messages"][-1]["content"])


def _lookback_entries(ert: Any, run_id: str) -> List[Dict[str, Any]]:
    """The look-back's captured diary entries, as its LLM_CALL ledger row records them (word-free metadata)."""
    rows = [r for r in ert.runtime.get_ledger(run_id)
            if isinstance(r, dict) and str((r.get("effect") or {}).get("type") or "") == "llm_call"
            and ((r.get("effect") or {}).get("payload") or {}).get("turn_id") == "t-reflect"
            and str(r.get("status") or "") == "completed"]
    assert len(rows) == 1
    return list((rows[0].get("result") or {}).get("diary_entries") or [])


def _answers(ert: Any, run_id: str) -> List[Dict[str, Any]]:
    return [((r.get("effect") or {}).get("payload") or {}) for r in ert.runtime.get_ledger(run_id)
            if isinstance(r, dict) and str((r.get("effect") or {}).get("type") or "") == "answer_user"
            and str(r.get("status") or "") == "completed"]


def _resolve_reply(ert: Any, *, evidence: str = "3") -> Callable[[Dict[str, Any]], str]:
    def reply(_payload: Dict[str, Any]) -> str:
        qid = _question_entry(ert)["entry_id"]
        return (
            "Looking back, Laurent answered what I asked.\n"
            f"```resolve\nresolves={qid}\nevidence={evidence}\n"
            "A trail is worth following when its steps are in time order - Laurent told me so.\n```\nGoodbye."
        )

    return reply


def test_lookback_lists_the_question_and_its_resolve_election_resolves_it(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path, slug="asker")
    llm = _Scripted([ASKS, "Thank you - that is the answer I was missing."])
    ert, wf = _open(home_dir, llm)
    try:
        llm.replies.append(_resolve_reply(ert))
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-q1")
        ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
        _say(ert, wf, run_id, "Ask me something.")
        q = _question_entry(ert)
        qgid = str(_projection(ert, q["entry_id"]).subject)
        assert qgid in _open_question_ids(ert), "the question the entity asked stands open"
        _say(ert, wf, run_id, ANSWER)
        state = _close(ert, wf, run_id)

        # The look-back lists the question it asked (entry id + title) and offers the resolution.
        prompt = _lookback_prompt(llm)
        assert q["entry_id"] in prompt and QUESTION_TITLE in prompt
        assert "```resolve" in prompt and "resolves=" in prompt and "evidence=" in prompt

        # Memory took it off the open questions: the resolving entry carries resolves=<its entry id>.
        assert qgid not in _open_question_ids(ert)
        resolving = [e for e in ert.home.diary.list_entries() if e.get("resolves") == q["entry_id"]]
        assert len(resolving) == 1 and resolving[0]["kind"] == "note"
        assert "time order" in resolving[0]["text"]

        # The capture's metadata carries the verdict and the notice the close serves.
        metas = [m for m in _lookback_entries(ert, run_id) if m.get("resolves")]
        assert len(metas) == 1
        meta = metas[0]
        assert (meta["resolves"], meta["resolves_status"]) == (q["entry_id"], "resolved_open_question")
        assert question_resolved_notice(meta) == {
            "kind": "question_resolved", "record_id": qgid, "title": QUESTION_TITLE,
        }

        # evidence=3 linked the entry to the moment that answered it (the third line of the visit's sheet: the
        # episode of the turn in which the person answered).
        sheet = list(state.vars["_visit"]["sheet"])
        answered_in = sheet[2][0]
        assert answered_in.startswith("ex:") and ANSWER[:20] in sheet[2][1]
        edges = [str(a.object) for a in ert.home.ms.store.query(__import__("abstractmemory").TripleQuery(
            subject=str(meta["projected_record_id"]), predicate="written_amid", limit=0))]
        assert answered_in in edges

        # The stored look-back says it in his marker; the operator sees nothing refused.
        assert f"resolves your open question {q['entry_id']}" in str(state.vars["_reflect"]["marked_reply"])
        assert not [n for n in state.output["reflection_notices"] if "resolve" in n.lower()]
    finally:
        ert.close()


def test_lookback_without_a_resolution_leaves_the_question_open(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path, slug="holder")
    llm = _Scripted([ASKS, "Thank you.", "Looking back, a good exchange. Goodbye."])
    ert, wf = _open(home_dir, llm)
    try:
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-q2")
        ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
        _say(ert, wf, run_id, "Ask me something.")
        _say(ert, wf, run_id, ANSWER)
        _close(ert, wf, run_id)
        q = _question_entry(ert)
        prompt = _lookback_prompt(llm)
        assert q["entry_id"] in prompt and "```resolve" in prompt  # offered
        qgid = str(_projection(ert, q["entry_id"]).subject)
        assert qgid in _open_question_ids(ert), "nothing resolved it: it stays open"
        assert not [e for e in ert.home.diary.list_entries() if e.get("resolves")]
        assert [question_resolved_notice(m) for m in _lookback_entries(ert, run_id)] == []
    finally:
        ert.close()


def test_a_turn_that_resolves_the_question_serves_question_resolved(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path, slug="resolver")
    llm = _Scripted([ASKS])
    ert, wf = _open(home_dir, llm)
    try:
        def answers_it(_payload: Dict[str, Any]) -> str:
            qid = _question_entry(ert)["entry_id"]
            return (f"```diary kind=note resolves={qid}\ngist: Time order makes a trail\n"
                    "Time order makes a trail worth following - Laurent's answer.\n```\nThat answers my question.")

        llm.replies.extend([answers_it, "Looking back, he answered me. Goodbye."])
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-q3")
        ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
        _say(ert, wf, run_id, "Ask me something.")
        q = _question_entry(ert)
        qgid = str(_projection(ert, q["entry_id"]).subject)
        _say(ert, wf, run_id, ANSWER)

        # The turn's served notices: the entry he kept, then the question it resolved (graph id + title).
        turn2 = [a for a in _answers(ert, run_id) if a.get("turn_id") == "t-0002"]
        assert len(turn2) == 1
        notices = turn2[0]["notices"]
        assert [n["kind"] for n in notices] == ["diary", "question_resolved"]
        assert notices[1] == {"kind": "question_resolved", "record_id": qgid, "title": QUESTION_TITLE}
        assert "[kept in diary" not in turn2[0]["message"], "the served reply carries no marker"
        assert qgid not in _open_question_ids(ert)

        _close(ert, wf, run_id)
        # The look-back names the resolution (the resolution->lesson bridge) and no longer lists it as open.
        prompt = _lookback_prompt(llm)
        assert f"you resolved the open question {q['entry_id']}" in prompt
        assert "```resolve" not in prompt, "no open question left to offer"
    finally:
        ert.close()


def test_capture_meta_carries_resolves_and_its_verdict() -> None:
    """The act-only capture's word-free metadata carries the claim and the book's verdict, refused claims too; the
    notice exists only for a confirmed open question."""
    from abstractruntime.identity.act_only import capture_diary_elections

    class _Run:
        run_id = "r-1"

    def handler_for(result: Dict[str, Any]) -> Callable[..., Any]:
        def write(run: Any, effect: Effect, dnn: Any) -> Any:
            return EffectOutcome.completed({"entry_id": "diary_feed01", "projected_record_id": "ex:a1", **result})

        return write

    refused = {"resolves": "diary_ghost1", "resolves_status": "target_not_found"}
    _m, entries, _w = capture_diary_elections(
        "```diary kind=note resolves=diary_ghost1\nAn answer to nothing.\n```", run=_Run(), turn_id="t-1",
        diary_write_handler=handler_for(refused),
    )
    assert (entries[0]["resolves"], entries[0]["resolves_status"]) == ("diary_ghost1", "target_not_found")
    assert question_resolved_notice(entries[0]) is None

    confirmed = {"resolves": "diary_q00001", "resolves_status": "resolved_open_question",
                 "resolved_record_id": "ex:q1", "resolved_title": "Is time order the trail"}
    _m, entries, _w = capture_diary_elections(
        "```diary kind=note visibility=private resolves=diary_q00001\nPrivate words answer it.\n```",
        run=_Run(), turn_id="t-2", diary_write_handler=handler_for(confirmed),
    )
    assert entries[0]["resolves_status"] == "resolved_open_question"
    assert "Private words" not in json.dumps(entries[0]), "a private entry's words never ride its metadata"
    assert question_resolved_notice(entries[0]) == {
        "kind": "question_resolved", "record_id": "ex:q1", "title": "Is time order the trail",
    }

    # No resolves= at all: no claim, no verdict keys.
    _m, entries, _w = capture_diary_elections(
        "```diary kind=note\nJust a note.\n```", run=_Run(), turn_id="t-3", diary_write_handler=handler_for({}),
    )
    assert "resolves" not in entries[0] and "resolves_status" not in entries[0]


def test_resolve_block_grammar() -> None:
    marked, elections, notices = parse_resolve_blocks(
        "Before.\n```resolve\nresolves=diary_q1\nevidence=3, #5\nThe answer, in my words.\n```\nAfter."
    )
    assert [(e.resolves, e.evidence, e.text) for e in elections] == [("diary_q1", ["3", "#5"], "The answer, in my words.")]
    assert "```resolve" not in marked and notices == []
    # No question named, or no words: refused loudly, his words kept in the marker.
    marked, elections, notices = parse_resolve_blocks("```resolve\nevidence=2\nIt was answered.\n```")
    assert elections == [] and "It was answered." in marked and notices
    marked, elections, notices = parse_resolve_blocks("```resolve\nresolves=diary_q1\n```")
    assert elections == [] and notices


def test_chat_lane_lookback_prompt_is_unchanged() -> None:
    """The chat lane passes no open questions: its look-back offers no resolution and lists nothing."""
    lines = ["1. an exchange"]
    prompt = build_reflection_prompt(lines)
    assert "```resolve" not in prompt
    assert build_reflection_prompt(lines, open_questions=[]) == prompt
    offered = build_reflection_prompt(lines, open_questions=[("diary_q1", "Is it so"), ("diary_q2", None)])
    assert '- diary_q1: "Is it so"' in offered and "diary_q2" in offered and "```resolve" in offered


def test_a_question_from_an_earlier_visit_that_recall_showed_is_listed(tmp_path: Path) -> None:
    """Touched: a question kept in an earlier visit, still open, that this visit's recall showed the entity, is
    listed by this visit's look-back (it was in front of it when the visitor answered)."""
    home_dir = _make_home(tmp_path, slug="rememberer")
    llm = _Scripted([ASKS, "Looking back. Goodbye."])
    ert, wf = _open(home_dir, llm)
    try:
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-q4a")
        ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
        _say(ert, wf, run_id, "Ask me something.")
        _close(ert, wf, run_id)
        q = _question_entry(ert)
        qgid = str(_projection(ert, q["entry_id"]).subject)

        llm.calls.clear()
        llm.replies.extend(["That is what I wondered about.", "Looking back. Goodbye."])
        wf2 = build_visit_workflow(ert.home, participants=[VISITOR], idle_seconds=3600,
                                   model_info={"provider": "test", "model": "scripted"}, visit_id="visit-q4b")
        run2 = ert.runtime.start(workflow=wf2, vars={}, session_id="visit-q4b")
        ert.runtime.tick(workflow=wf2, run_id=run2, max_steps=50)
        _say(ert, wf2, run2, "About what makes a trail worth following: " + ANSWER)
        state = _close(ert, wf2, run2)
        assert qgid in state.vars["_visit"]["touched"], "this visit's recall showed the question"
        assert qgid not in [rid for rid, _d in state.vars["_visit"]["sheet"]], "not asked in this visit"
        prompt = _lookback_prompt(llm)
        assert q["entry_id"] in prompt and QUESTION_TITLE in prompt and "```resolve" in prompt
    finally:
        ert.close()
