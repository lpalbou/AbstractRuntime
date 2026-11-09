"""Every memory's title and summary are AUTHORED with it (operator rulings 2026-11-09: "the title + summary must ALWAYS
exist for any memory"; "the metadata of any newly generated memory must be generated along with the memory. the
sleep phase is just to repair any issue that may still be there"). AbstractMemory c4ffc45 holds the pending state.

Formation: the entity's block, else the summaries capability model in the same step. Pending only when that step
failed (memory queues title.author / digest.author) + one line in <workspace>/untracked/authoring-failures/. The
drainer retries pending records at the tick boundary and a session's close; the sleep repairs what is left and
declares legacy records that lack a title or summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("abstractmemory")
sys.path.insert(0, str(Path(__file__).parent))

from test_title_failures import _session, _workspace_home  # noqa: E402

TITLE = "Patience as a practice"
SUMMARY = "He learned that waiting before answering lets the other person finish their thought."


class _Capability:
    """A fake summaries capability model: answers the structured call, or raises."""

    model_info = {"provider": "fake", "model": "summarizer"}

    def __init__(self, *, title: str = TITLE, summary: str = SUMMARY, exc: Exception | None = None) -> None:
        self.title, self.summary, self.exc = title, summary, exc
        self.prompts: List[str] = []

    def generate(self, *, prompt: str, system_prompt: str, response_model: Any) -> Any:
        self.prompts.append(prompt)
        if self.exc is not None:
            raise self.exc
        return response_model(title=self.title, summary=self.summary)


def _rows(home_dir: Path, kind: str) -> List[Any]:
    from abstractmemory import SQLiteTripleStore, TripleQuery

    store = SQLiteTripleStore(home_dir / "memory.sqlite3")
    try:
        return [a for a in store.query(TripleQuery(predicate="dcterms:abstract", limit=0))
                if (a.attributes or {}).get("record_kind") == kind]
    finally:
        store.close()


def _failures(tmp_path: Path) -> List[Dict[str, Any]]:
    folder = tmp_path / "ws" / "untracked" / "authoring-failures"
    return [json.loads(ln) for p in sorted(folder.glob("*.jsonl")) for ln in p.read_text().splitlines() if ln.strip()] \
        if folder.is_dir() else []


def _tasks(home_dir: Path, kind: str) -> List[Dict[str, Any]]:
    from abstractruntime.identity.chat import open_home

    home = open_home(home_dir)
    try:
        return list(home.ms.tasks.tasks(kind=(kind,)))
    finally:
        home.close()


def test_an_episode_forms_with_its_authored_title_and_summary(tmp_path: Path) -> None:
    home_dir = _workspace_home(tmp_path)
    cap = _Capability()
    home, s = _session(home_dir, ["Hello there."])
    s.llm_for_capability = lambda name: cap if name == "summaries" else None
    s.turn("hi")
    home.close()
    ep = _rows(home_dir, "episode")[-1]
    assert ep.attributes["title"] == TITLE and str(ep.object) == SUMMARY
    assert "pending" not in ep.attributes, "authored in the formation step: nothing pending"
    assert ep.attributes["metadata_authored"]["fields"] == ["title", "summary"]
    assert ep.attributes["metadata_authored"]["source"]["kind"] == "artifact"
    assert "hi" in cap.prompts[0] and "Hello there." in cap.prompts[0]
    assert not _tasks(home_dir, "title.author") and _failures(tmp_path) == []


def test_an_untitled_lesson_gets_its_title_authored_at_formation(tmp_path: Path) -> None:
    home_dir = _workspace_home(tmp_path)
    cap = _Capability()
    block = "Dr. Smith taught me patience.\nIt matters to how I teach."
    home, s = _session(home_dir, [f"Back.\n```lesson\n{block}\n```\n", "A quiet close."])
    s.llm_for_capability = lambda name: cap
    s.turn("hi")
    home.close()
    lesson = _rows(home_dir, "lesson")[-1]
    assert lesson.attributes["title"] == TITLE, "the capability model authored the missing title"
    assert str(lesson.object) == "Dr. Smith taught me patience. It matters to how I teach.", "his words stay"
    assert "pending" not in lesson.attributes
    assert any(block.split("\n")[0] in p for p in cap.prompts)


def test_a_failing_model_leaves_the_record_pending_with_its_task_and_a_failure_line(tmp_path: Path) -> None:
    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, ["Back.\n```interest\nwhy rivers bend\n```\n", "A quiet close."])
    s.llm_for_capability = lambda name: _Capability(exc=ValueError("no fields"))
    s.turn("hi")
    home.close()
    interest = _rows(home_dir, "interest")[-1]
    assert interest.attributes.get("title") in (None, "")
    assert interest.attributes["pending"]["title"] is True
    assert [t["status"] for t in _tasks(home_dir, "title.author") if t["target_id"] == str(interest.subject)] == ["pending"]
    lines = [f for f in _failures(tmp_path) if f["kind"] == "interest"]
    assert len(lines) == 1 and lines[0]["missing"] == "title" and lines[0]["reason"] == "malformed"
    assert lines[0]["record_id"] == str(interest.subject) and lines[0]["block_reason"] == "missing"


def test_the_drainer_authors_a_pending_record_on_the_next_tick_and_defers_an_unreachable_endpoint(
        tmp_path: Path) -> None:
    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, ["Back.\n```interest\nwhy rivers bend\n```\n", "A quiet close."])
    s.llm_for_capability = None  # no capability model at formation: the record forms pending
    s.turn("hi")
    target = str(_rows(home_dir, "interest")[-1].subject)
    # An unreachable endpoint: the task waits, no attempt spent.
    s.llm_for_capability = lambda name: _Capability(exc=ConnectionError("connection refused"))
    receipt = s.drain_authoring(window="turn", cap=3)
    assert receipt["deferred"] >= 1 and receipt["authored"] == 0
    task = [t for t in home.ms.tasks.tasks(kind=("title.author",)) if t["target_id"] == target][0]
    assert task["status"] == "deferred" and int(task.get("attempts") or 0) == 0, task
    # The endpoint is back: the drainer authors it (the defer's not_before is in the future, so release it now).
    from datetime import datetime, timedelta, timezone

    later = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    sel = home.ms.tasks.select_for_window(10_000.0, window="turn", kinds=["title.author"], now=later)
    assert any(t["target_id"] == target for t in sel["tasks"])
    s.llm_for_capability = lambda name: _Capability()
    from abstractruntime.identity.authoring import drain_authoring

    import abstractruntime.identity.authoring as authoring_mod

    # An hour later (past the defer): the selection and the claim read that clock.
    real_select, real_claim = home.ms.tasks.select_for_window, home.ms.tasks.claim
    home.ms.tasks.select_for_window = lambda *a, **k: real_select(*a, **{**k, "now": later})
    home.ms.tasks.claim = lambda *a, **k: real_claim(*a, **{**k, "now": later})
    try:
        receipt = drain_authoring(home.ms, llm_for_capability=s.llm_for_capability, artifacts=home.artifacts,
                                  home_dir=home_dir, entity=home.entity_id, holder="test", cap=3, window="turn")
    finally:
        home.ms.tasks.select_for_window, home.ms.tasks.claim = real_select, real_claim
    assert receipt["authored"] >= 1, receipt
    home.close()
    interest = [r for r in _rows(home_dir, "interest") if str(r.subject) == target][-1]
    assert interest.attributes["title"] == TITLE and "pending" not in interest.attributes
    assert authoring_mod.AUTHORING_TASK_KINDS == ("title.author", "digest.author")


def test_the_sleep_repairs_a_legacy_record_without_a_title(tmp_path: Path, monkeypatch) -> None:
    """(e) A record left from before titles were mandatory (here: the 2026-11-09 placeholder, no marker) is declared
    pending and authored as one repair step of the sleep pass, reported in its receipt."""
    import abstractmemory
    from abstractmemory import MemoryRecordInput

    from abstractruntime.identity.chat import open_home
    from abstractruntime.identity.life import build_consolidator

    home_dir = _workspace_home(tmp_path)
    home = open_home(home_dir)
    rid = home.ms.remember_many([MemoryRecordInput(
        kind="lesson", title="untitled lesson [3fa2c1]", digest="Waiting lets the other finish.",
        attributes={"title_missing": True, "title_failure_reason": "missing"},
    )], scope="life", owner_id=home.entity_id, idempotency_key="legacy-1", turn_id="legacy-1")[0]
    home.close()
    monkeypatch.setattr(abstractmemory, "sleep_pass", lambda *a, **k: {"maintenance": {}, "dream": {}})
    from abstractruntime.identity.life import write_entity_state

    write_entity_state(home_dir, "asleep", reason="self-elected sleep: full", mode="dreaming", written_by="self")
    result = build_consolidator(home_dir, out=lambda s: None, llm_for_capability=lambda name: _Capability())()
    assert result["authoring"]["declared"] == 1 and result["authoring"]["authored"] == 1, result["authoring"]
    lesson = [r for r in _rows(home_dir, "lesson") if str(r.subject) == str(rid)][-1]
    assert lesson.attributes["title"] == TITLE


def test_the_spawn_hands_the_capability_profiles_to_the_child(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    from test_entity_life_loop import _make_home

    from abstractruntime.identity.life import spawn_loop_process

    home_dir = _make_home(tmp_path)
    seen: List[Dict[str, str]] = []

    class _Proc:
        pid = 999998

        def poll(self):
            return None

    def fake_popen(argv, *a, **k):
        seen.append(dict(k.get("env") or {}))
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    try:
        spawn_loop_process(home_dir, provider="lmstudio", model="m", capability_endpoint_profiles={
            "summaries": {"spec": "endpoint:theforge", "profile": {"base_url": "http://x", "api_key": "k"}}})
    except Exception:  # noqa: BLE001 - only what the child would receive matters
        pass
    packed = json.loads(seen[0]["ABSTRACTRUNTIME_CAPABILITY_ENDPOINT_PROFILES_JSON"])
    assert packed == {"summaries": {"spec": "endpoint:theforge", "profile": {"base_url": "http://x", "api_key": "k"}}}


# ---------------------------------------------------------------- the durable visit lane: the authoring call is an
# LLM_CALL tagged capability=summaries, routed by open_entity_runtime(capability_handlers=...)


def _visit(tmp_path: Path, capability_handlers: Any):
    from test_visit_midturn_elections import _ScriptedHandler, _make_home as _visit_home

    from abstractruntime.core.models import EffectType, RunStatus
    from abstractruntime.identity.entity_runtime import open_entity_runtime
    from abstractruntime.identity.visit_workflow import VISITOR_WAIT_KEY, build_visit_workflow

    home_dir = _visit_home(tmp_path)
    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: _ScriptedHandler(["A plain reply."])},
                              capability_handlers=capability_handlers)
    wf = build_visit_workflow(ert.home, participants=["person:admin"], idle_seconds=3600,
                              model_info={"provider": "test", "model": "scripted"}, visit_id="visit-authoring")
    run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-authoring")
    assert ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50).status == RunStatus.WAITING
    state = ert.runtime.resume(workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
                               payload={"text": "hello", "speaker": "person:admin"}, max_steps=200)
    assert state.status == RunStatus.WAITING, state.error
    from abstractmemory import TripleQuery

    episodes = [a for a in ert.home.ms.store.query(TripleQuery(predicate="dcterms:abstract", limit=0))
                if (a.attributes or {}).get("record_kind") == "episode"]
    return ert, home_dir, episodes


def test_a_visit_turn_forms_its_episode_with_the_authored_title_and_summary(tmp_path: Path) -> None:
    from abstractruntime.core.runtime import EffectOutcome

    calls: List[Dict[str, Any]] = []

    def summaries(run: Any, effect: Any, dnn: Any = None) -> Any:
        calls.append(dict(effect.payload))
        return EffectOutcome.completed({"content": json.dumps({"title": TITLE, "summary": SUMMARY}), "model": "sum-1"})

    ert, home_dir, episodes = _visit(tmp_path, {"summaries": summaries})
    try:
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.attributes["title"] == TITLE and str(ep.object) == SUMMARY, ep.attributes
        assert "pending" not in ep.attributes
        assert ep.attributes["metadata_authored"]["capability"] == "summaries"
        payload = calls[0]
        assert payload["capability"] == "summaries" and payload["params"]["thinking"] == "off"
        assert payload["response_schema"]["required"] == ["title", "summary"]
        assert "hello" in payload["messages"][0]["content"] and "A plain reply." in payload["messages"][0]["content"]
    finally:
        ert.close()


@pytest.mark.parametrize("wiring", ["failing", "unrouted"])
def test_a_failed_or_unrouted_authoring_call_forms_pending_with_a_failure_line(tmp_path: Path, wiring: str) -> None:
    from abstractruntime.core.runtime import EffectOutcome

    handlers = {"summaries": lambda run, effect, dnn=None: EffectOutcome.failed("endpoint refused")} \
        if wiring == "failing" else {}
    ert, home_dir, episodes = _visit(tmp_path, handlers)
    try:
        ep = episodes[0]
        assert ep.attributes.get("title") in (None, "") and ep.attributes["pending"]["title"] is True
        assert ep.attributes["pending"]["summary"] is True
        tasks = {t["kind"] for t in ert.home.ms.tasks.tasks() if t["target_id"] == str(ep.subject)}
        assert {"title.author", "digest.author"} <= tasks
        log = (home_dir / "own_time.log").read_text(encoding="utf-8")
        lines = [json.loads(ln[len("#AUTHORING_FAILURE "):]) for ln in log.splitlines()
                 if ln.startswith("#AUTHORING_FAILURE ")]
        line = [ln for ln in lines if ln["kind"] == "episode"][0]
        assert line["record_id"] == str(ep.subject) and line["missing"] == "both"
        assert line["reason"] == ("unavailable" if wiring == "failing" else "capability_unrouted"), line
    finally:
        ert.close()


def _close_visit(tmp_path: Path, capability_handlers: Any):
    """One visit turn, then the close: the look-back elects an interest, a lesson and a realization, none titled."""
    from test_visit_midturn_elections import _ScriptedHandler, _make_home as _visit_home

    from abstractruntime.core.models import EffectType, RunStatus
    from abstractruntime.identity.entity_runtime import open_entity_runtime
    from abstractruntime.identity.visit_workflow import VISITOR_WAIT_KEY, build_visit_workflow

    def look_back(run: Any) -> str:
        rid = str(list((run.vars.get("_visit") or {}).get("sheet") or [])[0][0])
        return ("A good visit.\n```interest\nwhy rivers bend\n```\n```lesson\nwait before answering\n```\n"
                f"```realize\nI keep returning to rivers.\nevidence={rid}\n```\n")

    home_dir = _visit_home(tmp_path)
    ert = open_entity_runtime(home_dir, extra_handlers={
        EffectType.LLM_CALL: _ScriptedHandler(["A plain reply.", look_back])}, capability_handlers=capability_handlers)
    wf = build_visit_workflow(ert.home, participants=["person:admin"], idle_seconds=3600,
                              model_info={"provider": "test", "model": "scripted"}, visit_id="visit-close")
    run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-close")
    ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
    ert.runtime.resume(workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
                       payload={"text": "hello", "speaker": "person:admin"}, max_steps=200)
    state = ert.runtime.resume(workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
                               payload={"kind": "close"}, max_steps=600)
    assert state.status == RunStatus.COMPLETED, state.error
    from abstractmemory import TripleQuery

    rows = {}
    for a in ert.home.ms.store.query(TripleQuery(predicate="dcterms:abstract", limit=0)):
        kind = (a.attributes or {}).get("record_kind")
        if kind in ("summary", "interest", "lesson", "realization", "episode"):
            rows.setdefault(kind, []).append(a)
    return ert, home_dir, rows


def test_every_record_a_visit_close_forms_carries_its_authored_metadata(tmp_path: Path) -> None:
    from abstractruntime.core.runtime import EffectOutcome

    asked: List[str] = []

    def summaries(run: Any, effect: Any, dnn: Any = None) -> Any:
        content = effect.payload["messages"][0]["content"]
        kind = content.split("(kind: ", 1)[1].split(")", 1)[0]
        asked.append(kind)
        return EffectOutcome.completed({"content": json.dumps(
            {"title": f"{kind.capitalize()} title", "summary": f"What this {kind} holds."})})

    ert, home_dir, rows = _close_visit(tmp_path, {"summaries": summaries})
    try:
        for kind in ("summary", "interest", "lesson", "realization", "episode"):
            assert rows.get(kind), (kind, sorted(rows))
            for a in rows[kind]:
                assert "pending" not in (a.attributes or {}), (kind, a.attributes)
                assert a.attributes["title"] == f"{kind.capitalize()} title", (kind, a.attributes.get("title"))
                assert a.attributes["metadata_authored"]["capability"] == "summaries"
        # The summary's words are authored; an elected record keeps the entity's words as its summary.
        assert str(rows["summary"][0].object) == "What this summary holds."
        assert str(rows["interest"][0].object) == "why rivers bend"
        assert str(rows["lesson"][0].object) == "wait before answering"
        assert sorted(asked) == ["episode", "interest", "lesson", "realization", "summary"]
        assert not ert.home.ms.tasks.tasks(kind=("title.author", "digest.author"))
    finally:
        ert.close()


def test_a_failing_handler_leaves_the_close_records_pending_with_their_failure_lines(tmp_path: Path) -> None:
    from abstractruntime.core.runtime import EffectOutcome

    ert, home_dir, rows = _close_visit(
        tmp_path, {"summaries": lambda run, effect, dnn=None: EffectOutcome.failed("endpoint refused")})
    try:
        log = (home_dir / "own_time.log").read_text(encoding="utf-8")
        lines = [json.loads(ln[len("#AUTHORING_FAILURE "):]) for ln in log.splitlines()
                 if ln.startswith("#AUTHORING_FAILURE ")]
        for kind, missing in (("summary", "both"), ("interest", "title"), ("lesson", "title"),
                              ("realization", "title")):
            a = rows[kind][0]
            assert a.attributes["pending"]["title"] is True, (kind, a.attributes)
            mine = [ln for ln in lines if ln["kind"] == kind]
            assert len(mine) == 1 and mine[0]["record_id"] == str(a.subject) and mine[0]["missing"] == missing, mine
    finally:
        ert.close()


def test_the_sleep_repairs_legacy_mechanical_cards_and_dreams(tmp_path: Path, monkeypatch) -> None:
    """Memory 4f3a77e forms cards and dreams pending; ones stored before it carry memory's mechanical labels ("World
    model: <subject>", "Dream: ...", a mechanical-card body). The sleep's repair DECLARES them pending through
    memory's declare_pending (11687df) - the marker and tasks formation gives - so the drainer authors their titles
    (retitle) AND their summaries (rewrite_digest: the card through its own write, the dream in place); nothing is
    refused any more (the memory_refused path of 7d2419e is gone)."""
    import abstractmemory
    from abstractmemory import MemoryRecordInput

    from abstractruntime.identity.chat import open_home
    from abstractruntime.identity.life import build_consolidator, write_entity_state

    home_dir = _workspace_home(tmp_path)
    home = open_home(home_dir)
    ids = {}
    for kind, title, attrs in (
            ("world_model", "World model: person:sol",
             {"target": "person:sol", "digest_method": "mechanical-card-v2", "revision": 1}),
            ("dream", "Dream: rivers beside patience", {"interpretation_required": True})):
        ids[kind] = home.ms.remember_many([MemoryRecordInput(kind=kind, title=title, digest=f"mechanical {kind} body",
                                                             attributes=attrs)],
                                          scope="self", owner_id=home.entity_id, idempotency_key=f"legacy-{kind}",
                                          turn_id=f"legacy-{kind}")[0]
    home.close()
    monkeypatch.setattr(abstractmemory, "sleep_pass", lambda *a, **k: {"maintenance": {}, "dream": {}})
    write_entity_state(home_dir, "asleep", reason="self-elected sleep: full", mode="dreaming", written_by="self")
    result = build_consolidator(home_dir, out=lambda s: None, llm_for_capability=lambda name: _Capability())()
    receipt = result["authoring"]
    assert receipt["declared"] == 4 and receipt["authored"] == 4, receipt  # both titles and both summaries
    assert receipt["failed"] == 0 and receipt["refused"] == 0, receipt
    for kind, rid in ids.items():
        row = [r for r in _rows(home_dir, kind) if str(r.subject) == str(rid)][-1]
        assert row.attributes["title"] == TITLE, (kind, row.attributes["title"])
        assert row.object == SUMMARY, (kind, row.object)
        assert not isinstance(row.attributes.get("pending"), dict), (kind, row.attributes.get("pending"))
    assert _failures(tmp_path) == [], "no authoring failure, no memory_refused"
