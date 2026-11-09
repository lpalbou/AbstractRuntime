"""Every memory's title and summary are AUTHORED with it (operator rulings 2026-11-09: "the title + summary must ALWAYS
exist for any memory"; "the metadata of any newly generated memory must be generated along with the memory. the
sleep phase is just to repair any issue that may still be there"). AbstractMemory c4ffc45 holds the pending state.

Formation: the entity's block, else the summaries capability model in the same step. Pending only when that step
failed (memory queues title.author / digest.author) + one line in <workspace>/untracked/authoring-failures/. The
drainer retries pending records at the tick boundary and a session's close; the sleep repairs what is left and
declares legacy records that lack a title or summary.
"""

from __future__ import annotations

from abstractruntime.identity.lifecycle_fence import LifecycleFence  # noqa: E402
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
    assert len(lines) == 1 and lines[0]["missing"] == "title" and lines[0]["record_id"] == str(interest.subject)
    assert lines[0]["block_reason"] == "missing"
    # The client RAISED: its class, message and the capability model it was are on the line - never "malformed"
    # with no model (live gateway 2026-11-09: every visit-formed record failed that way and said nothing).
    line = lines[0]
    assert (line["reason"], line["error_class"], line["error_message"]) == ("client_error", "ValueError", "no fields")
    assert (line["provider"], line["model"], line["capability"]) == ("fake", "summarizer", "summaries"), line


def test_the_drainer_failure_line_names_the_client_exception_and_model(tmp_path: Path) -> None:
    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, ["Back.\n```interest\nwhy rivers bend\n```\n", "A quiet close."])
    s.llm_for_capability = None  # forms pending
    s.turn("hi")
    s.llm_for_capability = lambda name: _Capability(exc=RuntimeError("gateway 500: capability route missing"))
    receipt = s.drain_authoring(window="turn", cap=3)
    assert receipt["failed"] >= 1, receipt
    line = [f for f in _failures(tmp_path) if f.get("task_id")][-1]
    assert (line["reason"], line["error_class"]) == ("client_error", "RuntimeError"), line
    assert line["error_message"] == "gateway 500: capability route missing"
    assert (line["provider"], line["model"]) == ("fake", "summarizer"), line
    # A reply that truly failed to parse stays malformed - with the model that gave it.
    s.llm_for_capability = lambda name: _Capability(summary="")
    s.drain_authoring(window="turn", cap=3)
    home.close()
    bad = [f for f in _failures(tmp_path) if f.get("task_id")][-1]
    assert (bad["reason"], bad["model"]) == ("malformed", "summarizer"), bad


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

    write_entity_state(home_dir, "asleep", reason="self-elected sleep: full", mode="dreaming", written_by="self", fence=LifecycleFence.admit(home_dir))
    result = build_consolidator(home_dir, out=lambda s: None, llm_for_capability=lambda name: _Capability(), fence=LifecycleFence.admit(home_dir))()
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


def _close_visit(tmp_path: Path, capability_handlers: Any, *, diary: bool = False):
    """One visit turn, then the close: the look-back elects an interest, a lesson and a realization, none titled."""
    from test_visit_midturn_elections import _ScriptedHandler, _make_home as _visit_home

    from abstractruntime.core.models import EffectType, RunStatus
    from abstractruntime.identity.entity_runtime import open_entity_runtime
    from abstractruntime.identity.visit_workflow import VISITOR_WAIT_KEY, build_visit_workflow

    def look_back(run: Any) -> str:
        rid = str(list((run.vars.get("_visit") or {}).get("sheet") or [])[0][0])
        return ("A good visit.\n```interest\nwhy rivers bend\n```\n```lesson\nwait before answering\n```\n"
                f"```realize\nI keep returning to rivers.\nevidence={rid}\n```\n"
                + ("```diary kind=note\nThe visitor asked about rivers.\n```\n" if diary else ""))

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
        elif kind == "diary" and (a.attributes or {}).get("entry_id") != "diary_seedq":
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
    write_entity_state(home_dir, "asleep", reason="self-elected sleep: full", mode="dreaming", written_by="self", fence=LifecycleFence.admit(home_dir))
    result = build_consolidator(home_dir, out=lambda s: None, llm_for_capability=lambda name: _Capability(), fence=LifecycleFence.admit(home_dir))()
    receipt = result["authoring"]
    assert receipt["declared"] == 4 and receipt["authored"] == 4, receipt  # both titles and both summaries
    assert receipt["failed"] == 0 and receipt["refused"] == 0, receipt
    for kind, rid in ids.items():
        row = [r for r in _rows(home_dir, kind) if str(r.subject) == str(rid)][-1]
        assert row.attributes["title"] == TITLE, (kind, row.attributes["title"])
        assert row.object == SUMMARY, (kind, row.object)
        assert not isinstance(row.attributes.get("pending"), dict), (kind, row.attributes.get("pending"))
    assert _failures(tmp_path) == [], "no authoring failure, no memory_refused"


def _exhaust_title_task(home: Any, s: Any, target: str) -> Dict[str, Any]:
    """Fail the record's title.author task through its attempts (a capability client that raises, as the live
    gateway's did on 2026-11-09): it ends `failed`, the record stays marked pending with no live task."""
    s.llm_for_capability = lambda name: _Capability(exc=RuntimeError("gateway 500: capability route missing"))
    for _ in range(6):
        task = [t for t in home.ms.tasks.tasks(kind=("title.author",)) if t["target_id"] == target][-1]
        if task["status"] == "failed":
            return task
        from abstractruntime.identity.authoring import drain_authoring

        drain_authoring(home.ms, llm_for_capability=s.llm_for_capability, artifacts=home.artifacts,
                        home_dir=home.home_dir, entity=home.entity_id, holder="test", cap=3, window="turn")
    raise AssertionError(f"the task never ended failed: {task}")


def test_a_pending_record_whose_tasks_all_failed_is_rearmed_and_authored_at_the_next_drainer_pass(
        tmp_path: Path) -> None:
    import abstractruntime.identity.authoring as authoring_mod

    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, ["Back.\n```interest\nwhy rivers bend\n```\n", "A quiet close."])
    s.llm_for_capability = None  # forms pending
    s.turn("hi")
    target = str(_rows(home_dir, "interest")[-1].subject)
    failed = _exhaust_title_task(home, s, target)
    assert failed["status"] == "failed"
    # The capability is fixed: the next pass re-arms the record through memory and authors it.
    lines: List[str] = []
    receipt = authoring_mod.drain_authoring(home.ms, llm_for_capability=lambda name: _Capability(),
                                            artifacts=home.artifacts, home_dir=home_dir, entity=home.entity_id,
                                            holder="test", cap=3, window="turn", out=lines.append)
    home.close()
    # Every exhausted record of the turn is re-armed (the interest, and the episode whose tasks ran out beside it).
    assert receipt["rearmed"] >= 1 and receipt["authored"] >= 1, receipt
    interest = [r for r in _rows(home_dir, "interest") if str(r.subject) == target][-1]
    assert interest.attributes["title"] == TITLE and "pending" not in interest.attributes
    assert any("re-armed" in ln for ln in lines), lines


def test_without_memorys_rearm_verb_the_drainer_says_so_once_and_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    import abstractmemory

    import abstractruntime.identity.authoring as authoring_mod

    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, ["Back.\n```interest\nwhy rivers bend\n```\n", "A quiet close."])
    s.llm_for_capability = None
    s.turn("hi")
    target = str(_rows(home_dir, "interest")[-1].subject)
    _exhaust_title_task(home, s, target)
    monkeypatch.delattr(abstractmemory, "rearm_pending", raising=False)
    monkeypatch.setattr(authoring_mod, "_REARM_VERB_MISSING_LOGGED", False)
    lines: List[str] = []
    for _ in range(2):
        receipt = authoring_mod.drain_authoring(home.ms, llm_for_capability=lambda name: _Capability(),
                                                artifacts=home.artifacts, home_dir=home_dir, entity=home.entity_id,
                                                holder="test", cap=3, window="turn", out=lines.append)
        assert receipt["rearmed"] == 0 and receipt["authored"] == 0, receipt
    home.close()
    assert len([ln for ln in lines if ln.startswith("#MISSING_VERB abstractmemory.rearm_pending")]) == 1, lines
    assert [t["status"] for t in _tasks(home_dir, "title.author") if t["target_id"] == target] == ["failed"]


def test_a_declaration_memory_answers_rearmed_is_a_repair_not_a_refusal() -> None:
    """AbstractMemory 39e479e: declare_pending on a field already pending with no live task REPAIRS it and answers
    outcome "rearmed" (`repaired` the fields); "applied" may carry `repaired` beside new fields. The sleep's scan
    counts both as success, never as a refusal; an outcome it does not know is said loudly."""
    from types import SimpleNamespace

    from abstractruntime.identity.authoring import declare_missing_metadata

    answers = iter([
        {"outcome": "rearmed", "reason": None, "declared": [], "repaired": ["title"]},
        {"outcome": "applied", "declared": ["summary"], "repaired": ["title"]},
        {"outcome": "unchanged", "reason": "already_pending", "declared": [], "repaired": []},
        {"outcome": "mystery"},
    ])
    rows = [SimpleNamespace(subject=f"ex:interest-{i}", object="", assertion_id=f"a{i}",
                            attributes={"record_kind": "interest", "title": None}) for i in range(4)]
    ms = SimpleNamespace(tasks=object(), store=SimpleNamespace(query=lambda q: rows),
                         declare_pending=lambda rid, **kw: next(answers))
    lines: List[str] = []
    receipt = declare_missing_metadata(ms, scopes=[("self", "entity:x")], out=lines.append)
    assert (receipt["declared"], receipt["repaired"], receipt["refused"]) == (1, 2, 1), receipt
    assert [ln for ln in lines if ln.startswith("#REPAIR_REFUSED")] == [
        "#REPAIR_REFUSED memory answered an unknown outcome 'mystery' declaring interest ex:interest-3"]


# ---------------------------------------------------------------- Codex-2 #10: a diary entry's record is authored at formation


_DIARY_GISTLESS = "Back.\n```diary kind=note\nThe tide leaves a readable line on the sand.\nIt stays until the next wave.\n```\n"
_DIARY_GIST = "Back.\n```diary kind=note\ngist: the tide line\nThe tide leaves a readable line on the sand.\n```\n"


def _diary_rows(home_dir: Path) -> List[Any]:
    from abstractmemory import SQLiteTripleStore, TripleQuery

    store = SQLiteTripleStore(home_dir / "memory.sqlite3")
    try:
        return [a for a in store.query(TripleQuery(scope="diary", predicate="dcterms:abstract", limit=0))
                if (a.attributes or {}).get("record_kind") == "diary"
                and (a.attributes or {}).get("entry_id") != "diary_seedq"]  # the home's seeded question
    finally:
        store.close()


def test_a_diary_entry_forms_with_its_authored_title_and_summary_never_a_template(tmp_path: Path) -> None:
    """Live 2026-11-09 (castor): a new diary entry's record read "Diary entry (note) - <date>: ...". Its SUMMARY is
    authored BEFORE its record forms (one summaries call; AbstractMemory 71de051 takes it with its provenance), and so
    is its title without a gist; with one, his gist is his title, whole. Nothing is pending; no first line stands in
    (Codex review 3 #9/#11)."""
    home_dir = _workspace_home(tmp_path)
    cap = _Capability()
    both = _DIARY_GISTLESS + _DIARY_GIST.replace("Back.\n", "")
    home, s = _session(home_dir, [both, "A quiet close."])
    s.llm_for_capability = lambda name: cap
    s.turn("hi")
    home.close()
    rows = _diary_rows(home_dir)
    assert len(rows) == 2, rows
    gistless, gisted = sorted(rows, key=lambda r: r.attributes.get("title") != TITLE)
    assert (gistless.attributes["title"], str(gistless.object)) == (TITLE, SUMMARY)
    assert (gisted.attributes["title"], str(gisted.object)) == ("the tide line", SUMMARY)
    for r in rows:
        assert "Diary entry" not in str(r.attributes["title"]) and "pending" not in r.attributes
        assert "The tide leaves a readable line" not in str(r.object), "no first line stands in for the summary"
        stamp = r.attributes["understanding"]
        assert stamp["field"] == "digest" and stamp["author"] == {"kind": "capability-model", "id": "summaries"}
        assert stamp["reader"]["model"] == "summarizer" and stamp["sources"][0]["kind"] == "artifact"
        assert r.attributes["digest_method"] == "authored-v1"
    assert gistless.attributes["metadata_authored"]["fields"] == ["title", "summary"]
    assert gisted.attributes["metadata_authored"]["fields"] == ["summary"], "his gist stays his title"
    diary_prompts = [p for p in cap.prompts if p.startswith("The memory (kind: diary)")]
    assert len(diary_prompts) == 2 and all("The tide leaves a readable line" in p for p in diary_prompts)
    assert [f for f in _failures(tmp_path) if f["kind"] == "diary"] == []


def test_a_diary_entry_whose_authoring_fails_forms_pending_with_one_failure_line(tmp_path: Path) -> None:
    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, [_DIARY_GISTLESS, "A quiet close."])
    s.llm_for_capability = lambda name: _Capability(exc=RuntimeError("gateway 500: capability route missing"))
    s.turn("hi")
    home.close()
    (row,) = _diary_rows(home_dir)
    assert row.attributes.get("title") in (None, "") and row.attributes["pending"]["title"] is True
    assert row.attributes["pending"]["summary"] is True, "the summary waits for its author, never a stand-in"
    assert str(row.object) == "The tide leaves a readable line on the sand.", "provisional words, marked pending"
    lines = [f for f in _failures(tmp_path) if f["kind"] == "diary"]
    assert len(lines) == 1, lines
    line = lines[0]
    assert (line["record_id"], line["missing"], line["reason"]) == (str(row.subject), "both", "client_error")
    assert (line["error_class"], line["model"]) == ("RuntimeError", "summarizer"), line


def test_a_visit_close_diary_entry_forms_with_its_authored_title(tmp_path: Path) -> None:
    """Codex review 2 #10 / review 3 #9, the visit lane: a gistless public diary entry the close writes gets its title
    and summary from one capability=summaries LLM_CALL BEFORE its record forms."""
    from abstractruntime.core.runtime import EffectOutcome

    asked: List[str] = []

    def summaries(run: Any, effect: Any, dnn: Any = None) -> Any:
        content = effect.payload["messages"][0]["content"]
        kind = content.split("(kind: ", 1)[1].split(")", 1)[0]
        asked.append(kind)
        return EffectOutcome.completed({"capability": "summaries", "content": json.dumps(
            {"title": f"{kind.capitalize()} title", "summary": f"What this {kind} holds."})})

    ert, home_dir, rows = _close_visit(tmp_path, {"summaries": summaries}, diary=True)
    try:
        (entry,) = rows["diary"]
        assert entry.attributes["title"] == "Diary title" and "pending" not in entry.attributes, (
            sorted(entry.attributes.items()), asked, _failures(tmp_path))
        assert str(entry.object) == "What this diary holds.", "the summary is authored too (Codex review 3 #9)"
        assert entry.attributes["metadata_authored"]["fields"] == ["title", "summary"]
        assert entry.attributes["understanding"]["sources"], "the authoring call is named as its source"
        assert "diary" in asked
    finally:
        ert.close()


# ---------------------------------------------------------------- AbstractMemory c46b4cd: memory's passes author WITH the
# record through the runtime's summaries capability (Codex review 2 #10)


def _night_with_an_interest(tmp_path: Path, capability: Any) -> Any:
    from abstractruntime.identity.life import build_consolidator, write_entity_state

    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, ["Back.\n```interest\ntitle: Rivers and their bends\nwhy rivers bend\n```\n",
                                  "A quiet close."])
    s.llm_for_capability = lambda name: _Capability()
    s.turn("hi")
    home.close()
    write_entity_state(home_dir, "asleep", reason="self-elected sleep: full", mode="dreaming", written_by="self", fence=LifecycleFence.admit(home_dir))
    lines: List[str] = []
    build_consolidator(home_dir, out=lines.append, llm_for_capability=lambda name: capability, fence=LifecycleFence.admit(home_dir))()
    cards = [a for a in _rows(home_dir, "world_model")]
    return home_dir, cards, lines


def test_the_nights_world_model_card_forms_with_its_authored_title_and_summary(tmp_path: Path) -> None:
    """The sleep pass's world_model_pass forms the interest's card WITH its title and summary, authored by the
    summaries capability through the runtime's formation author - never pending for want of an author."""
    home_dir, cards, _lines = _night_with_an_interest(tmp_path, _Capability())
    assert cards, "the night formed the interest's card"
    card = cards[-1]
    assert card.attributes["title"] == TITLE and str(card.object) == SUMMARY, (card.attributes, card.object)
    assert not isinstance(card.attributes.get("pending"), dict), card.attributes.get("pending")
    # Authored AS it formed: no authoring task was ever queued for it (a pending card the sleep's repair step authors
    # afterwards would carry its title.author / digest.author tasks).
    owed = [t for k in ("title.author", "digest.author") for t in _tasks(home_dir, k)
            if t["target_id"] == str(card.subject)]
    assert owed == [], owed
    assert [f for f in _failures(tmp_path) if f["kind"] == "world_model"] == []


def test_a_raising_capability_leaves_the_nights_card_pending_with_its_failure_line(tmp_path: Path) -> None:
    home_dir, cards, _lines = _night_with_an_interest(tmp_path, _Capability(exc=RuntimeError("capability 500")))
    card = cards[-1]
    pending = card.attributes.get("pending")
    assert isinstance(pending, dict) and pending["title"] and pending["failure"]["reason"] == "author_raised"
    # The pass's own line (the drainer's retries of the card's tasks after it write theirs, client_error).
    lines = [f for f in _failures(tmp_path) if f["kind"] == "world_model" and f.get("pass") == "world_model_pass"]
    assert len(lines) == 1, _failures(tmp_path)
    line = lines[0]
    assert (line["reason"], line["error_class"], line["error_message"]) == ("author_raised", "RuntimeError",
                                                                             "capability 500")
    assert line["record_id"] == str(card.subject) and line["model"] == "summarizer", line


@pytest.mark.parametrize("kind", ["world_model", "dream", "summary"])
def test_the_runtime_author_satisfies_memorys_formation_contract_for_every_pass_kind(tmp_path: Path, kind: str) -> None:
    """Every memory pass (world models, dreams, candidates, consolidation, wake-cue dedup) asks the same author
    through formation_authoring.author_formation: the runtime's answer is authored metadata memory accepts (author
    capability-model, reader, sources = the call kept as an artifact) - no failure, nothing pending."""
    from abstractmemory.formation_authoring import author_formation

    from abstractruntime.identity.authoring import capability_formation_author
    from abstractruntime.identity.chat import open_home

    home = open_home(_workspace_home(tmp_path))
    try:
        author = capability_formation_author(lambda name: _Capability(), home.artifacts)
        got = author_formation(author, kind=kind, body="what the pass composed",
                               context={"pass": f"{kind}_pass", "record_kind": kind}, fields=["title", "summary"])
    finally:
        home.close()
    assert got.failure is None and got.pending == (), got
    assert (got.title, got.summary) == (TITLE, SUMMARY)
    assert got.author["kind"] == "capability-model" and got.reader["model"] == "summarizer" and got.sources


def test_a_sessions_card_update_passes_the_author_and_writes_its_failures(tmp_path: Path, monkeypatch) -> None:
    import abstractmemory

    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, ["Hello."])
    s.llm_for_capability = lambda name: _Capability()
    seen: Dict[str, Any] = {}

    def spy(ms, *, scopes, owner_id, targets, author=None, **kw):
        seen["author"] = author
        return {"formed": [], "authoring": {"authored": [], "failures": [{
            "record_id": "ex:world_model-1", "record_kind": "world_model", "pass": "world_model_update",
            "fields": ["title", "summary"], "reason": "invalid_result", "error_class": "ValueError",
            "message": "no title", "at": "2026-11-09T18:00:00+00:00"}]}}

    monkeypatch.setattr(abstractmemory, "world_model_update", spy)
    try:
        s._world_model_update(abstractmemory.world_model_update, ["person:sol"])
    finally:
        home.close()
    assert callable(seen["author"]), "the session's card update is authored by the summaries capability"
    lines = [f for f in _failures(tmp_path) if f["kind"] == "world_model"]
    assert len(lines) == 1 and (lines[0]["reason"], lines[0]["missing"]) == ("invalid_result", "both"), lines


def test_the_flow_brain_sleep_pass_is_authored_by_the_hosts_summaries_handler(tmp_path: Path, monkeypatch) -> None:
    import abstractmemory

    from abstractruntime.core.models import EffectType
    from abstractruntime.core.runtime import EffectOutcome
    from abstractruntime.identity.entity_runtime import open_entity_runtime

    home_dir = _workspace_home(tmp_path)
    seen: Dict[str, Any] = {}

    def spy(ms, **kw):
        seen["author"] = kw.get("author")
        return {"maintenance": {}, "dream": {}}

    def summaries(run: Any, effect: Any, dnn: Any = None) -> Any:
        return EffectOutcome.completed({"content": json.dumps({"title": TITLE, "summary": SUMMARY})})

    monkeypatch.setattr(abstractmemory, "sleep_pass", spy)
    ert = open_entity_runtime(home_dir, capability_handlers={"summaries": summaries})
    try:
        from types import SimpleNamespace

        handler = ert.runtime._handlers[EffectType.MEMORY_CONSOLIDATE]
        handler(SimpleNamespace(run_id="r1", vars={}), SimpleNamespace(payload={"report_only": True}), None)
        author = seen.get("author")
        assert callable(author), "MEMORY_CONSOLIDATE passes the host's summaries capability as the author"
        result = author("world_model", "a card's evidence", {"pass": "world_model_pass", "fields": ["title"]})
        assert result["title"] == TITLE and result["author"]["kind"] == "capability-model" and result["sources"]
    finally:
        ert.close()


def test_a_tick_lookback_summary_has_an_authored_title_never_the_tick_template(tmp_path: Path) -> None:
    """Codex review 3 #8: the tick look-back's summary record was titled "tick reflection: <session> #<n>" and only
    its summary was authored. Both are authored now; the tick's label stays as structure (lookback_label)."""
    from test_lookback_offers import LOOKBACK_REPLY, _Recorder

    from abstractruntime.identity.chat import ChatSession, open_home

    home_dir = _workspace_home(tmp_path)
    home = open_home(home_dir)
    s = ChatSession(home, _Recorder(["A tick.", LOOKBACK_REPLY]), participants=[home.entity_id],
                    context_window=20000, out=lambda _s: None, phase="personal")
    s.llm_for_capability = lambda name: _Capability()
    s.turn("(cue) one")
    result = s.tick_lookback()
    home.close()
    rid = result["session_record_id"]
    (row,) = [a for a in _rows(home_dir, "summary") if str(a.subject) == str(rid)]
    assert (row.attributes["title"], str(row.object)) == (TITLE, SUMMARY), row.attributes.get("title")
    assert row.attributes["lookback_label"] == f"{s.session_id} #1"
    assert row.attributes["metadata_authored"]["fields"] == ["title", "summary"]
    assert "pending" not in row.attributes


def test_every_failed_drainer_attempt_leaves_its_failure_line(tmp_path: Path, monkeypatch) -> None:
    """Codex review 3 #10: only a malformed reply used to be recorded. A deferred attempt (the endpoint unreachable),
    a task whose source is empty and an unreadable packet each leave their line too - record id, outcome, the original
    exception's class and message, the capability model when one was built."""
    from abstractruntime.identity.authoring import drain_authoring

    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, ["Back.\n```interest\nwhy rivers bend\n```\n", "A quiet close."])
    s.llm_for_capability = None  # forms pending
    s.turn("hi")
    target = str(_rows(home_dir, "interest")[-1].subject)

    def lines_for(reason: str) -> List[Dict[str, Any]]:
        return [f for f in _failures(tmp_path) if f.get("reason") == reason and f["record_id"] == target]

    drain_authoring(home.ms, llm_for_capability=lambda name: _Capability(exc=ConnectionError("connection refused")),
                    artifacts=home.artifacts, home_dir=home_dir, entity=home.entity_id, holder="t", cap=3, window="turn")
    (deferred,) = [f for f in lines_for("unavailable") if f.get("outcome") == "defer"]  # (formation wrote its own)
    assert (deferred["outcome"], deferred["error_class"], deferred["model"]) == ("defer", "ConnectionError", "summarizer")
    assert deferred["error_message"] == "connection refused"

    # An unreadable packet (the task's target cannot be read) - its own line, its exception named.
    from datetime import datetime, timedelta, timezone

    later = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    real_select, real_claim = home.ms.tasks.select_for_window, home.ms.tasks.claim
    monkeypatch.setattr(home.ms.tasks, "select_for_window", lambda *a, **k: real_select(*a, **{**k, "now": later}))
    monkeypatch.setattr(home.ms.tasks, "claim", lambda *a, **k: real_claim(*a, **{**k, "now": later}))

    def broken_packet(task_id: str) -> Dict[str, Any]:
        raise KeyError(f"no packet for {task_id}")

    monkeypatch.setattr(home.ms, "task_packet", broken_packet)
    drain_authoring(home.ms, llm_for_capability=lambda name: _Capability(), artifacts=home.artifacts,
                    home_dir=home_dir, entity=home.entity_id, holder="t", cap=3, window="turn")
    home.close()
    unreadable = lines_for("packet_unreadable")
    assert unreadable and unreadable[0]["outcome"] == "retry" and unreadable[0]["error_class"] == "KeyError", unreadable


def test_a_task_with_no_words_to_author_from_leaves_its_failure_line(tmp_path: Path) -> None:
    from abstractruntime.identity.authoring import drain_authoring

    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, [_DIARY_GISTLESS, "A quiet close."])
    s.llm_for_capability = None  # the entry's title forms pending
    s.turn("hi")
    (entry,) = _diary_rows(home_dir)
    receipt = drain_authoring(home.ms, llm_for_capability=lambda name: _Capability(), artifacts=home.artifacts,
                              home_dir=home_dir, entity=home.entity_id, holder="t", cap=3, window="turn",
                              diary_text=lambda entry_id: "")  # the book's words cannot be read
    home.close()
    assert receipt["failed"] >= 1, receipt
    lines = [f for f in _failures(tmp_path) if f.get("reason") == "no_source" and f["record_id"] == str(entry.subject)]
    # One line per task that could not run (its title and its summary are both still to be written).
    assert sorted(f["missing"] for f in lines) == ["summary", "title"], lines
    assert all(f["outcome"] == "retry" and "no words" in f["error_message"] for f in lines), lines


def test_the_sleep_repairs_a_legacy_public_diary_records_title_and_summary(tmp_path: Path, monkeypatch) -> None:
    """Codex review 3 #9 + the operator rule (title + summary always exist, authored): a public diary record formed
    before today carries the act-frame template title and his first line as summary. The sleep's repair declares
    both pending (AbstractMemory 8235daf takes a legacy public projection's summary under formation's rule) and the
    drainer authors them in the same step; the next night finds nothing left to declare. A private projection and an
    owner-direct entry are never declared (their summaries are his words or content-free); a projection claiming a
    chain position is refused by memory, said ONCE per night with the rule memory names, nothing written around it.
    Fable review R2: this pinned memory 71de051's protected_kind refusal of the legacy public summary until 8235daf."""
    import abstractmemory
    from abstractmemory import MemoryRecordInput

    from abstractruntime.identity.chat import open_home
    from abstractruntime.identity.diary import DiaryEntry
    from abstractruntime.identity.life import build_consolidator, write_entity_state

    first_line = "The tide leaves a readable line on the sand."
    home_dir = _workspace_home(tmp_path)
    home = open_home(home_dir)
    home.diary.append_entry(DiaryEntry(  # the book's entry the legacy record projects
        entry_id="diary_legacy01", author=home.entity_id, kind="note", gist="the tide line",
        text=f"{first_line}\nIt stays until the next wave.", visibility="self",
        written_at="2026-11-08T10:00:00+00:00"))

    def form(key: str, *, title: str, digest: str, attributes: Dict[str, Any], provenance: Dict[str, Any]) -> str:
        (gid,) = home.ms.remember_many([MemoryRecordInput(kind="diary", title=title, digest=digest,
                                                          attributes=attributes, provenance=provenance)],
                                       scope="diary", owner_id=home.entity_id, idempotency_key=key, turn_id=key)
        return str(gid)

    rid = form("legacy-diary", title="Diary entry (note) — 2026-11-08: the tide line", digest=first_line,
               attributes={"entry_id": "diary_legacy01", "diary_type": "note", "diary_kind": "note",
                           "written_at": "2026-11-08T10:00:00+00:00"},
               provenance={"source": "diary-projection", "entry_id": "diary_legacy01"})
    private = form("legacy-private", title="Diary entry (private) — 2026-11-08 [vate01]",
                   digest="Wrote a private diary entry.",
                   attributes={"entry_id": "diary_private01", "written_at": "2026-11-08T11:00:00+00:00",
                               "private": True},
                   provenance={"source": "diary-projection", "entry_id": "diary_private01"})
    owner_direct = form("legacy-owner-direct", title="A note in his own hand", digest="His own words, the body.",
                        attributes={"diary_type": "note"}, provenance={"source": "owner-direct"})
    chained = [form(f"legacy-chained-{i}", title=f"A projection claiming chain link {i}", digest=f"Link {i} words.",
                    attributes={"entry_id": f"diary_chained0{i}", "diary_type": "note", "prev_entry_hash": ""},
                    provenance={"source": "diary-projection", "entry_id": f"diary_chained0{i}"}) for i in (1, 2)]
    home.close()
    untouched = {gid: (str(r.object), r.attributes) for r in _rows(home_dir, "diary")
                 for gid in [str(r.subject)] if gid in (private, owner_direct, *chained)}
    assert len(untouched) == 4, untouched
    monkeypatch.setattr(abstractmemory, "sleep_pass", lambda *a, **k: {"maintenance": {}, "dream": {}})

    def night() -> tuple:
        write_entity_state(home_dir, "asleep", reason="self-elected sleep: full", mode="dreaming", written_by="self", fence=LifecycleFence.admit(home_dir))
        lines: List[str] = []
        result = build_consolidator(home_dir, out=lines.append, llm_for_capability=lambda name: _Capability(), fence=LifecycleFence.admit(home_dir))()
        return result["authoring"], lines

    receipt, lines = night()
    assert receipt["declared"] == 2 and receipt["authored"] == 2 and receipt["refused"] == 2, receipt
    row = [r for r in _rows(home_dir, "diary") if str(r.subject) == rid][-1]
    assert (row.attributes["title"], str(row.object)) == (TITLE, SUMMARY), (row.attributes.get("title"), row.object)
    assert "pending" not in row.attributes
    assert row.attributes["understanding"]["author"] == {"kind": "capability-model", "id": "summaries"}
    assert row.attributes["digest_method"] == "authored-v1"
    (task,) = [t for t in _tasks(home_dir, "digest.author") if t["target_id"] == rid]
    assert task["status"] == "done" and task["cause"]["cause"] == "repair" and task["cause"]["declared"] is True, task
    home = open_home(home_dir)
    try:
        v0 = home.ms.versions(rid)[0]
    finally:
        home.close()
    assert (v0["version"], v0["text"]) == (0, first_line), "his first line is kept as the provisional version 0"
    # Private, owner-direct and chained: nothing declared, nothing written, no task.
    now = {str(r.subject): (str(r.object), r.attributes) for r in _rows(home_dir, "diary")}
    assert {gid: now[gid] for gid in untouched} == untouched
    owed = [t for kind in ("digest.author", "title.author") for t in _tasks(home_dir, kind)
            if t["target_id"] in untouched]
    assert owed == [], owed
    refused = [ln for ln in lines if ln.startswith("#REPAIR_REFUSED")]
    assert len(refused) == 1, lines
    assert "2 legacy public diary summary(ies)" in refused[0] and "protected_kind" in refused[0]
    assert "hash_chained" in refused[0], refused[0]

    # The next night: the authored summary is no stand-in - nothing declared, nothing re-authored; the refusal is
    # said once again.
    receipt, lines = night()
    assert (receipt["declared"], receipt["authored"], receipt["refused"]) == (0, 0, 2), receipt
    again = [r for r in _rows(home_dir, "diary") if str(r.subject) == rid][-1]
    assert (again.attributes["title"], str(again.object)) == (TITLE, SUMMARY)
    assert len([t for t in _tasks(home_dir, "digest.author") if t["target_id"] == rid]) == 1
    assert len([ln for ln in lines if ln.startswith("#REPAIR_REFUSED")]) == 1, lines
