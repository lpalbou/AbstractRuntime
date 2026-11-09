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
