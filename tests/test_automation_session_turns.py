"""select_session_turns: the one definition of a session's turns (automations C3/E).

Turns = parent-less runs + automation occurrences; never descendants,
controllers, internal runs, legacy scheduled wrappers or drafts. History
bundles and session replay (growing-mode seeding) both read it, so an
automation's occurrences show up as conversation turns everywhere.
"""

from __future__ import annotations

import pytest

from abstractruntime import RunState, RunStatus, session_chat_messages
from abstractruntime.history_bundle import _best_effort_session_turns
from abstractruntime.session_turns import select_session_turns
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore

AUTO = "auto-1"
SID = "s-auto"


def make_store(kind, tmp_path):
    if kind == "memory":
        return InMemoryRunStore()
    if kind == "json":
        return JsonFileRunStore(tmp_path / "runs")
    return SqliteRunStore(SqliteDatabase(tmp_path / "runs.sqlite"))


def _save(store, run_id, minute, *, prompt="", answer="", parent=None, meta=None, workflow_id="wf",
          status=RunStatus.COMPLETED, lifecycle=None, session_id=SID):
    vars_obj = {"prompt": prompt, "context": {"messages": []}, "_meta": dict(meta or {})}
    if lifecycle:
        vars_obj["_run_lifecycle"] = lifecycle
    ts = f"2026-09-27T10:{minute:02d}:00+00:00"
    store.save(RunState(run_id=run_id, workflow_id=workflow_id, status=status, current_node="n", vars=vars_obj,
                        output={"answer": answer} if answer else {}, session_id=session_id, parent_run_id=parent,
                        created_at=ts, updated_at=ts))


def _occ(index, *, attempt=1, role="occurrence"):
    return {"occurrence": {"automation_id": AUTO, "occurrence_index": index, "attempt": attempt,
                           "role": role, "session_kind": "automation"}}


def populate(store):
    _save(store, AUTO, 0, meta={"automation": {"title": "t"}})                         # controller: never a turn
    _save(store, "chat-1", 1, prompt="hello", answer="hi")                             # a chat turn
    _save(store, "occ-1", 2, prompt="tick 1", answer="mem 41%", parent=AUTO, meta=_occ(1))
    _save(store, "occ-1-desc", 3, prompt="sub", answer="x", parent="occ-1", meta=_occ(1, role="descendant"))
    _save(store, "occ-2a1", 4, prompt="tick 2", parent=AUTO, meta=_occ(2), status=RunStatus.FAILED)
    _save(store, "occ-2a2", 5, prompt="tick 2", answer="mem 43%", parent=AUTO, meta=_occ(2, attempt=2))
    _save(store, "chat-2", 6, prompt="why up?", answer="because")
    _save(store, "occ-3", 7, prompt="tick 3", answer="mem 40%", parent=AUTO, meta=_occ(3))
    _save(store, "legacy", 8, prompt="old", answer="old", workflow_id="scheduled:x", meta={"schedule": {"kind": "scheduled_run"}})
    _save(store, "internal", 9, prompt="m", answer="m", workflow_id="__session_memory__")
    _save(store, "draft", 10, prompt="d", answer="d", lifecycle={"purpose": "draft_test"})


def ids(runs):
    return [r.run_id for r in runs]


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite"])
def test_turns_are_roots_plus_occurrences(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    populate(store)
    assert ids(select_session_turns(store, SID)) == ["chat-1", "occ-1", "occ-2a2", "chat-2", "occ-3"]
    assert ids(select_session_turns(store, SID, include_occurrences=False)) == ["chat-1", "chat-2"]
    assert ids(select_session_turns(store, SID, include_drafts=True))[-1] == "draft"
    assert ids(select_session_turns(store, SID, through_occurrence=2)) == ["chat-1", "occ-1", "occ-2a2"]
    assert ids(select_session_turns(store, SID, automation_id="other")) == ["chat-1", "chat-2"]
    assert ids(select_session_turns(store, SID, limit=2)) == ["chat-2", "occ-3"]
    until = 1790503560000  # 2026-09-27T10:06:00Z
    assert ids(select_session_turns(store, SID, until_ms=until)) == ["chat-1", "occ-1", "occ-2a2", "chat-2"]


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite"])
def test_history_bundle_turns_include_occurrences(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    populate(store)
    turns = _best_effort_session_turns(run_store=store, ledger_store=InMemoryLedgerStore(), artifact_store=None,
                                       session_id=SID, limit=50, include_stats=False, include_artifacts=False)
    assert [t["run_id"] for t in turns] == ["chat-1", "occ-1", "occ-2a2", "chat-2", "occ-3"]
    occ = next(t for t in turns if t["run_id"] == "occ-2a2")
    assert (occ["kind"], occ["automation_id"], occ["occurrence_index"]) == ("occurrence", AUTO, 2)


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite"])
def test_growing_mode_seed_replays_prior_occurrences(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    populate(store)
    messages = session_chat_messages(run_store=store, ledger_store=InMemoryLedgerStore(), session_id=SID)
    assert [m["content"] for m in messages] == [
        "hello", "hi", "tick 1", "mem 41%", "tick 2", "mem 43%", "why up?", "because", "tick 3", "mem 40%",
    ]


def test_legacy_scheduled_prefix_alone_is_not_a_turn(tmp_path) -> None:
    store = make_store("sqlite", tmp_path)
    _save(store, "chat", 1, prompt="p", answer="a")
    _save(store, "wrapper", 2, prompt="p", answer="a", workflow_id="scheduled:abc")  # no _meta.schedule
    assert ids(select_session_turns(store, SID)) == ["chat"]
