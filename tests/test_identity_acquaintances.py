"""Acquaintances: the entities and humans an entity or a session met (derived, never stored).

Each rule has a test that goes red when it is removed: the self exclusion, turn roots only, reserved runs,
person origins only, task `added` events only, the namespace rule and what is not counted, one interaction
per run, first/last seen, the order, the session scope and the identity ref.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.identity import Acquaintances, entity_acquaintances, session_acquaintances
from abstractruntime.identity.acquaintances import (
    NOT_COUNTED_NO_ACTOR,
    NOT_COUNTED_NO_NAMESPACE,
    NOT_COUNTED_OTHER_NAMESPACE,
)
from abstractruntime.identity.digital_identity import identity_ref, record_default_identity, stamp_run_identity_ref
from abstractruntime.identity.track_record import TrackRecordError
from abstractruntime.storage.in_memory import InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore

pytestmark = pytest.mark.basic

SELF = "entity:castor"
MANIFEST = {"entity_id": SELF, "slug": "castor", "home_id": "home-1", "spark_hash": "a" * 64}
VISIT = "visit/entity-visit-react"


def _ts(minute: int) -> str:
    return f"2026-11-04T10:{minute:02d}:00+00:00"


def _run(
    run_id: str,
    *,
    minute: int,
    session: str = "s1",
    workflow: str = "entity-visit-react",
    participants: Optional[List[str]] = None,
    origin: Optional[Dict[str, Any]] = None,
    parent: Optional[str] = None,
    ends: Optional[int] = None,
) -> RunState:
    vars_obj: Dict[str, Any] = {"_meta": {}, "prompt": "entity:ghost person:ghost said hello"}  # text is never read
    if participants is not None:
        vars_obj["_runtime"] = {"entity": {"phase": "visit", "participants": participants}}
    if origin is not None:
        vars_obj["_meta"]["origin"] = origin
    return RunState(
        run_id=run_id, workflow_id=workflow, status=RunStatus.COMPLETED, current_node="end", vars=vars_obj,
        created_at=_ts(minute), updated_at=_ts(ends if ends is not None else minute + 1), session_id=session, parent_run_id=parent,
    )


def _person(actor: Optional[str]) -> Dict[str, Any]:
    return {"kind": "person", "actor": actor, "client": "console", "ref": {}}


def _store(kind: str, tmp_path: Path):
    if kind == "sqlite":
        return SqliteRunStore(SqliteDatabase(tmp_path / "runtime_castor.sqlite3"))
    if kind == "json":
        return JsonFileRunStore(tmp_path / "runs")
    return InMemoryRunStore()


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "castor"
    home.mkdir(exist_ok=True)
    (home / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    events = [
        {"event": "added", "task_id": "t1", "title": "x", "at": _ts(14), "by": "person:ana"},
        {"event": "status", "task_id": "t1", "status": "done", "at": _ts(15), "by": "operator"},
        {"event": "added", "task_id": "t2", "title": "y", "at": _ts(16), "by": SELF},
    ]
    (home / "task_inbox.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return home


def _fixture(store) -> None:
    for run in [
        _run("v1", minute=1, ends=30, participants=["person:laurent", SELF]),  # a long visit: ends last
        _run("m1", minute=5, participants=["entity:pollux@home-9", "person:laurent", SELF]),  # a meet leg
        _run("v3", minute=8, participants=["person:ana", "agent:ariadne", "nobody", SELF, "person:ana"]),
        _run("v2", minute=10, participants=["person:laurent", SELF]),
        _run("c1", minute=12, workflow="chat-flow", origin=_person("default:laurent")),
        _run("c2", minute=13, workflow="chat-flow", origin=_person(None)),
        _run("a1", minute=20, workflow="chat-flow", origin={"kind": "agent", "actor": "default:bot", "client": None, "ref": {"origin_run_id": "x"}}),
        # never read: a sub-run carrying its root's stamp, and the reserved session memory run
        _run("v1-sub", minute=2, parent="v1", participants=["person:laurent", "entity:intruder", SELF]),
        _run("session_memory_s1", minute=0, workflow="__session_memory__", participants=["person:mallory"]),
    ]:
        store.save(run)


def _view(acq: Acquaintances) -> Dict[str, Any]:
    d = acq.to_dict()
    d.pop("computed_at")
    return d


@pytest.mark.parametrize("backend", ["sqlite", "json", "memory"])
def test_entity_acquaintances(tmp_path: Path, backend: str) -> None:
    store = _store(backend, tmp_path)
    _fixture(store)
    acq = entity_acquaintances(_home(tmp_path), store)
    assert acq.subject == "entity:castor"
    assert [a.to_dict() for a in acq.entities] == [
        {"id": "entity:pollux@home-9", "first_seen": _ts(5), "last_seen": _ts(6), "interactions": 1, "kinds": {VISIT: 1}},
    ]
    assert [a.to_dict() for a in acq.humans] == [
        {"id": "person:laurent", "first_seen": _ts(1), "last_seen": _ts(30), "interactions": 3, "kinds": {VISIT: 3}},
        {"id": "person:ana", "first_seen": _ts(8), "last_seen": _ts(14), "interactions": 2,
         "kinds": {VISIT: 1, "work/task_inbox": 1}},
        {"id": "default:laurent", "first_seen": _ts(12), "last_seen": _ts(13), "interactions": 1, "kinds": {"chat/chat-flow": 1}},
    ]
    assert dict(acq.not_counted) == {NOT_COUNTED_NO_ACTOR: 1, NOT_COUNTED_NO_NAMESPACE: 1, NOT_COUNTED_OTHER_NAMESPACE: 1}
    assert acq.as_of == _ts(30)


def test_deterministic_across_reads_and_backends(tmp_path: Path) -> None:
    views = []
    for backend in ("sqlite", "json", "memory"):
        sub = tmp_path / backend
        sub.mkdir()
        store = _store(backend, sub)
        _fixture(store)
        home = _home(sub)
        first, second = entity_acquaintances(home, store), entity_acquaintances(home, store)
        assert _view(first) == _view(second)
        views.append(_view(first))
    assert views[0] == views[1] == views[2]


def test_no_records_no_acquaintances(tmp_path: Path) -> None:
    home = tmp_path / "castor"
    home.mkdir()
    (home / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    acq = entity_acquaintances(home, InMemoryRunStore())
    assert (acq.entities, acq.humans, dict(acq.not_counted), acq.as_of) == ((), (), {}, None)


def test_identity_ref_of_the_subject(tmp_path: Path) -> None:
    store = InMemoryRunStore()
    home = _home(tmp_path)
    assert entity_acquaintances(home, store).identity_ref is None
    record_default_identity(home, entity_id=SELF, gateway_default_voice=None, provenance="test")
    assert entity_acquaintances(home, store).identity_ref == identity_ref("entity:castor", 1)


def test_session_scope_and_ref(tmp_path: Path) -> None:
    store = InMemoryRunStore()
    first = _run("r1", minute=1, session="sx", workflow="chat-flow", origin=_person("default:ana"))
    summon = _run("r2", minute=3, session="sx", workflow="chat-flow", participants=["person:ana", SELF], origin=_person("default:ana"))
    stamp_run_identity_ref(first.vars, identity_ref("session:sx", 1))
    stamp_run_identity_ref(summon.vars, identity_ref("session:sx", 2))
    for run in (first, summon, _run("other", minute=5, session="sy", workflow="chat-flow", origin=_person("default:zoe"))):
        store.save(run)
    acq = session_acquaintances("sx", store)
    assert acq.subject == "session:sx"
    assert acq.identity_ref == identity_ref("session:sx", 2)
    assert [(a.id, a.interactions) for a in acq.humans] == [("default:ana", 2), ("person:ana", 1)]
    assert [(a.id, a.interactions) for a in acq.entities] == [(SELF, 1)]  # a session has no self to exclude
    with pytest.raises(TrackRecordError):
        session_acquaintances(" ", store)


def test_bad_participants_shape_fails_loudly(tmp_path: Path) -> None:
    store = InMemoryRunStore()
    run = _run("v", minute=1, participants=[])
    run.vars["_runtime"]["entity"]["participants"] = "person:laurent"
    store.save(run)
    with pytest.raises(TrackRecordError, match="not a list"):
        entity_acquaintances(_home(tmp_path), store)
