"""The persisted discussion root anchors its session (automations contract B, amendment 4).

`Runtime.start` resolves every ROOT start that names a session through
`session_attribution`; in a discussion session it stamps the root's
`_meta.discussion` (no seed), its workspace root and a read-only workspace over
whatever the caller passed, and refuses the start when the lookup fails.
Ordinary sessions are untouched.
"""

from __future__ import annotations

import json

import pytest

from abstractruntime import Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.core.run_attribution import SessionAttributionError, session_attribution
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore

WF = WorkflowSpec(workflow_id="wf", entry_node="n", nodes={"n": lambda r, c: StepPlan(node_id="n", complete_output={})})
DISC = {"automation_id": "auto-1", "occurrence_index": 2, "revision": 1, "seed_run_id": "occ-2",
        "request_id": "d1", "discussion_root_run_id": "disc-root"}


def make_store(kind, tmp_path):
    if kind == "memory":
        return InMemoryRunStore()
    if kind == "json":
        return JsonFileRunStore(tmp_path / "runs")
    return SqliteRunStore(SqliteDatabase(tmp_path / "runs.sqlite"))


def _root(store, ws, *, meta=None, mount=None, whole_root_read_only=True):
    vars_ = {"workspace_root": str(ws),
             "_meta": {"discussion": meta if meta is not None else {**DISC, "seed_messages": [{"role": "user", "content": "x"}]}}}
    if whole_root_read_only:
        vars_["workspace_read_only"] = True
    if mount is not None:  # the mount model (operator ruling 2026-09-27)
        vars_.update({"workspace_access_mode": "workspace_or_allowed", "workspace_allowed_paths": [str(mount)],
                      "_runtime": {"workspace_read_only_paths": [str(mount)]}})
    store.save(RunState(run_id="disc-root", workflow_id="wf", status=RunStatus.COMPLETED, current_node="n",
                        session_id="disc-s", vars=vars_))


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite"])
def test_a_caller_cannot_start_a_writable_turn_in_a_discussion_session(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    _root(store, ws)
    rt = Runtime(run_store=store, ledger_store=InMemoryLedgerStore())
    rid = rt.start(workflow=WF, session_id="disc-s", vars={
        "prompt": "go on",
        "workspace_root": str(tmp_path / "elsewhere"),
        "workspace_read_only": False,
        "_runtime": {"workspace_read_only": False},
        "_meta": {"discussion": {"discussion_root_run_id": "forged"}},
    })
    vars_ = store.load(rid).vars
    assert vars_["workspace_read_only"] is True  # the root carries the whole-root flag
    assert vars_["workspace_root"] == str(ws)
    assert vars_["_meta"]["discussion"] == DISC  # the root's provenance, without the seed
    assert vars_["prompt"] == "go on"
    assert session_attribution(store, "disc-s")["kind"] == "discussion"


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_a_start_in_an_ordinary_session_is_untouched(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    store.save(RunState(run_id="chat-1", workflow_id="wf", status=RunStatus.COMPLETED, current_node="n",
                        session_id="chat-s", vars={"prompt": "hi"}))
    rt = Runtime(run_store=store, ledger_store=InMemoryLedgerStore())
    request = {"prompt": "again", "workspace_root": str(tmp_path), "_runtime": {"x": 1}}
    in_session = store.load(rt.start(workflow=WF, session_id="chat-s", vars=json.loads(json.dumps(request)))).vars
    fresh = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    baseline = fresh.get_state(fresh.start(workflow=WF, session_id="chat-s", vars=json.loads(json.dumps(request)))).vars
    assert json.dumps(in_session, sort_keys=True) == json.dumps(baseline, sort_keys=True)
    assert session_attribution(store, "chat-s") == {"kind": "chat"}
    assert session_attribution(store, "never-used") is None


@pytest.mark.parametrize("broken", ["no_root_id", "root_missing"])
def test_a_failed_lookup_refuses_the_start(broken, tmp_path) -> None:
    store = make_store("sqlite", tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    if broken == "no_root_id":
        _root(store, ws, meta={k: v for k, v in DISC.items() if k != "discussion_root_run_id"})
    else:
        store.save(RunState(run_id="disc-2", workflow_id="wf", status=RunStatus.COMPLETED, current_node="n",
                            session_id="disc-s", vars={"_meta": {"discussion": dict(DISC)}}))  # root never saved
    rt = Runtime(run_store=store, ledger_store=InMemoryLedgerStore())
    before = {r["run_id"] for r in store.list_run_index(limit=100)}
    with pytest.raises(SessionAttributionError) as info:
        rt.start(workflow=WF, session_id="disc-s", vars={"prompt": "?"})
    assert info.value.reason_code == "session_attribution_failed"
    assert {r["run_id"] for r in store.list_run_index(limit=100)} == before


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite"])
def test_later_turns_keep_the_roots_writable_workspace_and_its_mount(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    own = tmp_path / "own"
    own.mkdir()
    mount = tmp_path / "automation-ws"
    mount.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    _root(store, own, mount=mount, whole_root_read_only=False)
    rt = Runtime(run_store=store, ledger_store=InMemoryLedgerStore())
    rid = rt.start(workflow=WF, session_id="disc-s", vars={
        "workspace_root": str(mount), "workspace_access_mode": "all_except_ignored", "workspace_allowed_paths": [],
        "_runtime": {"workspace_read_only_paths": [str(extra)]},
    })
    vars_ = store.load(rid).vars
    assert vars_["workspace_root"] == str(own)
    assert vars_["workspace_access_mode"] == "workspace_or_allowed"
    assert vars_["workspace_allowed_paths"] == [str(mount)]
    assert vars_["_runtime"]["workspace_read_only_paths"] == [str(mount), str(extra)]  # caller may only add
    assert "workspace_read_only" not in vars_
