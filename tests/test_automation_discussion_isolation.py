"""Discuss (contract B, operator rulings 2026-09-27): a forked ROOT session that never writes back.

A discussion forked at occurrence N is seeded once with the automation's WHOLE
conversation through N (occurrences 1..N, whatever the context mode), works in
its OWN writable workspace, and sees the automation's workspace mounted
read-only alongside. Nothing it does reaches the automation's session, state,
ledger or workspace.
"""

from __future__ import annotations

import os

import pytest

from automation_harness import Clock, at, automation_state, children, create, drive, make_runtime, make_stores
from abstractruntime.automations import AutomationError, start_discussion
from abstractruntime.automations.ledger import automation_records
from abstractruntime.automations.service import automation_timeline_messages
from abstractruntime.core.models import RunStatus
from abstractruntime.session_history import session_chat_messages
from abstractruntime.session_turns import select_session_turns
from abstractruntime.utils.workspace_paths import READ_ONLY_KEY, READ_ONLY_PATHS_KEY, read_only_paths

TICKS = ("2026-01-01T00:02:00+00:00", "2026-01-01T00:04:00+00:00", "2026-01-01T00:06:00+00:00")


@pytest.fixture
def env(tmp_path, monkeypatch, request):
    return make_runtime(*make_stores(request.param, tmp_path)), Clock(monkeypatch), tmp_path


def _tick(runtime, run_id):
    run = runtime.get_state(run_id)
    return runtime.tick(workflow=runtime.workflow_registry.get(run.workflow_id), run_id=run_id)


def _workspaces(tmp_path):
    auto_ws, own_ws = tmp_path / "automation-ws", tmp_path / "discussion-ws"
    auto_ws.mkdir()
    own_ws.mkdir()
    return str(auto_ws), str(own_ws)


def _four_occurrences(runtime, clock, aid):
    drive(runtime, aid)
    for now in TICKS:
        at(runtime, clock, aid, now)
    kids = children(runtime, aid)
    assert len(kids) == 4 and all(k.status == RunStatus.COMPLETED for k in kids)
    return kids


def _pairs(seed):
    return [(seed[i]["metadata"]["occurrence_index"], seed[i + 1]["content"]) for i in range(0, len(seed), 2)]


@pytest.mark.parametrize("env", ["json", "sqlite"], indirect=True)
def test_independent_discussion_carries_the_whole_timeline_through_n(env):
    runtime, clock, tmp_path = env
    auto_ws, own_ws = _workspaces(tmp_path)
    aid = create(runtime, clock, workspace_root=auto_ws)  # independent: one session per occurrence
    kids = _four_occurrences(runtime, clock, aid)

    at3 = start_discussion(runtime, automation_id=aid, occurrence_index=3, request_id="d3", prompt="Trend?",
                           workspace_root=own_ws)
    seed = runtime.get_state(at3["run_id"]).vars["_meta"]["discussion"]["seed_messages"]
    assert _pairs(seed) == [(i + 1, kids[i].output["response"]) for i in range(3)]  # 1..3, in order, never 4
    assert [m["content"] for m in seed[2::2]] == [kids[1].vars["prompt"], kids[2].vars["prompt"]]
    head = seed[0]["content"]
    assert head.startswith('[Automation "Memory watch": 3 occurrence(s) through occurrence 3, showing the last 3.')
    assert f"mounted READ-ONLY at {auto_ws}" in head and f"your own workspace {own_ws} is writable" in head
    assert head.endswith(kids[0].vars["prompt"])

    at1 = start_discussion(runtime, automation_id=aid, occurrence_index=1, request_id="d1", prompt="Trend?",
                           workspace_root=own_ws)
    seed1 = runtime.get_state(at1["run_id"]).vars["_meta"]["discussion"]["seed_messages"]
    assert _pairs(seed1) == [(1, kids[0].output["response"])]

    # The discussion runs with that history.
    assert _tick(runtime, at3["run_id"]).output["response"] == "echo:Trend? | history=6"


@pytest.mark.parametrize("env", ["json", "sqlite"], indirect=True)
def test_growing_discussion_seed_matches_the_automation_session(env):
    runtime, clock, tmp_path = env
    auto_ws, own_ws = _workspaces(tmp_path)
    aid = create(runtime, clock, mode="growing", workspace_root=auto_ws)
    _four_occurrences(runtime, clock, aid)
    started = start_discussion(runtime, automation_id=aid, occurrence_index=2, request_id="g", prompt="?",
                               workspace_root=own_ws)
    seed = runtime.get_state(started["run_id"]).vars["_meta"]["discussion"]["seed_messages"]
    session = session_chat_messages(run_store=runtime.run_store, ledger_store=runtime.ledger_store,
                                    session_id=f"automation:{aid}", automation_id=aid, through_occurrence=2, strict=True)
    contents = [m["content"] for m in seed]
    contents[0] = contents[0].split("\n", 1)[1]  # drop the summary line
    assert contents == [m["content"] for m in session]


@pytest.mark.parametrize("env", ["json"], indirect=True)
def test_the_seed_keeps_failures_and_drops_the_oldest_under_the_budget(env):
    runtime, clock, tmp_path = env
    auto_ws, own_ws = _workspaces(tmp_path)
    aid = create(runtime, clock, workflow_id="flaky", workspace_root=auto_ws,
                 input_data={"prompt": "x" * 7000, "fail_until": 99}, retry={"max_attempts": 1})
    drive(runtime, aid)
    for now in TICKS:
        at(runtime, clock, aid, now)
    seed = automation_timeline_messages(runtime, aid, through_occurrence=4, workspace_root=own_ws,
                                        mounted_workspace=auto_ws)
    # Four ~7 000-char turns do not fit 24 000 chars: the oldest go first.
    assert [p[0] for p in _pairs(seed)] == [2, 3, 4]
    assert "4 occurrence(s) through occurrence 4, showing the last 3" in seed[0]["content"]
    assert seed[1]["content"].startswith("(This occurrence failed after 1 attempt(s): ")


@pytest.mark.parametrize("env", ["json", "sqlite"], indirect=True)
def test_discussion_runs_in_its_own_workspace_with_the_automation_mounted_read_only(env):
    runtime, clock, tmp_path = env
    auto_ws, own_ws = _workspaces(tmp_path)
    aid = create(runtime, clock, workspace_root=auto_ws)
    drive(runtime, aid)
    before_state = dict(automation_state(runtime, aid))
    before_records = len(automation_records(runtime.ledger_store, aid))
    before_turns = [t.run_id for t in select_session_turns(runtime.run_store, children(runtime, aid)[0].session_id)]

    started = start_discussion(runtime, automation_id=aid, occurrence_index=1, request_id="w", prompt="Plot it",
                               workspace_root=own_ws)
    assert started["session_id"] == f"discussion-session:{started['run_id']}" and started["session_kind"] == "discussion"
    disc = runtime.get_state(started["run_id"])
    assert disc.parent_run_id is None
    assert disc.vars["workspace_root"] == own_ws
    assert disc.vars["_runtime"][READ_ONLY_PATHS_KEY] == [auto_ws]
    assert read_only_paths(disc.vars) == (os.path.realpath(auto_ws),)
    assert disc.vars["workspace_access_mode"] == "workspace_or_allowed" and auto_ws in disc.vars["workspace_allowed_paths"]
    assert READ_ONLY_KEY not in disc.vars and READ_ONLY_KEY not in disc.vars["_runtime"]  # own workspace writable
    assert disc.vars["_meta"]["discussion"]["mounted_workspace"] == auto_ws

    assert _tick(runtime, started["run_id"]).status == RunStatus.COMPLETED
    assert automation_state(runtime, aid) == before_state
    assert len(automation_records(runtime.ledger_store, aid)) == before_records
    assert [t.run_id for t in select_session_turns(runtime.run_store, children(runtime, aid)[0].session_id)] == before_turns

    with pytest.raises(AutomationError) as exc:
        start_discussion(runtime, automation_id=aid, occurrence_index=1, request_id="same-ws", prompt="?",
                         workspace_root=auto_ws)
    assert (exc.value.reason_code, exc.value.field) == ("invalid_request", "workspace_root")


@pytest.mark.parametrize("env", ["json"], indirect=True)
def test_discussing_an_unknown_occurrence_is_refused(env):
    runtime, clock, tmp_path = env
    auto_ws, own_ws = _workspaces(tmp_path)
    aid = create(runtime, clock, workspace_root=auto_ws)
    drive(runtime, aid)
    with pytest.raises(AutomationError) as exc:
        start_discussion(runtime, automation_id=aid, occurrence_index=9, request_id="d", prompt="?", workspace_root=own_ws)
    assert exc.value.reason_code == "occurrence_not_found"


@pytest.mark.parametrize("env", ["json", "sqlite"], indirect=True)
def test_a_later_turn_in_the_discussion_session_is_anchored_to_the_root(env):
    runtime, clock, tmp_path = env
    auto_ws, own_ws = _workspaces(tmp_path)
    aid = create(runtime, clock, workspace_root=auto_ws)
    drive(runtime, aid)
    started = start_discussion(runtime, automation_id=aid, occurrence_index=1, request_id="d2", prompt="?",
                               workspace_root=own_ws)
    _tick(runtime, started["run_id"])
    later_id = runtime.start(workflow=runtime.workflow_registry.get("echo"), vars={"prompt": "And then?"},
                             session_id=started["session_id"])
    later = runtime.get_state(later_id)
    assert later.vars["_meta"]["discussion"]["discussion_root_run_id"] == started["run_id"]
    assert "seed_messages" not in later.vars["_meta"]["discussion"]
    history = session_chat_messages(run_store=runtime.run_store, ledger_store=runtime.ledger_store,
                                    session_id=started["session_id"], strict=True)
    assert len(history) == 4  # seed (1 occurrence) + the first discussion turn


@pytest.mark.parametrize("env", ["json"], indirect=True)
def test_a_later_turn_keeps_the_writable_workspace_and_the_read_only_mount(env):
    runtime, clock, tmp_path = env
    auto_ws, own_ws = _workspaces(tmp_path)
    aid = create(runtime, clock, workspace_root=auto_ws)
    drive(runtime, aid)
    started = start_discussion(runtime, automation_id=aid, occurrence_index=1, request_id="d3", prompt="?",
                               workspace_root=own_ws)
    later = runtime.get_state(runtime.start(workflow=runtime.workflow_registry.get("echo"), vars={"prompt": "x"},
                                            session_id=started["session_id"]))
    assert later.vars["workspace_root"] == own_ws
    assert later.vars["_runtime"][READ_ONLY_PATHS_KEY] == [auto_ws]
    assert READ_ONLY_KEY not in later.vars


@pytest.mark.parametrize("env", ["json", "sqlite"], indirect=True)
def test_request_ids_are_scoped_to_the_automation(env):
    runtime, clock, tmp_path = env
    ws_a, own_ws = _workspaces(tmp_path)
    ws_b = tmp_path / "b"
    ws_b.mkdir()
    a = create(runtime, clock, workspace_root=ws_a)
    b = create(runtime, clock, workspace_root=str(ws_b))
    drive(runtime, a)
    drive(runtime, b)
    da = start_discussion(runtime, automation_id=a, occurrence_index=1, request_id="disc-1", prompt="?",
                          workspace_root=own_ws, actor_id="ana")
    db = start_discussion(runtime, automation_id=b, occurrence_index=1, request_id="disc-1", prompt="?",
                          workspace_root=own_ws, actor_id="ana")
    assert da["session_id"] != db["session_id"] and da["run_id"] != db["run_id"]
    root_b = runtime.get_state(db["run_id"])
    assert root_b.vars["_meta"]["discussion"]["automation_id"] == b
    assert root_b.vars["_meta"]["discussion"]["mounted_workspace"] == str(ws_b)
    assert root_b.vars["_meta"]["discussion"]["seed_messages"]
    assert root_b.actor_id == "ana"
    assert start_discussion(runtime, automation_id=a, occurrence_index=1, request_id="disc-1", prompt="?",
                            workspace_root=own_ws, actor_id="ana") == da
    with pytest.raises(AutomationError) as exc:
        start_discussion(runtime, automation_id=a, occurrence_index=1, request_id="disc-1", prompt="something else",
                         workspace_root=own_ws)
    assert (exc.value.reason_code, exc.value.field) == ("identity_conflict", "request_id")


@pytest.mark.parametrize("env", ["json"], indirect=True)
def test_host_protection_allows_both_discussion_roots_and_nothing_else(env):
    from abstractruntime.integrations.abstractcore.workspace_scoped_tools import WorkspaceScope, rewrite_tool_arguments
    from abstractruntime.utils.workspace_paths import BUILTIN_ALLOW_KEY, BUILTIN_DENY_KEY

    runtime, clock, tmp_path = env
    data_dir = tmp_path / "gateway-data"
    auto_ws = data_dir / "workspaces" / "automation"
    own_ws = data_dir / "workspaces" / "discussion"
    secrets = data_dir / "secrets"
    for d in (auto_ws, own_ws, secrets):
        d.mkdir(parents=True)
    aid = create(runtime, clock, workspace_root=str(auto_ws),
                 input_data={"prompt": "p", BUILTIN_DENY_KEY: [str(data_dir)], BUILTIN_ALLOW_KEY: [str(auto_ws)]})
    drive(runtime, aid)
    started = start_discussion(runtime, automation_id=aid, occurrence_index=1, request_id="h", prompt="?",
                               workspace_root=str(own_ws))
    disc = runtime.get_state(started["run_id"])
    assert disc.vars[BUILTIN_DENY_KEY] == [str(data_dir)]  # unchanged
    assert disc.vars[BUILTIN_ALLOW_KEY] == [str(own_ws), str(auto_ws)]
    scope = WorkspaceScope.from_input_data(disc.vars)

    def write(path):
        return rewrite_tool_arguments(tool_name="write_file", args={"file_path": str(path), "content": "x"}, scope=scope)

    assert write(own_ws / "analysis.md")["file_path"].endswith("analysis.md")  # own root: allowed
    with pytest.raises(ValueError):
        write(auto_ws / "state.json")  # the mount: read-only
    with pytest.raises(ValueError):
        write(secrets / "token")  # elsewhere in the data dir: denied
