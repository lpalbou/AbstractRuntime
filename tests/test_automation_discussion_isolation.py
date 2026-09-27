"""Discuss (contract B, C4): a forked ROOT session seeded once, never writing back.

The discussion runs the occurrence's workflow in its own durable session, seeded
from the automation's conversation through the chosen occurrence, on the
occurrence's workspace mounted read-only. Later turns of the discussion session
append only there; the automation's session, state and ledger never change.
"""

from __future__ import annotations

import pytest

import abstractruntime.core.run_attribution as run_attribution
from automation_harness import Clock, at, automation_state, children, create, drive, make_runtime, make_stores
from abstractruntime.automations import AutomationError, start_discussion
from abstractruntime.automations.ledger import automation_records
from abstractruntime.core.models import RunStatus
from abstractruntime.integrations.abstractcore.workspace_scoped_tools import WorkspaceScope
from abstractruntime.session_history import session_chat_messages
from abstractruntime.session_turns import select_session_turns

# Contract B amendment 4: `Runtime.start` resolves a root start in a discussion
# session through `session_attribution(run_store, session_id)` and restamps the
# discussion provenance and the read-only workspace over caller values.
ANCHOR = hasattr(run_attribution, "session_attribution")
needs_anchor = pytest.mark.skipif(
    not ANCHOR, reason="needs R1 seam: session_attribution(run_store, session_id) + Runtime.start discussion anchor (contract B, amendment 4)"
)


@pytest.fixture
def env(tmp_path, monkeypatch, request):
    return make_runtime(*make_stores(request.param, tmp_path)), Clock(monkeypatch), tmp_path


def _tick(runtime, run_id):
    run = runtime.get_state(run_id)
    return runtime.tick(workflow=runtime.workflow_registry.get(run.workflow_id), run_id=run_id)


@pytest.mark.parametrize("env", ["json", "sqlite"], indirect=True)
def test_discussion_is_a_seeded_read_only_fork(env):
    runtime, clock, tmp_path = env
    workspace = tmp_path / "ws"
    workspace.mkdir()
    aid = create(runtime, clock, mode="growing", workspace_root=str(workspace))
    drive(runtime, aid)
    at(runtime, clock, aid, "2026-01-01T00:02:00+00:00")
    at(runtime, clock, aid, "2026-01-01T00:04:00+00:00")
    occurrences = children(runtime, aid)
    automation_session = f"automation:{aid}"
    before_turns = [t.run_id for t in select_session_turns(runtime.run_store, automation_session)]
    before_state = dict(automation_state(runtime, aid))
    before_records = len(automation_records(runtime.ledger_store, aid))

    started = start_discussion(runtime, automation_id=aid, occurrence_index=2, request_id="d1", prompt="Why did it rise?")
    assert started["session_id"] == "discussion-session:d1" and started["session_kind"] == "discussion"
    # Idempotent per request id.
    assert start_discussion(runtime, automation_id=aid, occurrence_index=2, request_id="d1", prompt="Why did it rise?") == started
    disc = runtime.get_state(started["run_id"])
    assert disc.parent_run_id is None and disc.session_id == "discussion-session:d1"
    meta = disc.vars["_meta"]["discussion"]
    assert (meta["automation_id"], meta["occurrence_index"], meta["seed_run_id"]) == (aid, 2, occurrences[1].run_id)
    # Seeded through occurrence 2 only: two turns, never occurrence 3.
    assert [m["content"] for m in meta["seed_messages"]][1::2] == [o.output["response"] for o in occurrences[:2]]
    assert disc.vars["prompt"] == "Why did it rise?"
    assert disc.vars["workspace_root"] == str(workspace) and disc.vars["workspace_read_only"] is True
    assert disc.vars["_runtime"]["workspace_read_only"] is True
    assert WorkspaceScope.from_input_data(disc.vars).read_only is True

    assert _tick(runtime, started["run_id"]).output["response"] == "echo:Why did it rise? | history=4"

    # A later turn in the discussion session sees the seed first, then the first discussion turn.
    follow_up = runtime.start(
        workflow=runtime.workflow_registry.get("echo"),
        vars={"prompt": "And then?", "_meta": {"discussion": {k: v for k, v in meta.items() if k != "seed_messages"}},
              "workspace_root": str(workspace), "workspace_read_only": True},
        session_id="discussion-session:d1",
    )
    history = session_chat_messages(run_store=runtime.run_store, ledger_store=runtime.ledger_store,
                                    session_id="discussion-session:d1", strict=True)
    assert [m["content"] for m in history][:4] == [m["content"] for m in meta["seed_messages"]]
    assert len(history) == 6  # seed (4) + the first discussion turn (2)
    assert _tick(runtime, follow_up).status == RunStatus.COMPLETED

    # Nothing was written back into the automation.
    assert [t.run_id for t in select_session_turns(runtime.run_store, automation_session)] == before_turns
    assert automation_state(runtime, aid) == before_state
    assert len(automation_records(runtime.ledger_store, aid)) == before_records
    assert all(o.status == RunStatus.COMPLETED for o in children(runtime, aid))


@pytest.mark.parametrize("env", ["json"], indirect=True)
def test_discussing_an_unknown_occurrence_is_refused(env):
    runtime, clock, _ = env
    aid = create(runtime, clock)
    drive(runtime, aid)
    with pytest.raises(AutomationError) as exc:
        start_discussion(runtime, automation_id=aid, occurrence_index=9, request_id="d", prompt="?")
    assert exc.value.reason_code == "occurrence_not_found"


@needs_anchor
@pytest.mark.parametrize("env", ["json"], indirect=True)
def test_a_raw_start_in_a_discussion_session_is_restamped_read_only(env):
    runtime, clock, tmp_path = env
    workspace = tmp_path / "ws"
    workspace.mkdir()
    aid = create(runtime, clock, workspace_root=str(workspace))
    drive(runtime, aid)
    started = start_discussion(runtime, automation_id=aid, occurrence_index=1, request_id="d2", prompt="?")
    later_id = runtime.start(workflow=runtime.workflow_registry.get("echo"), vars={"prompt": "And then?"},
                             session_id=started["session_id"])
    later = runtime.get_state(later_id)
    assert later.vars["workspace_read_only"] is True and later.vars["workspace_root"] == str(workspace)
    assert later.vars["_meta"]["discussion"]["discussion_root_run_id"] == started["run_id"]
    assert "seed_messages" not in later.vars["_meta"]["discussion"]
