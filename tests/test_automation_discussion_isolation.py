"""Discuss (contract B, C4): a forked ROOT session seeded once, never writing back.

The discussion runs the occurrence's workflow in its own durable session, seeded
from the automation's conversation through the chosen occurrence, on the
occurrence's workspace mounted read-only. Later turns of the discussion session
append only there; the automation's session, state and ledger never change.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

import abstractruntime.core.run_attribution as run_attribution
from automation_harness import Clock, at, automation_state, children, create, drive, make_runtime, make_stores
from abstractruntime.automations import AutomationError, start_discussion
from abstractruntime.automations.ledger import automation_records
from abstractruntime.core.models import RunStatus
from abstractruntime.integrations.abstractcore.workspace_scoped_tools import WorkspaceScope
from abstractruntime.session_history import session_chat_messages
from abstractruntime.session_turns import select_session_turns

_PARAMS = inspect.signature(session_chat_messages).parameters
MISSING = [
    name
    for name, present in (
        ("session_chat_messages(strict=, automation_id=, through_occurrence=)",
         all(p in _PARAMS for p in ("strict", "automation_id", "through_occurrence"))),
        ("session_attribution(run_store, session_id) + Runtime.start discussion anchor",
         hasattr(run_attribution, "session_attribution")),
        ("WorkspaceScope.read_only", "read_only" in {f.name for f in dataclasses.fields(WorkspaceScope)}),
    )
    if not present
]
needs_r1 = pytest.mark.skipif(bool(MISSING), reason=f"needs R1 seam(s): {'; '.join(MISSING)}")


@pytest.fixture
def env(tmp_path, monkeypatch, request):
    return make_runtime(*make_stores(request.param, tmp_path)), Clock(monkeypatch), tmp_path


def _tick(runtime, run_id):
    run = runtime.get_state(run_id)
    return runtime.tick(workflow=runtime.workflow_registry.get(run.workflow_id), run_id=run_id)


@needs_r1
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
    assert WorkspaceScope.from_input_data(disc.vars).read_only is True

    assert _tick(runtime, started["run_id"]).output["response"] == "echo:Why did it rise? | history=4"

    # A later turn started raw in the discussion session is still a read-only discussion turn.
    later_id = runtime.start(workflow=runtime.workflow_registry.get("echo"), vars={"prompt": "And then?"},
                             session_id="discussion-session:d1")
    later = runtime.get_state(later_id)
    assert later.vars["workspace_read_only"] is True
    assert later.vars["_meta"]["discussion"]["automation_id"] == aid
    assert "seed_messages" not in later.vars["_meta"]["discussion"]
    history = session_chat_messages(run_store=runtime.run_store, ledger_store=runtime.ledger_store,
                                    session_id="discussion-session:d1", strict=True)
    assert len(history) == 6  # seed (4) + the first discussion turn (2)

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
