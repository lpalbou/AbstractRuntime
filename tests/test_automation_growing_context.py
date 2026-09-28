"""Occurrences as conversation turns: Independent vs Growing (contract D `prepare_context`, C3).

Independent: each occurrence is the only turn of its own session and receives
no history. Growing: occurrences are turns of the automation's session and
each one receives the prior turns (bounded) as `context.messages`, read through
the strict history path; a history that cannot be read fails the admission
instead of running without context.
"""

from __future__ import annotations

import inspect

import pytest

from automation_harness import Clock, at, children, create, drive, make_runtime, make_stores
from abstractruntime.core.models import RunStatus
from abstractruntime.session_history import session_chat_messages
from abstractruntime.session_turns import select_session_turns

STORES = ["json", "sqlite"]
_PARAMS = inspect.signature(session_chat_messages).parameters
STRICT_HISTORY = all(p in _PARAMS for p in ("strict", "automation_id", "through_occurrence"))
needs_strict_history = pytest.mark.skipif(
    not STRICT_HISTORY,
    reason="needs R1 seam: session_chat_messages(strict=, automation_id=, through_occurrence=) (contract C3/E)",
)


@pytest.fixture
def env(tmp_path, monkeypatch, request):
    return make_runtime(*make_stores(request.param, tmp_path)), Clock(monkeypatch)


def _three_occurrences(runtime, clock, aid):
    drive(runtime, aid)
    at(runtime, clock, aid, "2026-01-01T00:02:00+00:00")
    at(runtime, clock, aid, "2026-01-01T00:04:00+00:00")
    kids = children(runtime, aid)
    assert len(kids) == 3 and all(k.status == RunStatus.COMPLETED for k in kids)
    return kids


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_independent_occurrences_are_single_turn_sessions(env):
    runtime, clock = env
    aid = create(runtime, clock)
    kids = _three_occurrences(runtime, clock, aid)
    assert len({k.session_id for k in kids}) == 3
    for kid in kids:
        assert kid.output["response"].endswith("| history=0")
        assert [t.run_id for t in select_session_turns(runtime.run_store, kid.session_id)] == [kid.run_id]
    # The controller's own session holds no turns (controllers are never turns).
    assert select_session_turns(runtime.run_store, f"automation:{aid}") == []


@needs_strict_history
@pytest.mark.parametrize("env", STORES, indirect=True)
def test_growing_occurrences_see_prior_turns(env):
    runtime, clock = env
    aid = create(runtime, clock, mode="growing")
    kids = _three_occurrences(runtime, clock, aid)
    session = f"automation:{aid}"
    assert {k.session_id for k in kids} == {session}
    assert [k.vars["_meta"]["occurrence"]["session_kind"] for k in kids] == ["automation"] * 3
    assert [k.output["response"].rsplit("=", 1)[1] for k in kids] == ["0", "2", "4"]
    second_history = kids[1].vars["context"]["messages"]
    assert [m["role"] for m in second_history] == ["user", "assistant"]
    assert second_history[1]["content"] == kids[0].output["response"]
    assert [t.run_id for t in select_session_turns(runtime.run_store, session)] == [k.run_id for k in kids]


@needs_strict_history
@pytest.mark.parametrize("env", STORES, indirect=True)
def test_growing_admission_fails_loudly_when_history_cannot_be_read(env, monkeypatch):
    runtime, clock = env
    aid = create(runtime, clock, mode="growing")
    drive(runtime, aid)

    def unreadable(*args, **kwargs):
        raise RuntimeError("history index unavailable")

    monkeypatch.setattr(type(runtime.run_store), "list_run_index", unreadable)
    clock.set("2026-01-01T00:02:00+00:00")
    from automation_harness import wake

    wake(runtime, aid)
    with pytest.raises(Exception):
        drive(runtime, aid)
    assert len(children(runtime, aid)) == 1  # no occurrence started without its context


@needs_strict_history
@pytest.mark.parametrize("env", STORES, indirect=True)
def test_growing_history_is_the_50k_token_window_and_the_run_records_it(env):
    """Operator ruling 2026-09-28: the growing-mode history is the most recent
    50,000 tokens of whole turns — no 40-message / 24,000-char cap — and the
    occurrence run (plus its `automation.admitted` record) says what was
    replayed and dropped (ADR-0026: explicit, observable)."""
    from abstractruntime.automations.ledger import automation_records
    from abstractruntime.core.models import RunState

    runtime, clock = env
    aid = create(runtime, clock, mode="growing")
    session = f"automation:{aid}"
    answer = "a long prior answer " * 100  # 2,000 chars per answer
    for i in range(30):  # 60 prior messages, ~60,000 chars: both old caps exceeded
        ts = f"2025-12-31T23:{i:02d}:00+00:00"
        runtime.run_store.save(RunState(
            run_id=f"prior-{i:02d}", workflow_id="echo", status=RunStatus.COMPLETED, current_node="done",
            vars={"prompt": f"prior question {i}", "context": {"task": f"prior question {i}", "messages": []}},
            output={"response": answer}, error=None, created_at=ts, updated_at=ts, actor_id="t",
            session_id=session, parent_run_id=None, waiting=None,
        ))
    drive(runtime, aid)
    kid = children(runtime, aid)[0]
    history = kid.vars["context"]["messages"]
    assert len(history) == 60 and kid.output["response"].endswith("| history=60")
    assert all(m["content"] == answer.strip() for m in history[1::2])  # uncut
    note = kid.vars["_runtime"]["session_history"]
    assert note["max_tokens"] == 50_000 and note["policy"] == "most_recent_whole_turns"
    assert note["replayed_messages"] == 60 and note["dropped_messages"] == 0
    assert note["session_kind"] == "automation" and note["strict"] is True
    admitted = automation_records(runtime.ledger_store, aid, "automation.admitted")[0]["payload"]
    assert admitted["prepared"]["input_data"]["_runtime"]["session_history"] == note


@needs_strict_history
@pytest.mark.parametrize("env", STORES, indirect=True)
def test_the_context_mode_decides_use_context_not_a_stale_target_input(env, tmp_path):
    """An automation created before use_context was server-owned carries
    `use_context: false` in its target; basic-agent then read NO history. The
    context mode is the one rule: growing (and discussions) read history,
    independent does not — recorded in the run as `_runtime.automation_context`."""
    from abstractruntime.automations import start_discussion

    runtime, clock = env
    old_target = {"prompt": "tick", "use_context": False}
    gro = create(runtime, clock, mode="growing", workflow_id="reader", input_data=dict(old_target))
    ind = create(runtime, clock, workflow_id="reader", input_data={"prompt": "tick", "use_context": True})
    for aid in (gro, ind):
        drive(runtime, aid)
    clock.set("2026-01-01T00:02:00+00:00")
    for aid in (gro, ind):
        at(runtime, clock, aid, "2026-01-01T00:02:00+00:00")
    g1, g2 = children(runtime, gro)
    i1, i2 = children(runtime, ind)
    assert g2.output["response"].endswith("| seen=2")  # replays despite the stale False
    assert g2.vars["use_context"] is True
    assert g2.vars["_runtime"]["automation_context"] == {"mode": "growing", "use_context": True, "target_use_context": False}
    assert i2.output["response"].endswith("| seen=0") and i2.vars["use_context"] is False
    assert i2.vars["_runtime"]["automation_context"] == {"mode": "independent", "use_context": False, "target_use_context": True}

    # A discussion forked from an independent occurrence (frozen use_context=False) reads its seed.
    ws = tmp_path / "disc-ws"
    ws.mkdir()
    started = start_discussion(runtime, automation_id=ind, occurrence_index=2, request_id="d", prompt="why?",
                               workspace_root=str(ws))
    fork = runtime.get_state(started["run_id"])
    assert fork.vars["use_context"] is True
    assert fork.vars["_runtime"]["automation_context"] == {"mode": "discussion", "use_context": True, "target_use_context": False}
