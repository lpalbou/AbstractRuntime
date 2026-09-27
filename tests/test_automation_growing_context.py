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
