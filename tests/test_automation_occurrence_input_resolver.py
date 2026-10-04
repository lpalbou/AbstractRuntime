"""`Runtime.set_occurrence_input_resolver` (round 13, R13-W2 gateway follow-up):
a host hook that re-resolves an occurrence's inputs at ADMISSION (the
occurrence's run start) — the gateway uses it so an automation that follows
its owner's default workspaces sees the default as it is at each run, never a
snapshot taken when the automation was saved. The answer is frozen with the
occurrence's inputs (the dispatched child carries exactly it); without a
resolver nothing changes.

Real runtime, real stores, the packaged controller bundle, deterministic
targets (automation_harness)."""

from __future__ import annotations

import pytest

from automation_harness import Clock, at, children, create, drive, make_runtime, make_stores
from abstractruntime.automations.ledger import automation_records


@pytest.fixture
def env(tmp_path, monkeypatch):
    run_store, ledger_store = make_stores("json", tmp_path)
    runtime = make_runtime(run_store, ledger_store)
    return runtime, Clock(monkeypatch)


def test_without_a_resolver_the_definition_inputs_are_used_as_stored(env):
    runtime, clock = env
    assert runtime.occurrence_input_resolver is None
    aid = create(runtime, clock, input_data={"prompt": "check", "workspace": {"configured": False}})
    drive(runtime, aid)
    at(runtime, clock, aid, "2026-01-01T00:00:30+00:00")
    first = children(runtime, aid)[0]
    assert first.vars["workspace"] == {"configured": False}
    assert "workspace_allowed_paths" not in first.vars


def test_the_resolver_runs_at_each_admission_and_its_answer_reaches_the_occurrence(env):
    runtime, clock = env
    current = {"paths": ["/data/pictures"]}
    calls = []

    def resolver(definition, input_data):
        calls.append((definition["automation_id"] if "automation_id" in definition else definition.get("session_id"), dict(input_data)))
        out = dict(input_data)
        if out.get("workspace") == {"configured": False}:
            out["workspace_allowed_paths"] = list(current["paths"])
        return out

    runtime.set_occurrence_input_resolver(resolver)
    aid = create(runtime, clock, input_data={"prompt": "check", "workspace": {"configured": False}})
    drive(runtime, aid)
    at(runtime, clock, aid, "2026-01-01T00:00:30+00:00")
    # The owner's default widens between two runs: the NEXT occurrence sees it.
    current["paths"] = ["/data/pictures", "/data/downloads"]
    at(runtime, clock, aid, "2026-01-01T00:02:30+00:00")
    kids = children(runtime, aid)
    assert len(kids) == 2
    assert kids[0].vars["workspace_allowed_paths"] == ["/data/pictures"]
    assert kids[1].vars["workspace_allowed_paths"] == ["/data/pictures", "/data/downloads"]
    # Once per admission, with the definition's (prompt-rendered) inputs.
    assert len(calls) == 2
    assert all(c[1]["workspace"] == {"configured": False} for c in calls)
    # The admission record froze the answer (a replayed dispatch sends the same vars).
    admitted = [r for r in automation_records(runtime.ledger_store, aid) if r["name"] == "automation.admitted"]
    frozen = [r["payload"]["prepared"]["input_data"]["workspace_allowed_paths"] for r in admitted]
    assert frozen == [["/data/pictures"], ["/data/pictures", "/data/downloads"]]


def test_a_resolver_that_does_not_answer_a_dict_fails_the_admission_loudly(env):
    from abstractruntime.automations.controller import ControllerSeamError

    runtime, clock = env
    runtime.set_occurrence_input_resolver(lambda definition, input_data: None)
    aid = create(runtime, clock)
    with pytest.raises(ControllerSeamError, match="occurrence input resolver returned NoneType"):
        drive(runtime, aid)
        at(runtime, clock, aid, "2026-01-01T00:00:30+00:00")
    # Never a child started on stale inputs.
    assert not children(runtime, aid)


def test_the_setter_refuses_a_non_callable():
    from automation_harness import make_runtime as _mk  # noqa: F401
    from abstractruntime import Runtime
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    with pytest.raises(TypeError):
        rt.set_occurrence_input_resolver("not callable")  # type: ignore[arg-type]
    rt.set_occurrence_input_resolver(None)
    assert rt.occurrence_input_resolver is None
