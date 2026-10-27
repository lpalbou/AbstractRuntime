"""Session tool policy (contract v1): a person's standing tool decision for one conversation.

The runtime consults the session's policy BEFORE it decides whether a tool batch must wait: a
covered call runs with no wait and no interaction, a denied call is refused with no wait, and each
decision writes one `abstract.tool_policy_decision` record (an activity, never a question). Every
run of the session sees it: the turn's root run, a delegated child, a run resumed after a restart.
"""

from __future__ import annotations

import pytest

from automation_harness import make_stores
from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunStatus
from abstractruntime.integrations.abstractcore.effect_handlers import (
    SESSION_POLICY_DENIED_ERROR,
    TOOL_POLICY_DECISION_EVENT,
    _execute_with_run_policy,
    make_tool_calls_handler,
)
from abstractruntime.integrations.abstractcore.tool_executor import (
    ApprovalToolExecutor,
    MappingToolExecutor,
    ToolApprovalPolicy,
)
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.session_tool_policy import (
    InMemorySessionToolPolicyStore,
    SessionToolPolicyConflict,
    SqliteSessionToolPolicyStore,
    session_policy_names,
)

EXECUTED: list = []
SET_BY = {"tenant_id": "default", "user_id": "laurent", "client": "test"}


def _fetch(url: str = "") -> str:
    EXECUTED.append(("fetch_url", url))
    return f"fetched {url}"


def _write(path: str = "", content: str = "") -> str:
    EXECUTED.append(("write_file", path))
    return f"wrote {path}"


def _calls_node(key: str, next_node: str):
    def _node(run, ctx):
        calls = run.vars.get(key) or []
        return StepPlan(
            node_id=key,
            effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": calls}, result_key=f"{key}_result"),
            next_node=next_node,
        )

    return _node


def _done(run, ctx):
    return StepPlan(node_id="done", complete_output={"success": True, "first": run.vars.get("first_result"),
                                                      "second": run.vars.get("second_result")})


TOOLS = WorkflowSpec(
    workflow_id="tools",
    entry_node="first",
    nodes={"first": _calls_node("first", "second"), "second": _calls_node("second", "done"), "done": _done},
)


def _delegate(run, ctx):
    payload = {"workflow_id": "tools", "vars": {"first": run.vars.get("first"), "second": run.vars.get("second")}}
    if run.vars.get("child_session_id"):
        payload["session_id"] = run.vars["child_session_id"]
    return StepPlan(
        node_id="delegate",
        effect=Effect(
            type=EffectType.START_SUBWORKFLOW,
            payload=payload,
            result_key="child",
        ),
        next_node="end",
    )


def _end(run, ctx):
    return StepPlan(node_id="end", complete_output={"success": True})


PARENT = WorkflowSpec(workflow_id="parent", entry_node="delegate", nodes={"delegate": _delegate, "end": _end})


def _runtime(tmp_path, *, store=None, kind="json", static=None):
    run_store, ledger_store = make_stores(kind, tmp_path)
    registry = WorkflowRegistry()
    registry.register(TOOLS)
    registry.register(PARENT)
    tools = ApprovalToolExecutor(
        delegate=MappingToolExecutor({"fetch_url": _fetch, "write_file": _write}),
        policy=static or ToolApprovalPolicy(),  # the default: fetch_url and write_file ask
    )
    rt = Runtime(run_store=run_store, ledger_store=ledger_store, workflow_registry=registry,
                 effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)},
                 session_tool_policy_store=store)
    rt.set_tool_executor_for_resume(tools)
    return rt


def _call(name, call_id, **args):
    return {"name": name, "call_id": call_id, "arguments": args}


FETCH = [_call("fetch_url", "f1", url="https://example.invalid/a")]
FETCH2 = [_call("fetch_url", "f2", url="https://example.invalid/b")]
WRITE = [_call("write_file", "w1", path="notes.md", content="x")]


def _policy(approve=None, deny=None, ask=None):
    return {"approve": dict(approve or {}), "deny": dict(deny or {}), "ask": list(ask or []),
            "set_by": dict(SET_BY), "set_at": "2026-10-27T10:00:00+00:00"}


def _records(rt, run_id):
    return rt.ledger_store.list(run_id)


def _decisions(rt, run_id):
    out = []
    for rec in _records(rt, run_id):
        eff = rec.get("effect") or {}
        payload = eff.get("payload") or {}
        if eff.get("type") == "emit_event" and payload.get("name") == TOOL_POLICY_DECISION_EVENT:
            out.append(payload["payload"])
    return out


def _waits(rt, run_id):
    return [r for r in _records(rt, run_id) if r.get("status") == "waiting"]


@pytest.fixture(autouse=True)
def _clear():
    EXECUTED.clear()


def _start(rt, *, first, second=None, session_id="s1", workflow=TOOLS, extra_vars=None):
    vars_ = {"first": first, "second": second or []}
    vars_.update(extra_vars or {})
    run_id = rt.start(workflow=workflow, vars=vars_, session_id=session_id)
    return run_id, rt.tick(workflow=workflow, run_id=run_id)


# --- the decision, end to end through a Runtime ----------------------------------------------


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_a_session_approved_tool_runs_with_no_wait_and_an_audit_record(tmp_path, kind):
    store = InMemorySessionToolPolicyStore()
    store.put("s1", _policy(approve={"fetch_url": "kind:web"}))
    rt = _runtime(tmp_path, store=store, kind=kind)
    run_id, state = _start(rt, first=FETCH, second=FETCH2)
    assert state.status == RunStatus.COMPLETED, state.waiting
    assert EXECUTED == [("fetch_url", "https://example.invalid/a"), ("fetch_url", "https://example.invalid/b")]
    assert _waits(rt, run_id) == []  # no interaction ever reached the ledger
    first, second = _decisions(rt, run_id)
    assert first["decision"] == "auto_approved" and first["source"] == "session_policy"
    assert first["session_id"] == "s1" and first["node_id"] == "first"
    assert first["tool_calls"] == [{"call_id": "f1", "name": "fetch_url", "matched": "kind:web"}]
    assert first["policy"] == {"revision": 1, "set_by": SET_BY, "set_at": "2026-10-27T10:00:00+00:00"}
    assert second["tool_calls"][0]["call_id"] == "f2"


def test_without_a_session_policy_the_same_call_waits(tmp_path):
    rt = _runtime(tmp_path, store=InMemorySessionToolPolicyStore())
    run_id, state = _start(rt, first=FETCH)
    assert state.status == RunStatus.WAITING
    assert state.waiting.details["mode"] == "approval_required"
    assert EXECUTED == [] and _decisions(rt, run_id) == []


def test_a_delegated_child_run_uses_the_session_policy(tmp_path):
    store = InMemorySessionToolPolicyStore()
    store.put("s1", _policy(approve={"fetch_url": "all", "write_file": "all"}))
    rt = _runtime(tmp_path, store=store)
    run_id, state = _start(rt, first=FETCH, second=WRITE, workflow=PARENT)
    assert state.status == RunStatus.COMPLETED, state.waiting
    child_ids = [r.run_id for r in rt.run_store.list_children(parent_run_id=run_id)]
    assert len(child_ids) == 1
    child = rt.get_state(child_ids[0])
    assert child.session_id == "s1" and child.status == RunStatus.COMPLETED
    assert [c[0] for c in EXECUTED] == ["fetch_url", "write_file"]
    assert [d["tool_calls"][0]["name"] for d in _decisions(rt, child.run_id)] == ["fetch_url", "write_file"]
    assert _waits(rt, child.run_id) == []


def test_a_policy_set_mid_run_applies_from_the_next_batch_and_one_off_answers_still_work(tmp_path):
    store = InMemorySessionToolPolicyStore()
    rt = _runtime(tmp_path, store=store)
    run_id, state = _start(rt, first=FETCH, second=FETCH2)
    assert state.status == RunStatus.WAITING
    # "Approve all" on the shown wait: the policy first, then the ordinary one-off answer.
    store.put("s1", _policy(approve={"fetch_url": "kind:web"}))
    state = rt.resume(workflow=TOOLS, run_id=run_id, wait_key=state.waiting.wait_key, payload={"approved": True})
    if state.status != RunStatus.COMPLETED:
        state = rt.tick(workflow=TOOLS, run_id=run_id)
    assert state.status == RunStatus.COMPLETED
    assert [c[1] for c in EXECUTED] == ["https://example.invalid/a", "https://example.invalid/b"]
    assert len(_waits(rt, run_id)) == 1  # the one shown before the policy existed
    [decision] = _decisions(rt, run_id)
    assert decision["tool_calls"][0]["call_id"] == "f2"


def test_deny_refuses_with_no_wait_and_a_denied_record(tmp_path):
    store = InMemorySessionToolPolicyStore()
    store.put("s1", _policy(approve={"fetch_url": "all"}, deny={"write_file": "tool"}))
    rt = _runtime(tmp_path, store=store)
    run_id, state = _start(rt, first=WRITE + FETCH)
    assert state.status == RunStatus.COMPLETED, state.waiting
    results = state.output["first"]["results"]
    assert results[0]["success"] is False and results[0]["error"] == SESSION_POLICY_DENIED_ERROR
    assert results[1]["success"] is True
    assert EXECUTED == [("fetch_url", "https://example.invalid/a")]
    denied, approved = _decisions(rt, run_id)
    assert denied["decision"] == "denied" and denied["tool_calls"] == [{"call_id": "w1", "name": "write_file", "matched": "tool"}]
    assert approved["decision"] == "auto_approved"
    assert _waits(rt, run_id) == []


def test_a_batch_with_an_uncovered_call_still_waits_as_a_whole(tmp_path):
    store = InMemorySessionToolPolicyStore()
    store.put("s1", _policy(approve={"fetch_url": "kind:web"}))
    rt = _runtime(tmp_path, store=store)
    run_id, state = _start(rt, first=FETCH + WRITE)
    assert state.status == RunStatus.WAITING
    assert [c["name"] for c in state.waiting.details["tool_calls"]] == ["fetch_url", "write_file"]
    assert EXECUTED == [] and _decisions(rt, run_id) == []


def test_session_approval_overrides_the_start_time_require_list(tmp_path):
    store = InMemorySessionToolPolicyStore()
    store.put("s1", _policy(approve={"fetch_url": "all"}))
    rt = _runtime(tmp_path, store=store)
    # What AbstractCode web and the desktop send today: every enabled tool on the require list.
    run_id, state = _start(rt, first=FETCH, extra_vars={"_runtime": {"tool_policy": {
        "auto_approve_tools": [], "require_approval_tools": ["fetch_url", "write_file"]}}})
    assert state.status == RunStatus.COMPLETED, state.waiting
    assert len(_decisions(rt, run_id)) == 1


@pytest.mark.parametrize(
    "case",
    ["ask_list", "withheld", "untrusted", "automation", "sessionless", "no_store"],
)
def test_out_of_scope_runs_and_names_keep_asking(tmp_path, case):
    store = InMemorySessionToolPolicyStore()
    store.put("s1", _policy(approve={"fetch_url": "all"}, ask=["fetch_url"] if case == "ask_list" else None))
    rt = _runtime(tmp_path, store=None if case == "no_store" else store)
    extra = {
        "withheld": {"_runtime": {"tool_policy": {"auto_approve_tools": [], "withheld_tools": ["fetch_url"]}}},
        "untrusted": {"_runtime": {"untrusted_input": True}},
        "automation": {"_meta": {"occurrence": {"automation_id": "a1", "role": "descendant"}}},
    }.get(case, {})
    run_id, state = _start(rt, first=FETCH, session_id=None if case == "sessionless" else "s1", extra_vars=extra)
    assert state.status == RunStatus.WAITING, case
    assert EXECUTED == [] and _decisions(rt, run_id) == []


@pytest.mark.parametrize("case", ["untrusted", "automation"])
def test_deny_reaches_runs_that_approve_never_reaches(tmp_path, case):
    store = InMemorySessionToolPolicyStore()
    store.put("s1", _policy(approve={"fetch_url": "all"}, deny={"write_file": "tool"}))
    rt = _runtime(tmp_path, store=store)
    extra = {
        "untrusted": {"_runtime": {"untrusted_input": True}},
        # An automation grant that would auto-run write_file: the conversation's deny still wins.
        "automation": {"_meta": {"occurrence": {"automation_id": "a1"}},
                       "_runtime": {"tool_policy": {"auto_approve_tools": ["write_file"], "source": "automation-policy"}}},
    }[case]
    run_id, state = _start(rt, first=WRITE, extra_vars=extra)
    assert state.status == RunStatus.COMPLETED, state.waiting
    assert state.output["first"]["results"][0]["error"] == SESSION_POLICY_DENIED_ERROR
    assert EXECUTED == [] and [d["decision"] for d in _decisions(rt, run_id)] == ["denied"]


def test_a_child_with_its_own_session_id_uses_the_conversations_policy(tmp_path):
    store = InMemorySessionToolPolicyStore()
    store.put("other", _policy(approve={"fetch_url": "all"}))  # another conversation's grant
    rt = _runtime(tmp_path, store=store)
    run_id, state = _start(rt, first=FETCH, workflow=PARENT, extra_vars={"child_session_id": "other"})
    [child] = rt.run_store.list_children(parent_run_id=run_id)
    assert child.session_id == "other"
    assert rt.get_state(child.run_id).status == RunStatus.WAITING  # s1 granted nothing
    assert EXECUTED == []


def test_a_deny_set_while_a_wait_is_shown_wins_over_its_approval(tmp_path):
    store = InMemorySessionToolPolicyStore()
    rt = _runtime(tmp_path, store=store)
    run_id, state = _start(rt, first=FETCH)
    assert state.status == RunStatus.WAITING
    store.put("s1", _policy(deny={"fetch_url": "kind:web"}))
    state = rt.resume(workflow=TOOLS, run_id=run_id, wait_key=state.waiting.wait_key, payload={"approved": True})
    if state.status != RunStatus.COMPLETED:
        state = rt.tick(workflow=TOOLS, run_id=run_id)
    assert state.status == RunStatus.COMPLETED
    assert state.output["first"]["results"][0]["error"] == SESSION_POLICY_DENIED_ERROR
    assert EXECUTED == []
    assert [d["decision"] for d in _decisions(rt, run_id)] == ["denied"]


def test_a_reset_policy_asks_again(tmp_path):
    store = InMemorySessionToolPolicyStore()
    store.put("s1", _policy(approve={"fetch_url": "all"}))
    store.delete("s1", actor=SET_BY)
    rt = _runtime(tmp_path, store=store)
    _run_id, state = _start(rt, first=FETCH)
    assert state.status == RunStatus.WAITING


def test_a_restart_keeps_the_policy(tmp_path):
    db = tmp_path / "data" / "session_tool_policy.sqlite3"
    SqliteSessionToolPolicyStore(str(db)).put("s1", _policy(approve={"fetch_url": "kind:web"}))
    # A new process: new store instance, new Runtime over the same data.
    rt = _runtime(tmp_path, store=SqliteSessionToolPolicyStore(str(db)))
    run_id, state = _start(rt, first=FETCH)
    assert state.status == RunStatus.COMPLETED
    assert _decisions(rt, run_id)[0]["policy"]["revision"] == 1


def test_no_audit_record_means_no_auto_approval():
    tools = ApprovalToolExecutor(delegate=MappingToolExecutor({"fetch_url": _fetch}), policy=ToolApprovalPolicy())

    class _Run:
        vars = {}
        run_id = "r1"
        session_id = "s1"
        current_node = "n"

    out = _execute_with_run_policy(tools, list(FETCH), _Run(), session_approved={"fetch_url": "all"},
                                   on_session_approved=lambda names: False)
    assert out["mode"] == "approval_required" and EXECUTED == []
    out = _execute_with_run_policy(tools, list(FETCH), _Run(), session_approved={"fetch_url": "all"},
                                   on_session_approved=lambda names: True)
    assert out["mode"] == "executed" and len(EXECUTED) == 1


# --- the store ------------------------------------------------------------------------------


@pytest.mark.parametrize("factory", ["memory", "sqlite"])
def test_store_revisions_conflicts_reset_and_history(tmp_path, factory):
    store = InMemorySessionToolPolicyStore() if factory == "memory" else SqliteSessionToolPolicyStore(str(tmp_path / "p.sqlite3"))
    assert store.get("s1") is None
    first = store.put("s1", _policy(approve={"fetch_url": "tool"}))
    assert first["revision"] == 1 and first["session_id"] == "s1" and first["version"] == 1
    with pytest.raises(SessionToolPolicyConflict):
        store.put("s1", _policy(), if_revision=0)
    second = store.put("s1", _policy(deny={"write_file": "tool"}), if_revision=1)
    assert second["revision"] == 2 and store.get("s1")["deny"] == {"write_file": "tool"}
    assert store.delete("s1", actor=SET_BY) == 3 and store.get("s1") is None
    assert store.put("s1", _policy())["revision"] == 4  # revisions never go back after a reset
    assert [h["action"] for h in store.history("s1")] == ["put", "put", "delete", "put"]
    assert store.get("other") is None


def test_policy_names_deny_wins_and_ask_removes():
    approved, denied = session_policy_names(
        {"approve": {"a": "all", "b": "kind:web", "c": "tool"}, "deny": {"b": "tool"}, "ask": ["c"]}
    )
    assert approved == {"a": "all"} and denied == {"b": "tool"}
    assert session_policy_names(None) == ({}, {})
    assert session_policy_names({"approve": ["a"], "deny": "b"}) == ({}, {})
