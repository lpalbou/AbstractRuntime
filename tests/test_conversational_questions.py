"""Durable human questions: strict cutoff, fresh identity and no invented consent."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from abstractruntime.core.models import Effect, EffectType, RunStatus, StepPlan
from abstractruntime.core.runtime import Runtime, StaleResumeError
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.replay import replay
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


POLICY = {"importance": "clarification", "timeout_s": 120, "fallback": "Compare both options."}


def workflow(policy=POLICY, *, twice=False):
    def ask(run, ctx):
        generation = run.vars.get("generation", 0) + 1
        run.vars["generation"] = generation
        payload = {"prompt": f"Inside or outside? ({generation})"}
        if policy is not None:
            payload["question"] = policy
        return StepPlan(node_id="ask", effect=Effect(type=EffectType.ASK_USER, payload=payload,
                                                     result_key="answer"), next_node="next")

    def next_node(run, ctx):
        if twice and run.vars["generation"] < 2:
            return StepPlan(node_id="next", next_node="ask")
        return StepPlan(node_id="next", complete_output=run.vars["answer"])

    return WorkflowSpec(workflow_id="questions", entry_node="ask", nodes={"ask": ask, "next": next_node})


@pytest.fixture
def clock(monkeypatch):
    now = ["2027-01-01T12:00:00+00:00"]
    monkeypatch.setattr("abstractruntime.core.runtime.utc_now_iso", lambda: now[0])
    return now


def park(wf):
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    rid = rt.start(workflow=wf, vars={})
    state = rt.tick(workflow=wf, run_id=rid)
    assert state.status == RunStatus.WAITING
    return rt, rid, state.waiting.wait_key


def test_legacy_ask_remains_indefinite_and_essential_has_no_deadline(clock):
    for policy in (None, {"importance": "essential"}):
        rt, rid, key = park(workflow(policy))
        state = rt.get_state(rid)
        assert state.waiting.until is None
        if policy is None:
            assert state.waiting.details is None
            assert key == f"user:{rid}:ask"
        else:
            assert state.waiting.details["question"]["purpose"] == "conversation"
            assert key.startswith("question:")
        clock[0] = "2027-01-02T12:00:00+00:00"
        assert rt.tick(workflow=workflow(policy), run_id=rid).status == RunStatus.WAITING


@pytest.mark.parametrize("policy", [
    {"importance": "essential", "timeout_s": 120},
    {"importance": "clarification", "timeout_s": True, "fallback": "x"},
    {"importance": "clarification", "timeout_s": 119, "fallback": "x"},
    {"importance": "clarification", "timeout_s": 301, "fallback": "x"},
    {"importance": "clarification", "timeout_s": "120", "fallback": "x"},
    {"importance": "clarification", "timeout_s": 120},
    {"importance": "urgent"},
])
def test_invalid_policies_fail_without_starting_a_wait(policy):
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    wf = workflow(policy)
    rid = rt.start(workflow=wf, vars={})
    assert rt.tick(workflow=wf, run_id=rid).status == RunStatus.FAILED


def test_answer_after_cutoff_expires_even_before_due_scan_and_keeps_text(clock):
    wf = workflow()
    rt, rid, key = park(wf)
    assert rt.get_state(rid).waiting.until == "2027-01-01T12:02:00+00:00"
    clock[0] = "2027-01-01T12:02:00+00:00"  # exact equality belongs to expiry
    out = rt.resume(workflow=wf, run_id=rid, wait_key=key, payload={"response": "Outside"})
    assert out.output["timed_out"] is True
    assert out.output["fallback"] == POLICY["fallback"]
    assert "response" not in out.output and "approved" not in out.output
    rows = rt.get_ledger(rid)
    retained = [r for r in rows if (r.get("effect") or {}).get("payload", {}).get("name") == "abstract.question_reply_retained"]
    assert retained[0]["result"]["payload"]["response"] == "Outside"
    view = replay({rid: rows}, root_run_id=rid).to_json()
    assert not any(m["role"] == "user" and m["text"] == POLICY["fallback"] for m in view["messages"])


def test_early_answer_and_late_old_generation_do_not_answer_next_question(clock):
    wf = workflow(twice=True)
    rt, rid, first = park(wf)
    second = rt.resume(workflow=wf, run_id=rid, wait_key=first, payload={"response": "Inside"})
    assert second.waiting.wait_key != first
    second_key = second.waiting.wait_key
    rt.resume(workflow=wf, run_id=rid, wait_key=first, payload={"response": "Old reply"})
    assert rt.get_state(rid).waiting.wait_key == second_key
    assert rt.get_state(rid).vars["_runtime"]["last_question_reply"]["status"] == "retained"
    with pytest.raises(StaleResumeError):
        rt.resume(workflow=wf, run_id=rid, wait_key=None, payload={"response": "No identity"})


def test_pause_restart_and_due_scan_preserve_original_deadline(clock):
    wf = workflow()
    rt, rid, key = park(wf)
    until = rt.get_state(rid).waiting.until
    rt.pause_run(rid)
    clock[0] = "2027-01-01T12:04:00+00:00"
    rt = Runtime(run_store=rt.run_store, ledger_store=rt.ledger_store)
    assert rt.tick(workflow=wf, run_id=rid).status == RunStatus.WAITING
    assert rt.get_state(rid).waiting.until == until
    rt.resume_run(rid)
    assert rid in [r.run_id for r in rt.run_store.list_due_wait_until(now_iso=clock[0])]
    assert rt.tick(workflow=wf, run_id=rid).output["timed_out"] is True


def test_concurrent_answer_cancel_has_one_question_outcome(clock):
    wf = workflow()
    rt, rid, key = park(wf)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(rt.resume, workflow=wf, run_id=rid, wait_key=key,
                               payload={"response": "Inside"}, max_steps=0),
                   pool.submit(rt.cancel_run, rid)]
        for future in futures:
            future.result(timeout=3)
    history = rt.get_state(rid).vars["_runtime"]["question_outcomes"]
    assert len(history) == 1 and history[0]["outcome"] in ("answered", "cancelled")
    assert rt.get_state(rid).status == RunStatus.CANCELLED


@pytest.mark.parametrize("backend", ["memory", "json", "sqlite"])
def test_deadline_survives_store_reopen_and_is_indexed(backend, tmp_path, clock):
    if backend == "json":
        from abstractruntime.storage.json_files import JsonFileRunStore
        store = JsonFileRunStore(tmp_path)
    elif backend == "sqlite":
        from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore
        store = SqliteRunStore(SqliteDatabase(tmp_path / "runs.sqlite"))
    else:
        store = InMemoryRunStore()
    rt = Runtime(run_store=store, ledger_store=InMemoryLedgerStore())
    wf = workflow()
    rid = rt.start(workflow=wf, vars={})
    rt.tick(workflow=wf, run_id=rid)
    if backend == "json":
        store = JsonFileRunStore(tmp_path)
    elif backend == "sqlite":
        store = SqliteRunStore(SqliteDatabase(tmp_path / "runs.sqlite"))
    assert store.next_wait_deadline() == "2027-01-01T12:02:00+00:00"
    assert [r.run_id for r in store.list_due_wait_until(now_iso="2027-01-01T12:03:00+00:00")] == [rid]


@pytest.mark.parametrize("winner", ["answered", "expired", "cancelled"])
def test_resolution_receipt_survives_checkpoint_failure(winner, tmp_path, clock, monkeypatch):
    from abstractruntime.storage.json_files import JsonFileRunStore

    store = JsonFileRunStore(tmp_path)
    ledger = InMemoryLedgerStore()
    rt = Runtime(run_store=store, ledger_store=ledger)
    wf = workflow()
    rid = rt.start(workflow=wf)
    key = rt.tick(workflow=wf, run_id=rid).waiting.wait_key
    save = store.save

    def fail_checkpoint(run):
        raise OSError("checkpoint unavailable after outcome receipt")

    monkeypatch.setattr(store, "save", fail_checkpoint)
    with pytest.raises(OSError):
        if winner == "cancelled":
            rt.cancel_run(rid)
        else:
            if winner == "expired":
                clock[0] = "2027-01-01T12:02:00+00:00"
            rt.resume(workflow=wf, run_id=rid, wait_key=key, payload={"response": "Inside"}, max_steps=0)
    monkeypatch.setattr(store, "save", save)
    clock[0] = "2027-01-01T12:04:00+00:00"
    rt = Runtime(run_store=JsonFileRunStore(tmp_path), ledger_store=ledger)
    state = rt.tick(workflow=wf, run_id=rid)
    receipts = [r for r in rt.get_ledger(rid) if (r.get("effect") or {}).get("payload", {}).get("name") == "abstract.question_resolved"]
    assert len(receipts) == 1
    assert receipts[0]["result"]["question"]["outcome"] == winner
    if winner == "cancelled":
        assert state.status == RunStatus.CANCELLED
    elif winner == "answered":
        assert state.output["response"] == "Inside"
    else:
        assert state.output["timed_out"] is True and "response" not in state.output


@pytest.mark.parametrize("configured", [True, False])
def test_visual_question_policy_and_expiry_outputs(configured, clock):
    from abstractruntime.visualflow_compiler import compile_visualflow

    data = {"nodeType": "ask_user", "pinDefaults": {"prompt": "Inside or outside?"},
            "inputs": [{"id": "exec-in", "type": "execution"}, {"id": "prompt", "type": "string"}, {"id": "question", "type": "object"}],
            "outputs": [{"id": "exec-out", "type": "execution"}, {"id": "response", "type": "string"}]}
    if configured:
        data["effectConfig"] = {"question": POLICY}
    else:
        data["pinDefaults"]["question"] = POLICY
    result_pins = [{"id": "timed_out", "type": "boolean"}, {"id": "fallback", "type": "string"},
                   {"id": "question", "type": "object"}, {"id": "response", "type": "string"}]
    data["outputs"] = [{"id": "exec-out", "type": "execution"}] + result_pins
    wf = compile_visualflow({"id": "visual-question", "entryNode": "start", "nodes": [
        {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start", "outputs": [{"id": "exec-out", "type": "execution"}]}},
        {"id": "ask", "type": "ask_user", "data": data},
        {"id": "end", "type": "on_flow_end", "data": {"nodeType": "on_flow_end", "inputs": [{"id": "exec-in", "type": "execution"}] + result_pins}},
    ], "edges": [
        {"id": "begin", "source": "start", "sourceHandle": "exec-out", "target": "ask", "targetHandle": "exec-in"},
        {"id": "exec", "source": "ask", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"},
        *[{"source": "ask", "sourceHandle": pin["id"], "target": "end", "targetHandle": pin["id"]} for pin in result_pins]]})
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    rid = rt.start(workflow=wf)
    state = rt.tick(workflow=wf, run_id=rid)
    assert state.waiting is not None, (state.status, state.error, state.output)
    assert state.waiting.details["question"]["importance"] == "clarification"
    clock[0] = "2027-01-01T12:02:00+00:00"
    state = rt.tick(workflow=wf, run_id=rid)
    outputs = state.output
    assert outputs["timed_out"] is True and outputs["fallback"] == POLICY["fallback"]
    assert outputs["question"]["outcome"] == "expired" and not outputs.get("response")
