from __future__ import annotations

from typing import Any

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_terminal_resume_appends_completion_record_to_ledger() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)

    vars: dict[str, Any] = {"_temp": {}, "_limits": {}, "_runtime": {}}

    def ask_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="ask",
            effect=Effect(
                type=EffectType.ASK_USER,
                payload={"prompt": "who are you?"},
                result_key="_temp.ask",
            ),
            next_node=None,  # terminal wait node on purpose
        )

    wf = WorkflowSpec(workflow_id="wf_terminal_resume_ledger", entry_node="ask", nodes={"ask": ask_node})
    run_id = runtime.start(workflow=wf, vars=vars)

    st1 = runtime.tick(workflow=wf, run_id=run_id, max_steps=10)
    assert st1.status == RunStatus.WAITING
    assert st1.waiting is not None

    ledger_before = runtime.get_ledger(run_id)
    assert isinstance(ledger_before, list)
    assert len(ledger_before) >= 2  # started + waiting for ask_user

    wait_key = st1.waiting.wait_key
    assert isinstance(wait_key, str) and wait_key

    st2 = runtime.resume(workflow=wf, run_id=run_id, wait_key=wait_key, payload={"response": "I am a test"}, max_steps=0)
    assert st2.status == RunStatus.COMPLETED

    ledger_after = runtime.get_ledger(run_id)
    # Resume should append at least a completion record + an abstract.status emission.
    # Some runtimes may also append an explicit "resume applied" effect record.
    assert len(ledger_after) >= len(ledger_before) + 2

    completed = [
        r
        for r in ledger_after
        if isinstance(r, dict)
        and isinstance(r.get("result"), dict)
        and (r.get("result") or {}).get("completed") is True
        and (r.get("result") or {}).get("via") == "resume"
    ]
    assert completed, "expected a completion record appended by resume()"
    last_completed = completed[-1]
    assert isinstance(last_completed, dict)
    assert last_completed.get("status") == "completed"
    result = last_completed.get("result")
    assert isinstance(result, dict)
    assert result.get("completed") is True
    assert result.get("via") == "resume"
    assert result.get("wait_key") == wait_key
    assert result.get("output") == {"success": True, "result": {"response": "I am a test"}}

    last = ledger_after[-1]
    assert isinstance(last, dict)
    eff = last.get("effect")
    assert isinstance(eff, dict)
    assert eff.get("type") == "emit_event"
    payload = eff.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("name") == "abstract.status"
