from __future__ import annotations

import pytest

from abstractruntime import (
    Effect,
    EffectType,
    InMemoryArtifactStore,
    InMemoryLedgerStore,
    InMemoryRunStore,
    RUN_HISTORY_BUNDLE_VERSION_V1,
    RunState,
    RunStatus,
    Runtime,
    StepPlan,
    WorkflowSpec,
    export_run_history_bundle,
    persist_workflow_snapshot,
)
from abstractruntime.core.models import StepRecord, StepStatus


pytestmark = pytest.mark.basic


def test_persist_workflow_snapshot_persists_ref_and_artifact() -> None:
    run_store = InMemoryRunStore()
    artifact_store = InMemoryArtifactStore()

    wf = WorkflowSpec(workflow_id="wf", entry_node="done", nodes={"done": lambda run, ctx: StepPlan(node_id="done", complete_output={"ok": True})})
    rt = Runtime(run_store=run_store, ledger_store=InMemoryLedgerStore(), effect_handlers={})
    run_id = rt.start(workflow=wf, vars={"context": {"messages": []}})

    ref = persist_workflow_snapshot(
        run_store=run_store,
        artifact_store=artifact_store,
        run_id=run_id,
        workflow_id="wf",
        snapshot={"kind": "unit", "wf": "wf"},
        format="unit_json",
    )

    assert isinstance(ref, dict)
    assert ref.get("workflow_id") == "wf"
    assert ref.get("format") == "unit_json"
    assert isinstance(ref.get("sha256"), str) and ref["sha256"]
    assert isinstance(ref.get("artifact_id"), str) and ref["artifact_id"]

    reloaded = run_store.load(run_id)
    assert reloaded is not None
    runtime_ns = (reloaded.vars or {}).get("_runtime") if isinstance(reloaded.vars, dict) else None
    assert isinstance(runtime_ns, dict)
    assert runtime_ns.get("workflow_snapshot", {}).get("artifact_id") == ref["artifact_id"]

    stored = artifact_store.load_json(ref["artifact_id"])
    assert stored == {"kind": "unit", "wf": "wf"}


def test_export_run_history_bundle_filters_input_data_and_tails_ledgers() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    artifact_store = InMemoryArtifactStore()

    run_id = "run_1"
    run_store.save(
        RunState(
            run_id=run_id,
            workflow_id="wf",
            status=RunStatus.COMPLETED,
            current_node="done",
            vars={
                "prompt": "hi",
                "context": {"messages": [{"role": "user", "content": "hi"}], "attachments": []},
                "_runtime": {"secret": True},
            },
            output={"response": "ok"},
            error=None,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            actor_id="tester",
            session_id="sess1",
            parent_run_id=None,
            waiting=None,
        )
    )
    for i in range(5):
        ledger_store.append(
            StepRecord(
                run_id=run_id,
                step_id=f"s{i+1}",
                node_id="n",
                status=StepStatus.COMPLETED,
                effect={"type": "llm_call", "payload": {"i": i + 1}, "result_key": None},
                result={"ok": True, "i": i + 1},
                error=None,
                started_at=f"2026-01-01T00:00:0{i}+00:00",
                ended_at=f"2026-01-01T00:00:0{i}+00:00",
                actor_id="tester",
                session_id="sess1",
                attempt=1,
                idempotency_key=None,
                prev_hash=None,
                record_hash=None,
                signature=None,
            )
        )

    bundle = export_run_history_bundle(
        run_id=run_id,
        run_store=run_store,
        ledger_store=ledger_store,
        artifact_store=artifact_store,
        include_subruns=False,
        include_session=True,
        session_turn_limit=50,
        ledger_mode="tail",
        ledger_max_items=2,
    )

    assert bundle.get("version") == RUN_HISTORY_BUNDLE_VERSION_V1
    assert bundle.get("root_run_id") == run_id
    assert bundle.get("run", {}).get("run_id") == run_id
    assert bundle.get("run", {}).get("status") in {RunStatus.RUNNING.value, RunStatus.COMPLETED.value, RunStatus.WAITING.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}

    # Input data: private namespaces are filtered out.
    input_data = bundle.get("input_data") or {}
    assert "_runtime" not in input_data
    assert input_data.get("prompt") == "hi"

    ledgers = bundle.get("ledgers") or {}
    assert run_id in ledgers
    ledger = ledgers[run_id]
    assert ledger.get("total") >= 1
    assert ledger.get("cursor_start") == 4
    assert len(ledger.get("items") or []) <= 2
    items = ledger.get("items") or []
    if len(items) == 2:
        assert items[0]["cursor"] == ledger["cursor_start"]
        assert items[1]["cursor"] == ledger["cursor_start"] + 1

    # Session turns: should include the run (chat-like vars shape).
    turns = (bundle.get("session") or {}).get("turns") or []
    assert any(t.get("run_id") == run_id and t.get("kind") == "chat" for t in turns)
    turn0 = next((t for t in turns if t.get("run_id") == run_id), None)
    assert turn0 is not None
    assert turn0.get("answer") == "ok"


def test_resume_appends_resume_record_to_ledger() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()

    def start(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="start",
            effect=Effect(type=EffectType.ASK_USER, payload={"prompt": "name?"}, result_key="answer"),
            next_node="done",
        )

    def done(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"answer": run.vars.get("answer")})

    wf = WorkflowSpec(workflow_id="wf_wait", entry_node="start", nodes={"start": start, "done": done})
    rt = Runtime(run_store=run_store, ledger_store=ledger_store, effect_handlers={})

    run_id = rt.start(workflow=wf, vars={})
    st = rt.tick(workflow=wf, run_id=run_id, max_steps=5)
    assert st.status == RunStatus.WAITING

    rt.resume(workflow=wf, run_id=run_id, wait_key=st.waiting.wait_key if st.waiting else None, payload={"response": "alice"}, max_steps=5)

    records = ledger_store.list(run_id)
    assert any(isinstance(r, dict) and isinstance(r.get("effect"), dict) and r["effect"].get("type") == "resume" for r in records)
