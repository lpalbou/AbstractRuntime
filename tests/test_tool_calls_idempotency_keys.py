from __future__ import annotations

from pathlib import Path

import pytest

from abstractruntime import (
    Effect,
    EffectType,
    RunStatus,
    StepPlan,
    WorkflowSpec,
    compute_idempotency_key,
)
from abstractruntime.core.runtime import Runtime, _ensure_tool_calls_have_runtime_ids
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore


@pytest.mark.basic
def test_tool_calls_idempotency_ignores_model_call_id_and_allowlist_order() -> None:
    effect1 = Effect(
        type=EffectType.TOOL_CALLS,
        payload={
            "tool_calls": [
                {"name": "t", "arguments": {"x": 1}, "call_id": "call_a"},
                {"name": "t", "arguments": {"x": 2}, "call_id": "call_b"},
            ],
            "allowed_tools": ["b", "a"],
        },
    )
    effect2 = Effect(
        type=EffectType.TOOL_CALLS,
        payload={
            "tool_calls": [
                {"name": "t", "arguments": {"x": 1}, "call_id": "call_x"},
                {"name": "t", "arguments": {"x": 2}, "call_id": "call_y"},
            ],
            "allowed_tools": ["a", "b"],
        },
    )

    key1 = compute_idempotency_key(run_id="run-1", node_id="node-1", effect=effect1)
    key2 = compute_idempotency_key(run_id="run-1", node_id="node-1", effect=effect2)
    assert key1 == key2


@pytest.mark.basic
def test_runtime_call_ids_are_stable_and_call_id_falls_back() -> None:
    effect = Effect(
        type=EffectType.TOOL_CALLS,
        payload={
            "tool_calls": [
                {"name": "t", "arguments": {"x": 1}, "call_id": "provider_call_1"},
                {"name": "t", "arguments": {"x": 2}},  # missing call_id
            ],
            "allowed_tools": ["b", "a", "a"],
        },
    )
    key = compute_idempotency_key(run_id="run-1", node_id="node-1", effect=effect)
    normalized = _ensure_tool_calls_have_runtime_ids(effect=effect, idempotency_key=key)

    calls = normalized.payload.get("tool_calls")
    assert isinstance(calls, list) and len(calls) == 2
    assert calls[0].get("call_id") == "provider_call_1"
    assert isinstance(calls[0].get("runtime_call_id"), str) and calls[0]["runtime_call_id"]

    # When provider call_id is absent, the runtime uses the stable runtime id as call_id.
    assert calls[1].get("call_id") == calls[1].get("runtime_call_id")

    # Allowlists are canonicalized deterministically.
    assert normalized.payload.get("allowed_tools") == ["a", "b"]

    # Runtime ids are stable across model/provider call-id drift.
    effect2 = Effect(
        type=EffectType.TOOL_CALLS,
        payload={
            "tool_calls": [
                {"name": "t", "arguments": {"x": 1}, "call_id": "provider_call_2"},
                {"name": "t", "arguments": {"x": 2}},  # missing call_id again
            ],
            "allowed_tools": ["a", "b"],
        },
    )
    key2 = compute_idempotency_key(run_id="run-1", node_id="node-1", effect=effect2)
    assert key2 == key
    normalized2 = _ensure_tool_calls_have_runtime_ids(effect=effect2, idempotency_key=key2)
    calls2 = normalized2.payload.get("tool_calls")
    assert isinstance(calls2, list) and len(calls2) == 2
    assert [c.get("runtime_call_id") for c in calls2] == [c.get("runtime_call_id") for c in calls]


@pytest.mark.integration
def test_tool_calls_restart_reuses_prior_result_even_if_call_id_changes(tmp_path: Path) -> None:
    base = tmp_path / "stores"
    run_store = JsonFileRunStore(base)
    ledger_store = JsonlLedgerStore(base)

    executed = {"count": 0}

    def append_line(*, file_path: str, text: str) -> dict:
        executed["count"] += 1
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
        return {"ok": True}

    tools = MappingToolExecutor({"append_line": append_line})
    rt1 = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)},
    )

    out_file = tmp_path / "out.txt"

    def _workflow(call_id: str) -> WorkflowSpec:
        tool_calls = [{"name": "append_line", "arguments": {"file_path": str(out_file), "text": "x"}, "call_id": call_id}]

        def tools_node(run, ctx) -> StepPlan:
            del ctx
            return StepPlan(
                node_id="TOOLS",
                effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": tool_calls}, result_key="tool_results"),
                next_node="DONE",
            )

        def done_node(run, ctx) -> StepPlan:
            del ctx
            return StepPlan(node_id="DONE", complete_output={"tool_results": run.vars.get("tool_results")})

        return WorkflowSpec(workflow_id="tool_idempotency_restart_test", entry_node="TOOLS", nodes={"TOOLS": tools_node, "DONE": done_node})

    wf1 = _workflow("provider_call_1")
    run_id = rt1.start(workflow=wf1)
    state1 = rt1.tick(workflow=wf1, run_id=run_id)
    assert state1.status == RunStatus.COMPLETED
    assert executed["count"] == 1
    assert out_file.read_text(encoding="utf-8") == "x\n"

    # Ledger should include runtime_call_id in both effect payload and results.
    records = ledger_store.list(run_id)
    completed_tool = [
        r
        for r in records
        if r.get("status") == "completed"
        and isinstance(r.get("effect"), dict)
        and r.get("effect", {}).get("type") == EffectType.TOOL_CALLS.value
    ]
    assert completed_tool, "Expected at least one completed TOOL_CALLS record"
    tool_record = completed_tool[-1]
    payload = tool_record.get("effect", {}).get("payload", {})
    calls = payload.get("tool_calls") if isinstance(payload, dict) else None
    assert isinstance(calls, list) and isinstance(calls[0], dict)
    assert isinstance(calls[0].get("runtime_call_id"), str) and calls[0]["runtime_call_id"]
    results = tool_record.get("result", {}).get("results") if isinstance(tool_record.get("result"), dict) else None
    assert isinstance(results, list) and isinstance(results[0], dict)
    assert results[0].get("runtime_call_id") == calls[0].get("runtime_call_id")

    # Simulate restart: reset run state and change provider call id.
    run = run_store.load(run_id)
    run.status = RunStatus.RUNNING
    run.current_node = "TOOLS"
    run.output = None
    run.vars.pop("tool_results", None)
    run_store.save(run)

    wf2 = _workflow("provider_call_2")
    rt2 = Runtime(
        run_store=JsonFileRunStore(base),
        ledger_store=JsonlLedgerStore(base),
        effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)},
    )
    state2 = rt2.tick(workflow=wf2, run_id=run_id)
    assert state2.status == RunStatus.COMPLETED
    assert executed["count"] == 1  # tool NOT executed again
    assert out_file.read_text(encoding="utf-8") == "x\n"
