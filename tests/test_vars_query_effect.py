from __future__ import annotations

import json
from typing import Any, Dict

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_vars_query_effect_reads_scratchpad_paths_and_keys() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)

    vars: Dict[str, Any] = {
        "context": {"task": "t", "messages": []},
        "scratchpad": {"foo": {"bar": [1, 2, 3]}},
        "_runtime": {},
        "_temp": {},
        "_limits": {},
    }

    def inspect_value(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="inspect_value",
            effect=Effect(
                type=EffectType.VARS_QUERY,
                payload={"path": "scratchpad.foo.bar[1]"},
                result_key="_temp.value",
            ),
            next_node="inspect_keys",
        )

    def inspect_keys(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="inspect_keys",
            effect=Effect(
                type=EffectType.VARS_QUERY,
                payload={"path": "scratchpad.foo", "keys_only": True},
                result_key="_temp.keys",
            ),
            next_node="inspect_missing",
        )

    def inspect_missing(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="inspect_missing",
            effect=Effect(
                type=EffectType.VARS_QUERY,
                payload={"path": "scratchpad.nope"},
                result_key="_temp.missing",
            ),
            next_node="done",
        )

    def done(run, ctx) -> StepPlan:
        temp = run.vars.get("_temp") if isinstance(run.vars.get("_temp"), dict) else {}
        return StepPlan(
            node_id="done",
            complete_output={
                "value": temp.get("value"),
                "keys": temp.get("keys"),
                "missing": temp.get("missing"),
            },
        )

    wf = WorkflowSpec(
        workflow_id="wf_vars_query",
        entry_node="inspect_value",
        nodes={
            "inspect_value": inspect_value,
            "inspect_keys": inspect_keys,
            "inspect_missing": inspect_missing,
            "done": done,
        },
    )

    run_id = runtime.start(workflow=wf, vars=vars)
    state = runtime.tick(workflow=wf, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    out = state.output or {}

    value = out.get("value")
    assert isinstance(value, dict)
    v_results = value.get("results")
    assert isinstance(v_results, list) and v_results
    v_text = str(v_results[0].get("output") or "")
    v_payload = json.loads(v_text)
    assert v_payload["path"] == "scratchpad.foo.bar[1]"
    assert v_payload["value"] == 2

    keys = out.get("keys")
    assert isinstance(keys, dict)
    k_results = keys.get("results")
    assert isinstance(k_results, list) and k_results
    k_text = str(k_results[0].get("output") or "")
    k_payload = json.loads(k_text)
    assert k_payload["path"] == "scratchpad.foo"
    assert "bar" in (k_payload.get("keys") or [])

    missing = out.get("missing")
    assert isinstance(missing, dict)
    m_results = missing.get("results")
    assert isinstance(m_results, list) and m_results
    assert m_results[0].get("success") is False
    assert "Missing key" in str(m_results[0].get("error") or "")

