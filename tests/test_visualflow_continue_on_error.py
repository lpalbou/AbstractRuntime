"""continueOnError -> _absorb_failure mapping (flow's c2851 ask, ruling c2896).

Shape (c): a node-config flag on DIRECT effect nodes compiles to the shipped
`_absorb_failure` payload key — a terminally failed effect lands
{"ok": False, "absorbed_failure": ...} at the node's result and the run
continues. Absent flag = terminate-run, byte-unchanged existing flows.

LEDGER HONESTY (correcting the c2896 reply's replay claim, on the record):
the absorbed step's ledger record is a FAILED StepRecord — the failure is
recorded honestly and absorption only converts the RUN-LEVEL outcome. On
crash-replay before the post-absorb save, the effect re-executes exactly
like any failed effect under retry; the absorbed RESULT becomes durable at
the post-absorb save.
"""

from __future__ import annotations

from abstractruntime.core.models import RunStatus
from abstractruntime.core.runtime import Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler.compiler import compile_flow
from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json


def _flow(continue_on_error: bool):
    effect_config = {"provider": "none", "model": "none"}
    if continue_on_error:
        effect_config["continueOnError"] = True
    return load_visualflow_json(
        {
            "id": "coe",
            "name": "coe",
            "nodes": [
                {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
                {
                    "id": "call",
                    "type": "llm_call",
                    "data": {
                        "nodeType": "llm_call",
                        "effectConfig": effect_config,
                        "pinDefaults": {"prompt": "hello"},
                    },
                },
                {
                    "id": "end",
                    "type": "on_flow_end",
                    "data": {
                        "nodeType": "on_flow_end",
                        "inputs": [{"id": "exec-in", "label": "", "type": "execution"}],
                    },
                },
            ],
            "edges": [
                {"source": "start", "sourceHandle": "exec-out", "target": "call", "targetHandle": "exec-in"},
                {"source": "call", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"},
            ],
        }
    )


def _run(continue_on_error: bool):
    # No LLM_CALL handler registered -> the effect fails terminally
    # (non-retryable "no handler" failure), which is exactly the outage
    # class flow named (dead verifier endpoint).
    wf = compile_flow(visual_to_flow(_flow(continue_on_error)))
    runs, ledger = InMemoryRunStore(), InMemoryLedgerStore()
    rt = Runtime(run_store=runs, ledger_store=ledger)
    rid = rt.start(workflow=wf, vars={})
    state = rt.tick(workflow=wf, run_id=rid, max_steps=20)
    return state, ledger.list(rid)


def test_continue_on_error_absorbs_and_run_completes() -> None:
    state, records = _run(continue_on_error=True)
    assert state.status == RunStatus.COMPLETED, f"run continues past the failed effect: {state.error}"
    absorbed = state.vars.get("_temp", {}).get("effects", {}).get("call")
    assert isinstance(absorbed, dict)
    assert absorbed.get("ok") is False
    assert absorbed.get("absorbed_failure"), "the error text is readable at the node result"


def test_without_flag_the_run_still_fails() -> None:
    state, _ = _run(continue_on_error=False)
    assert state.status == RunStatus.FAILED, "absent flag = today's terminate-run"


def test_absorbed_step_ledger_record_is_honestly_failed() -> None:
    """Absorption converts the RUN outcome, never the RECORD: the ledger
    keeps the failure (correcting c2896's 'COMPLETED record' claim)."""
    state, records = _run(continue_on_error=True)
    assert state.status == RunStatus.COMPLETED
    call_steps = [
        r for r in records
        if (r.get("node_id") if isinstance(r, dict) else r.node_id) == "call"
    ]
    assert call_steps, "the effect step is in the ledger"

    def _status(rec):
        return rec.get("status") if isinstance(rec, dict) else getattr(rec, "status", None)

    statuses = {str(_status(r)) for r in call_steps}
    assert not any("completed" in s.lower() for s in statuses), (
        "an absorbed failure must never masquerade as a completed effect"
    )


def test_subworkflow_nodes_never_inherit_the_flag() -> None:
    """Ruling scope: continueOnError covers DIRECT effects only — the
    compiler must not stamp start_subworkflow payloads."""
    from abstractruntime.visualflow_compiler.compiler import _create_effect_node_handler

    handler = _create_effect_node_handler(
        node_id="sub",
        effect_type="start_subworkflow",
        effect_config={"workflow_id": "w", "continueOnError": True},
        next_node=None,
        input_key=None,
        output_key=None,
        data_aware_handler=None,
    )
    # The stamp wrapper is bypassed for start_subworkflow: the returned
    # handler is the raw base handler (no `stamped` closure around it).
    assert handler.__name__ != "stamped"
