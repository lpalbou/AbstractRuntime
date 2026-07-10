"""Visual wait_event D3 passthrough (flow's follow-through ask, 2026-07-10).

The runtime's WAIT_EVENT accepts `until` + `details`; the VISUAL node's
adapter must pass both through so visual residents get durable idle
deadlines and self-describing parks — and stay byte-unchanged when the
pins are absent (older flows).
"""

from __future__ import annotations

from typing import Any

from abstractruntime.core.models import RunStatus, WaitReason
from abstractruntime.core.runtime import Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler.adapters.effect_adapter import (
    create_wait_event_handler,
)


def _run(vars: dict) -> Any:
    handler = create_wait_event_handler("WAIT", None, input_key="_in", output_key="_temp.evt")
    wf = WorkflowSpec(workflow_id="wf_visual_wait", entry_node="WAIT", nodes={"WAIT": handler})
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    run_id = rt.start(workflow=wf, vars={"_in": vars})
    return rt.tick(workflow=wf, run_id=run_id, max_steps=5)


def test_until_and_details_pass_through_the_visual_node() -> None:
    state = _run({
        "event_key": "visitor_input",
        "until": "2999-01-01T12:00:00+02:00",
        "details": {"kind": "visitor_message"},
    })
    assert state.status == RunStatus.WAITING
    assert state.waiting.reason == WaitReason.EVENT
    assert state.waiting.wait_key == "visitor_input"
    assert state.waiting.until == "2999-01-01T10:00:00+00:00"  # UTC-normalized
    assert (state.waiting.details or {}).get("kind") == "visitor_message"


def test_absent_pins_stay_absent() -> None:
    state = _run({"event_key": "plain"})
    assert state.status == RunStatus.WAITING
    assert state.waiting.until is None
    assert state.waiting.details is None
