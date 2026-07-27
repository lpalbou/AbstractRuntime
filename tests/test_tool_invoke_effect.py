"""TOOL_INVOKE pins (flow c4206 contract; commons c4204 ruling).

The trust model, pinned from both sides:
- an AUTHORED deterministic invocation executes approval-REQUIRING tools
  WITHOUT a wait (the whole point);
- the same tool through TOOL_CALLS still gates (agent lane unchanged);
- the result envelope matches TOOL_CALLS (results[0].output raw) so
  flow's first.output -> pin mapping works verbatim;
- one call per effect; a missing name refuses loudly.
"""
from __future__ import annotations

from typing import Any, Dict, List

from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.integrations.abstractcore.effect_handlers import (
    build_effect_handlers,
    make_tool_invoke_handler,
)


class _GatedExecutor:
    """Approval-gated executor double: execute() gates everything,
    execute_approved() runs - the ApprovalGatedToolExecutor contract."""

    def __init__(self) -> None:
        self.approved_calls: List[Dict[str, Any]] = []

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "mode": "approval_required",
            "wait_reason": "user",
            "tool_calls": tool_calls,
            "details": {"kind": "tool_approval"},
        }

    def execute_approved(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.approved_calls.extend(tool_calls)
        return {
            "mode": "executed",
            "results": [
                {"name": c.get("name"), "output": {"path": "/ws/p.jpg", "media": ["m1"]}}
                for c in tool_calls
            ],
        }


def _run() -> RunState:
    return RunState(
        run_id="r-ti", workflow_id="w", status=RunStatus.RUNNING,
        current_node="n1", vars={},
    )


def test_authored_invoke_skips_the_gate_and_returns_raw_output() -> None:
    ex = _GatedExecutor()
    handler = make_tool_invoke_handler(tools=ex)
    effect = Effect(
        type=EffectType.TOOL_INVOKE,
        payload={"name": "camera_capture_photo", "arguments": {"camera": "uid0"}},
        result_key="out",
    )
    outcome = handler(_run(), effect, "next")
    assert str(outcome.status) in ("completed", "EffectStatus.COMPLETED") or getattr(outcome.status, "value", outcome.status) == "completed", f"no wait: {outcome!r}"
    results = (outcome.result or {}).get("results")
    assert results and results[0]["output"] == {"path": "/ws/p.jpg", "media": ["m1"]}, \
        "raw tool output verbatim (flow's first.output mapping)"
    assert ex.approved_calls and ex.approved_calls[0]["name"] == "camera_capture_photo"


def test_same_tool_through_tool_calls_still_gates() -> None:
    ex = _GatedExecutor()
    handlers = build_effect_handlers(llm=None, tools=ex)  # type: ignore[arg-type]
    effect = Effect(
        type=EffectType.TOOL_CALLS,
        payload={"tool_calls": [{"name": "camera_capture_photo", "arguments": {}}]},
        result_key="out",
    )
    outcome = handlers[EffectType.TOOL_CALLS](_run(), effect, "next")
    assert getattr(outcome.status, "value", outcome.status) == "waiting", "agent lane unchanged - the gate holds"


def test_invoke_refusals_are_loud() -> None:
    handler = make_tool_invoke_handler(tools=_GatedExecutor())
    out = handler(_run(), Effect(type=EffectType.TOOL_INVOKE, payload={}), None)
    assert getattr(out.status, "value", out.status) == "failed" and "name" in str(out.error)
    out = handler(_run(), Effect(
        type=EffectType.TOOL_INVOKE,
        payload={"name": "x", "arguments": "not-a-dict"}), None)
    assert getattr(out.status, "value", out.status) == "failed" and "arguments" in str(out.error)


def test_registered_beside_tool_calls() -> None:
    handlers = build_effect_handlers(llm=None, tools=_GatedExecutor())  # type: ignore[arg-type]
    assert EffectType.TOOL_INVOKE in handlers and EffectType.TOOL_CALLS in handlers


def test_camera_node_verb_is_baked_never_authored() -> None:
    """The compiler-side forgery guard (flow's dm#49 nodes, my owner review):
    a camera node's tool verb comes ONLY from CAMERA_TOOL_INVOKE_VERBS keyed
    by node TYPE - a hostile effectConfig/pin naming another tool is ignored.
    An author-editable verb would reopen the approval bypass the effect
    class exists to close."""
    from abstractruntime.visualflow_compiler.compiler import (
        _create_tool_invoke_base_handler,
    )

    handler = _create_tool_invoke_base_handler(
        node_id="n1", node_type="camera_capture_photo",
        next_node=None, input_key="_temp.in", output_key="out",
    )

    class _Run:
        vars = {"_temp.in": {"camera": "uid-1", "timeout_s": 5,
                             "name": "execute_command",  # hostile: not an arg pin
                             "tool": "delete_everything"}}

    plan = handler(_Run(), None)
    assert plan.effect is not None
    assert plan.effect.payload["name"] == "camera_capture_photo", "verb baked from node type"
    args = plan.effect.payload["arguments"]
    assert args == {"camera": "uid-1", "timeout_s": 5}, \
        f"only declared arg pins forwarded, hostile keys dropped: {args!r}"
