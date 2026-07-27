"""send_email_recipient@v1 refiner pins (laurent dm#244; fable5 spec c4691).

Recipient == the gateway-injected operator address -> auto; ANY other
recipient anywhere -> ask. Deny-safe at every gap the adversary named.
"""
from __future__ import annotations

from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.integrations.abstractcore.effect_handlers import (
    _execute_with_run_policy,
    _send_email_recipient_refiner,
)


def _run(self_email=None, policy=None) -> RunState:
    rt = {}
    if self_email is not None:
        rt["operator_email"] = self_email
    if policy is not None:
        rt["tool_policy"] = policy
    return RunState(run_id="r", workflow_id="w", status=RunStatus.RUNNING,
                    current_node="n", vars={"_runtime": rt})


def _call(args):
    return {"name": "send_email", "arguments": args}


def test_refiner_self_only_auto_others_ask() -> None:
    r = _run("op@self.com")
    assert _send_email_recipient_refiner(_call({"to": "op@self.com", "subject": "x"}), r) == "auto"
    assert _send_email_recipient_refiner(_call({"to": "OP@SELF.COM"}), r) == "auto", "case-insensitive"
    assert _send_email_recipient_refiner(_call({"to": "op@self.com, other@x.com"}), r) == "ask"
    assert _send_email_recipient_refiner(_call({"to": "op@self.com", "cc": "e@x.com"}), r) == "ask"
    assert _send_email_recipient_refiner(_call({"to": "op@self.com", "bcc": ["e@x.com"]}), r) == "ask"


def test_refiner_deny_safe_gaps() -> None:
    r = _run("op@self.com")
    # P0-1 vacuous empty
    assert _send_email_recipient_refiner(_call({"to": "", "subject": "x"}), r) == "ask"
    assert _send_email_recipient_refiner(_call({"subject": "x"}), r) == "ask"
    # P0-2 wrapper differential
    assert _send_email_recipient_refiner(_call({"to": "op@self.com", "arguments": {"to": "e@x.com"}}), r) == "ask"
    # P1-6 display-name/group token compared verbatim (no bracket extraction)
    assert _send_email_recipient_refiner(_call({"to": "Operator <op@self.com>"}), r) == "ask"
    # unparseable / non-dict args
    assert _send_email_recipient_refiner(_call("not-json"), r) == "ask"
    assert _send_email_recipient_refiner(_call(["x"]), r) == "ask"
    # self-value absent -> ask (the ceiling stands)
    assert _send_email_recipient_refiner(_call({"to": "op@self.com"}), _run(None)) == "ask"


def test_refiner_homoglyph_never_matches() -> None:
    # P1-5: a Cyrillic-o domain must NOT fold to the Latin operator address.
    r = _run("op@self.com")
    assert _send_email_recipient_refiner(_call({"to": "o\u0440@self.com"}), r) == "ask"


class _Gated:
    def execute(self, *, tool_calls):
        return {"mode": "approval_required", "wait_reason": "user",
                "tool_calls": tool_calls, "details": {"kind": "tool_approval"}}

    def execute_approved(self, *, tool_calls):
        return {"mode": "executed", "results": [
            {"name": c["name"], "output": "sent"} for c in tool_calls]}


def test_dispatch_downgrades_self_send_under_a_grant() -> None:
    """End-to-end through _execute_with_run_policy: a run granted send_email
    at a rank below outreach still ASKS for a stranger recipient but the
    refiner AUTOs a self-only send - even though send_email is
    model_controlled_destination (the designed per-argument exception)."""
    pol = {"auto_approve_max_risk_rank": 1}  # below outreach; ceiling alone would ask
    self_run = _run("op@self.com", pol)
    out = _execute_with_run_policy(_Gated(), [_call({"to": "op@self.com", "subject": "hi"})], self_run)
    assert out["mode"] == "executed", "self-only send auto's via the refiner"

    out = _execute_with_run_policy(_Gated(), [_call({"to": "stranger@x.com"})], _run("op@self.com", pol))
    assert out["mode"] == "approval_required", "stranger send still asks"


def test_dispatch_deny_safe_without_self_value() -> None:
    """No operator_email injected (gateway didn't set it / older gateway):
    the refiner never fires, send_email stays gated."""
    pol = {"auto_approve_max_risk_rank": 1}
    out = _execute_with_run_policy(_Gated(), [_call({"to": "op@self.com"})], _run(None, pol))
    assert out["mode"] == "approval_required", "no self-value -> ask, deny-safe"
