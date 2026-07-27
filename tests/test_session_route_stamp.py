"""Session-route stamp on delegated-sight tools (vision ruling, 2026-07-26).

Laurent's ruling: fallbacks are SOLELY for models without vision — delegated
sight must use the session model. Core's analyze_media resolves the run's own
route FIRST via a schema-hidden `_session_route` param; the runtime half
(pinned here) is the TRUST BOUNDARY stamp in the TOOL_CALLS handler:

- derived from `_runtime.provider` / `_runtime.model` on the RUN, never from
  payload claims (a model-supplied route is always popped — the
  `_agora_agent` / `_registry_namespace` force-stamp discipline);
- injected ONLY for declared consumer tools (`_SESSION_ROUTE_TOOL_NAMES`,
  exact names — a custom tool that merely sounds sight-delegating never
  receives the route);
- absent route vars stamp NOTHING (core's graceful degradation: unstamped =
  pre-ruling fallback behavior, byte-identical).
"""

from __future__ import annotations

from typing import Any, Dict, List

from abstractruntime.core.models import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler


class _CapturingExecutor:
    """Records the exact arguments each tool call executes with."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def execute(self, *, tool_calls: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        results = []
        for tc in tool_calls:
            self.calls.append(dict(tc))
            results.append(
                {
                    "call_id": str(tc.get("call_id") or ""),
                    "name": str(tc.get("name") or ""),
                    "success": True,
                    "output": "ok",
                }
            )
        return {"results": results}


def _effect(name: str, arguments: Dict[str, Any]) -> Effect:
    return Effect(
        type=EffectType.TOOL_CALLS,
        payload={"tool_calls": [{"call_id": "c1", "name": name, "arguments": arguments}]},
    )


def _run(vars: Dict[str, Any]) -> RunState:
    return RunState.new(workflow_id="wf", entry_node="n", session_id="s1", vars=vars)


def test_analyze_media_receives_the_derived_route() -> None:
    ex = _CapturingExecutor()
    handler = make_tool_calls_handler(tools=ex)
    run = _run({"_runtime": {"provider": "lmstudio", "model": "qwen3-vl-8b"}})

    outcome = handler(run, _effect("analyze_media", {"path": "a.png"}), None)
    assert str(outcome.status) == "completed"
    args = ex.calls[0]["arguments"]
    assert args["_session_route"] == {"provider": "lmstudio", "model": "qwen3-vl-8b"}
    assert args["path"] == "a.png"


def test_payload_claimed_route_is_overwritten_never_honored() -> None:
    """A model claiming another route is re-stamped from run vars."""
    ex = _CapturingExecutor()
    handler = make_tool_calls_handler(tools=ex)
    run = _run({"_runtime": {"provider": "openai", "model": "gpt-5.6-sol"}})

    outcome = handler(
        run,
        _effect("analyze_media", {"path": "a.png", "_session_route": {"provider": "evil", "model": "spoof"}}),
        None,
    )
    assert str(outcome.status) == "completed"
    assert ex.calls[0]["arguments"]["_session_route"] == {"provider": "openai", "model": "gpt-5.6-sol"}


def test_no_route_vars_stamps_nothing_and_strips_claims() -> None:
    """Unstamped = pre-ruling behavior; a spoofed route never survives."""
    ex = _CapturingExecutor()
    handler = make_tool_calls_handler(tools=ex)
    run = _run({})

    outcome = handler(
        run,
        _effect("analyze_media", {"path": "a.png", "_session_route": {"provider": "evil", "model": "spoof"}}),
        None,
    )
    assert str(outcome.status) == "completed"
    assert "_session_route" not in ex.calls[0]["arguments"]


def test_partial_route_stamps_what_exists() -> None:
    """A partial route stamps what exists — so core's degradation is LOUD.

    Core's consumer requires BOTH provider and model and discards a partial
    stamp with a labeled '#FALLBACK: incomplete _session_route stamp'
    warning (common_tools.py). Stamping the partial anyway is deliberate:
    an ABSENT stamp degrades to the configured fallback silently, while a
    partial stamp makes the degradation observable. Model-only sessions do
    NOT get session-route sight today — they get the fallback, loudly
    (adversary P2 finding, 2026-07-26)."""
    ex = _CapturingExecutor()
    handler = make_tool_calls_handler(tools=ex)
    run = _run({"_runtime": {"model": "qwen3-vl-8b"}})

    handler(run, _effect("analyze_media", {"path": "a.png"}), None)
    assert ex.calls[0]["arguments"]["_session_route"] == {"provider": None, "model": "qwen3-vl-8b"}


def test_undeclared_tools_never_receive_the_route_but_spoofs_still_strip() -> None:
    """Exact-name gate: other tools get no stamp; claimed routes are popped everywhere."""
    ex = _CapturingExecutor()
    handler = make_tool_calls_handler(tools=ex)
    run = _run({"_runtime": {"provider": "lmstudio", "model": "qwen3-vl-8b"}})

    outcome = handler(
        run,
        _effect("read_file", {"path": "a.txt", "_session_route": {"provider": "evil", "model": "spoof"}}),
        None,
    )
    assert str(outcome.status) == "completed"
    args = ex.calls[0]["arguments"]
    assert "_session_route" not in args
    assert args["path"] == "a.txt"
