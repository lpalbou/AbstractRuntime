"""A PAYLOAD-SHAPE refusal is deterministic; retrying it only multiplies the wait.

Root cause of the `basic-llm-test` report: a run started with no prompt (the
Run dialog's field left empty) was ACCEPTED, then failed three attempts later
with `Effect failed after 3 attempts: llm_call requires payload.prompt or
payload.messages` -- a runtime-internal payload key, three times the latency,
and no mention of the flow input the caller actually omitted. The provider
cascade was never involved; the message made it look as though it was.

The identical payload meets the identical refusal on every attempt, so it is
not retryable, and it names where to supply the missing value.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.integrations.abstractcore.effect_handlers import (
    make_llm_call_handler,
    make_tool_calls_handler,
    make_tool_invoke_handler,
)


class _NeverCalledLLM:
    def generate(self, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - must not run
        raise AssertionError("a payload-shape refusal must never reach the provider")


def _run_state(node: str = "node-2") -> RunState:
    return RunState(
        run_id="run-test",
        workflow_id="basic-llm-test@0.0.0:ec5a574d",
        status=RunStatus.RUNNING,
        current_node=node,
        vars={},
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt": "   "},
        {"messages": []},
        {"provider": "lmstudio", "model": "qwen3-0.6b"},
    ],
)
def test_a_missing_prompt_fails_once_and_says_where_to_supply_it(payload: Dict[str, Any]) -> None:
    handler = make_llm_call_handler(llm=_NeverCalledLLM(), artifact_store=None)
    outcome = handler(_run_state(), Effect(type=EffectType.LLM_CALL, payload=dict(payload)), None)

    assert outcome.status == "failed"
    assert outcome.retryable is False, "an empty payload cannot become non-empty on attempt 2"
    assert "requires payload.prompt or payload.messages" in (outcome.error or "")
    assert "node node-2" in (outcome.error or ""), "name the node that refused"
    assert "input_data" in (outcome.error or ""), "name the run input that fills it"


def test_the_media_variant_of_the_refusal_is_equally_deterministic() -> None:
    handler = make_llm_call_handler(llm=_NeverCalledLLM(), artifact_store=None)
    outcome = handler(
        _run_state(),
        Effect(type=EffectType.LLM_CALL, payload={"params": {"output": [{"type": "text"}]}}),
        None,
    )

    assert outcome.status == "failed"
    assert outcome.retryable is False
    assert "payload.text" in (outcome.error or "")


def test_tool_effect_shape_refusals_are_not_retried() -> None:
    tool_calls = make_tool_calls_handler(tools=None, artifact_store=None, run_store=None)
    outcome = tool_calls(_run_state(), Effect(type=EffectType.TOOL_CALLS, payload={}), None)
    assert outcome.status == "failed" and outcome.retryable is False
    assert "payload.tool_calls" in (outcome.error or "")

    tool_invoke = make_tool_invoke_handler(tools=None, artifact_store=None, run_store=None)
    outcome = tool_invoke(_run_state(), Effect(type=EffectType.TOOL_INVOKE, payload={}), None)
    assert outcome.status == "failed" and outcome.retryable is False
    assert "payload.name" in (outcome.error or "")


def test_a_prompt_that_is_present_still_reaches_the_provider() -> None:
    """The refusal must not have become over-eager: a real prompt runs."""

    class _LLM:
        def __init__(self) -> None:
            self.calls: List[Optional[str]] = []

        def generate(self, *, prompt: str, **kwargs: Any) -> Dict[str, Any]:
            self.calls.append(prompt)
            return {"content": "ok", "finish_reason": "stop", "metadata": {}}

    llm = _LLM()
    handler = make_llm_call_handler(llm=llm, artifact_store=None)
    outcome = handler(_run_state(), Effect(type=EffectType.LLM_CALL, payload={"prompt": "hello"}), None)

    assert outcome.status == "completed"
    assert llm.calls and "hello" in str(llm.calls[0])
