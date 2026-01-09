from __future__ import annotations

from typing import Any, Dict, List

from abstractruntime import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler


def _make_validation_error() -> Exception:
    from pydantic import BaseModel, ValidationError

    class _TmpModel(BaseModel):
        x: int

    try:
        _TmpModel.model_validate({})
    except ValidationError as e:
        return e
    raise AssertionError("Expected ValidationError")


class _FailStructuredThenReturn:
    def __init__(self, *, content: str) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._content = content

    def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        params = kwargs.get("params") if isinstance(kwargs, dict) else None
        params = params if isinstance(params, dict) else {}
        if "response_model" in params:
            raise _make_validation_error()
        return {"content": self._content, "metadata": {}}


def test_llm_call_structured_output_fallback_disabled_still_fails() -> None:
    llm = _FailStructuredThenReturn(content='{"choice":"neutral"}')
    handler = make_llm_call_handler(llm=llm)

    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hi",
            "response_schema": {"type": "object", "properties": {"choice": {"type": "string"}}},
        },
    )

    outcome = handler(run, effect, None)
    assert outcome.status == "failed"
    assert len(llm.calls) == 1


def test_llm_call_structured_output_fallback_retries_and_parses() -> None:
    llm = _FailStructuredThenReturn(content='{"choice":"neutral"}')
    handler = make_llm_call_handler(llm=llm)

    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hi",
            "response_schema": {"type": "object", "properties": {"choice": {"type": "string"}}},
            "structured_output_fallback": True,
        },
    )

    outcome = handler(run, effect, None)
    assert outcome.status == "completed"
    assert len(llm.calls) == 2

    assert isinstance(outcome.result, dict)
    assert outcome.result.get("data") == {"choice": "neutral"}
    meta = outcome.result.get("metadata")
    assert isinstance(meta, dict)
    assert isinstance(meta.get("_structured_output_fallback"), dict)
    assert meta["_structured_output_fallback"].get("used") is True
