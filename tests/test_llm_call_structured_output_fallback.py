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


class _FailStructuredThenRepair:
    def __init__(self, *, invalid_content: str, repaired_content: str) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._invalid_content = invalid_content
        self._repaired_content = repaired_content

    def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        params = kwargs.get("params") if isinstance(kwargs, dict) else None
        params = params if isinstance(params, dict) else {}
        if "response_model" in params:
            raise _make_validation_error()
        if len(self.calls) == 2:
            return {"content": self._invalid_content, "metadata": {}}
        return {"content": self._repaired_content, "metadata": {}}


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


def test_llm_call_structured_output_fallback_repairs_invalid_json_shape() -> None:
    llm = _FailStructuredThenRepair(
        invalid_content='{"mode":"image","prompt":"draw a cat"}',
        repaired_content='{"mode":"image","assistant_message":"Generating the image now.","media_prompt":"draw a cat"}',
    )
    handler = make_llm_call_handler(llm=llm)

    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "draw a cat",
            "response_schema": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string"},
                    "assistant_message": {"type": "string"},
                    "media_prompt": {"type": "string"},
                },
                "required": ["mode", "assistant_message", "media_prompt"],
            },
            "structured_output_fallback": True,
        },
    )

    outcome = handler(run, effect, None)
    assert outcome.status == "completed"
    assert len(llm.calls) == 3

    assert isinstance(outcome.result, dict)
    assert outcome.result.get("data") == {
        "mode": "image",
        "assistant_message": "Generating the image now.",
        "media_prompt": "draw a cat",
    }

    repair_prompt = str(llm.calls[2].get("prompt") or "")
    assert "Validation error" in repair_prompt
    assert '"assistant_message"' in repair_prompt
    assert '"media_prompt"' in repair_prompt

    meta = outcome.result.get("metadata")
    assert isinstance(meta, dict)
    assert meta.get("_structured_output_repair") == {"used": True, "error": ""}
