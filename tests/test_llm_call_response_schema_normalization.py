from __future__ import annotations

import json
from typing import Any, Dict, Optional

from abstractruntime import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler


class _CaptureLLM:
    def __init__(self) -> None:
        self.kwargs: Optional[Dict[str, Any]] = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return {"content": "ok", "metadata": {}}


def _model_field_names(model: Any) -> set[str]:
    fields = getattr(model, "model_fields", None)
    if isinstance(fields, dict):
        return set(fields.keys())
    fields = getattr(model, "__fields__", None)
    if isinstance(fields, dict):
        return set(fields.keys())
    return set()


def test_llm_call_response_schema_field_map_is_coerced_to_object_schema() -> None:
    llm = _CaptureLLM()
    handler = make_llm_call_handler(llm=llm)

    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hi",
            # Common authoring shortcut: a field map (not a JSON Schema).
            "response_schema": {"choice": "the choice you selected"},
        },
    )

    outcome = handler(run, effect, None)
    assert outcome.status == "completed"

    assert isinstance(llm.kwargs, dict)
    params = llm.kwargs.get("params")
    assert isinstance(params, dict)
    response_model = params.get("response_model")
    assert response_model is not None
    assert "choice" in _model_field_names(response_model)


def test_llm_call_response_schema_inner_wrapper_is_unwrapped() -> None:
    llm = _CaptureLLM()
    handler = make_llm_call_handler(llm=llm)

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {"choice": {"type": "string"}},
        "required": ["choice"],
    }
    inner_wrapper = {"name": "ChoiceSchema", "strict": True, "schema": schema}

    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})
    effect = Effect(type=EffectType.LLM_CALL, payload={"prompt": "hi", "response_schema": inner_wrapper})

    outcome = handler(run, effect, None)
    assert outcome.status == "completed"

    assert isinstance(llm.kwargs, dict)
    response_model = llm.kwargs.get("params", {}).get("response_model")
    assert response_model is not None
    assert _model_field_names(response_model) == {"choice"}


def test_llm_call_response_schema_outer_wrapper_is_unwrapped() -> None:
    llm = _CaptureLLM()
    handler = make_llm_call_handler(llm=llm)

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {"choice": {"type": "string"}},
        "required": ["choice"],
    }
    outer_wrapper = {"type": "json_schema", "json_schema": {"name": "ChoiceSchema", "strict": True, "schema": schema}}

    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})
    effect = Effect(type=EffectType.LLM_CALL, payload={"prompt": "hi", "response_schema": outer_wrapper})

    outcome = handler(run, effect, None)
    assert outcome.status == "completed"

    assert isinstance(llm.kwargs, dict)
    response_model = llm.kwargs.get("params", {}).get("response_model")
    assert response_model is not None
    assert _model_field_names(response_model) == {"choice"}


def test_llm_call_response_schema_missing_type_is_treated_as_object_when_properties_exist() -> None:
    llm = _CaptureLLM()
    handler = make_llm_call_handler(llm=llm)

    schema_missing_type: Dict[str, Any] = {
        "properties": {"choice": {"type": "string"}},
        "required": ["choice"],
    }

    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})
    effect = Effect(type=EffectType.LLM_CALL, payload={"prompt": "hi", "response_schema": schema_missing_type})

    outcome = handler(run, effect, None)
    assert outcome.status == "completed"

    assert isinstance(llm.kwargs, dict)
    response_model = llm.kwargs.get("params", {}).get("response_model")
    assert response_model is not None
    assert _model_field_names(response_model) == {"choice"}


def test_llm_call_response_schema_string_json_is_parsed() -> None:
    llm = _CaptureLLM()
    handler = make_llm_call_handler(llm=llm)

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})
    effect = Effect(type=EffectType.LLM_CALL, payload={"prompt": "hi", "response_schema": json.dumps(schema)})

    outcome = handler(run, effect, None)
    assert outcome.status == "completed"

    assert isinstance(llm.kwargs, dict)
    response_model = llm.kwargs.get("params", {}).get("response_model")
    assert response_model is not None
    assert _model_field_names(response_model) == {"answer"}


def test_llm_call_response_schema_non_object_schema_still_errors() -> None:
    llm = _CaptureLLM()
    handler = make_llm_call_handler(llm=llm)

    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})
    effect = Effect(type=EffectType.LLM_CALL, payload={"prompt": "hi", "response_schema": {"type": "string"}})

    outcome = handler(run, effect, None)
    assert outcome.status == "failed"
    assert isinstance(outcome.error, str)
    assert "response_schema must be a JSON schema object" in outcome.error

