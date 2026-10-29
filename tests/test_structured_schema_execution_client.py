"""Schema data crosses the client boundary; provider classes stay at execution."""
from __future__ import annotations

import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from abstractruntime import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient

SCHEMA = {
    "type": "object",
    "properties": {
        "nested": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        "items": {"type": "array", "items": {"type": "integer"}},
        "mode": {"type": "string", "enum": ["chat", "tool"]},
        "maybe": {"type": ["string", "null"]},
        "flag": {"type": "boolean", "default": False},
    },
    "required": ["nested", "items", "mode"],
}


class Provider:
    def __init__(self, answer: Any):
        self.answer = answer
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        assert "response_schema" not in kwargs and "response_schema_name" not in kwargs
        value = self.answer(kwargs) if callable(self.answer) else self.answer
        return kwargs["response_model"].model_validate(value)


def local_client(provider):
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "fake"
    client._model = "fake-model"
    client._artifact_store = None
    client._generate_lock = None
    client._capability_defaults = {}
    client._llm = provider
    client._on_token = None
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None
    client._prompt_cache_state_lock = threading.Lock()
    client._prompt_cache_state = {}
    return client


class DataBoundary:
    def __init__(self, client):
        self.client = client
        self.params = []

    def generate(self, **kwargs):
        params = kwargs["params"]
        assert "response_model" not in params
        # Fail if any future generated class sneaks back into this call.
        self.params.append(json.loads(json.dumps(params)))
        pickle.dumps(params)
        return self.client.generate(**kwargs)


def effect_call(boundary, schema=SCHEMA):
    return make_llm_call_handler(llm=boundary)(
        RunState.new(workflow_id="schema", entry_node="route_call", vars={}),
        Effect(type=EffectType.LLM_CALL, payload={
            "prompt": "answer", "response_schema": schema,
            "response_schema_name": "LLM_StructuredOutput", "params": {"stream": True},
        }),
        None,
    )


def test_effect_to_execution_client_preserves_original_nested_schema_and_stream_behavior():
    provider = Provider({"nested": {"n": "42"}, "items": ["1", 2], "mode": "chat", "maybe": None})
    boundary = DataBoundary(local_client(provider))
    outcome = effect_call(boundary)
    assert outcome.status == "completed", outcome.error
    assert outcome.result["data"] == {
        "nested": {"n": 42}, "items": [1, 2], "mode": "chat", "maybe": None, "flag": False,
    }
    assert boundary.params[0]["response_schema"] == SCHEMA
    assert boundary.params[0]["response_schema_name"] == "LLM_StructuredOutput"
    assert provider.calls[0]["stream"] is False


@pytest.mark.parametrize("answer", [
    {"nested": {"n": "not-int"}, "items": [1], "mode": "chat"},
    {"nested": {"n": 1}, "items": [1], "mode": "invalid"},
    {"items": [1], "mode": "chat"},
])
def test_invalid_output_fails_without_weaker_nested_validation(answer):
    provider = Provider(answer)
    outcome = effect_call(DataBoundary(local_client(provider)))
    assert outcome.status == "failed"
    assert "validation error" in outcome.error
    assert len(provider.calls) == 1


def test_same_name_concurrent_schemas_do_not_share_a_generated_class():
    provider = Provider(lambda kwargs: {"a": "13"} if kwargs["prompt"].endswith("alpha") else {"b": "hello"})
    client = local_client(provider)
    def run(prompt, field, kind):
        return client.generate(prompt=prompt, params={
            "response_schema": {"type": "object", "properties": {field: {"type": kind}}, "required": [field]},
            "response_schema_name": "SameName",
        })
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run, "alpha", "a", "integer")
        second = executor.submit(run, "beta", "b", "string")
    assert first.result()["data"] == {"a": 13}
    assert second.result()["data"] == {"b": "hello"}
    assert provider.calls[0]["response_model"] is not provider.calls[1]["response_model"]


class PositiveModel(BaseModel):
    n: int

    @field_validator("n")
    @classmethod
    def positive(cls, value):
        if value <= 0:
            raise ValueError("positive required")
        return value


def test_explicit_custom_response_model_keeps_its_validator():
    provider = Provider({"n": -1})
    with pytest.raises(ValidationError, match="positive required"):
        local_client(provider).generate(prompt="answer", params={"response_model": PositiveModel})
    assert provider.calls[0]["response_model"] is PositiveModel


@pytest.mark.parametrize("other", [{"response_model": PositiveModel}, {"response_format": {"type": "json_object"}}])
def test_conflicting_schema_inputs_fail_before_provider_call(other):
    provider = Provider({})
    with pytest.raises(ValueError, match="cannot be combined"):
        local_client(provider).generate(prompt="answer", params={"response_schema": SCHEMA, **other})
    assert not provider.calls


def test_matching_source_format_is_consumed_before_schema_execution():
    provider = Provider({"nested": {"n": 1}, "items": [2], "mode": "chat"})
    boundary = DataBoundary(local_client(provider))
    outcome = make_llm_call_handler(llm=boundary)(
        RunState.new(workflow_id="schema", entry_node="call", vars={}),
        Effect(type=EffectType.LLM_CALL, payload={
            "prompt": "answer", "response_schema": SCHEMA,
            "params": {"response_format": {"type": "json_schema", "json_schema": {"name": "Reply", "schema": SCHEMA}}},
        }), None)
    assert outcome.status == "completed", outcome.error
    assert "response_format" not in boundary.params[0]


def test_raw_openai_response_format_remains_at_core_boundary():
    seen = []
    class RawProvider:
        def generate(self, **kwargs):
            seen.append(kwargs)
            return PositiveModel(n=1)
    response_format = {"type": "json_schema", "json_schema": {
        "name": "Reply", "schema": {"type": "object", "properties": {"n": {"type": "integer"}}}}}
    outcome = make_llm_call_handler(llm=local_client(RawProvider()))(
        RunState.new(workflow_id="schema", entry_node="call", vars={}),
        Effect(type=EffectType.LLM_CALL, payload={"prompt": "answer", "params": {"response_format": response_format}}), None)
    assert outcome.status == "completed", outcome.error
    assert seen[0]["response_format"] == response_format
    assert "response_model" not in seen[0]


def test_effect_trace_uses_actual_step_context_and_does_not_invent_one():
    from abstractruntime.core.step_context import StepContext, effect_step_context
    from abstractruntime.integrations.abstractcore.effect_handlers import _trace_context
    run = RunState.new(workflow_id="schema", entry_node="call", vars={})
    assert "step_id" not in _trace_context(run)
    with effect_step_context(StepContext(run_id=run.run_id, step_id="attempt-step", seq=7,
                                        node_id="call", attempt=2, effect_type="llm_call")):
        context = _trace_context(run)
    assert context["step_id"] == "attempt-step" and context["attempt"] == "2"
    assert "step_id" not in _trace_context(run)
