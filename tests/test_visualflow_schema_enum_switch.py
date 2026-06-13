from __future__ import annotations

from types import SimpleNamespace


def _ctx():
    return SimpleNamespace(now_iso=lambda: "2026-01-16T00:00:00Z")


def _model_field_names(model):
    fields = getattr(model, "model_fields", None)
    if isinstance(fields, dict):
        return set(fields.keys())
    fields = getattr(model, "__fields__", None)
    if isinstance(fields, dict):
        return set(fields.keys())
    return set()


def test_visualflow_llm_call_pin_default_response_schema_preserves_enum() -> None:
    from abstractruntime.core.models import RunState, RunStatus
    from abstractruntime.visualflow_compiler import compile_visualflow

    response_schema = {
        "type": "object",
        "properties": {
            "choice": {"type": "string", "enum": ["approve", "reject"]},
        },
        "required": ["choice"],
    }

    raw = {
        "id": "schema-enum-flow",
        "name": "schema-enum-flow",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "on_flow_start",
                    "outputs": [{"id": "exec-out", "label": "", "type": "execution"}],
                },
            },
            {
                "id": "call",
                "type": "llm_call",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "llm_call",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "prompt", "label": "prompt", "type": "string"},
                        {"id": "resp_schema", "label": "resp_schema", "type": "object"},
                    ],
                    "pinDefaults": {"prompt": "Classify the request.", "resp_schema": response_schema},
                    "effectConfig": {"provider": "lmstudio", "model": "unit-test-model", "temperature": 0.0},
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "call", "targetHandle": "exec-in"},
        ],
    }

    spec = compile_visualflow(raw)
    run = RunState(run_id="r1", workflow_id=spec.workflow_id, status=RunStatus.RUNNING, current_node="start", vars={})

    plan1 = spec.get_node("start")(run, _ctx())
    assert plan1.next_node == "call"
    run.current_node = plan1.next_node

    plan2 = spec.get_node("call")(run, _ctx())
    assert plan2.effect is not None
    payload = dict(plan2.effect.payload or {})
    assert payload["response_schema"] == response_schema
    assert payload["response_schema"]["properties"]["choice"]["enum"] == ["approve", "reject"]


def test_visualflow_llm_call_auto_defaults_still_preserve_response_schema() -> None:
    from abstractruntime.core.models import RunState, RunStatus
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
    from abstractruntime.visualflow_compiler import compile_visualflow

    class CaptureLLM:
        kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return {"content": '{"choice":"summarize","confidence":0.95}', "metadata": {}}

    response_schema = {
        "type": "object",
        "properties": {
            "choice": {"type": "string", "description": "The selected task route."},
            "confidence": {"type": "number", "description": "Confidence in range 0.0 to 1.0."},
        },
        "required": ["choice", "confidence"],
    }

    raw = {
        "id": "schema-auto-default-flow",
        "name": "schema-auto-default-flow",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "on_flow_start",
                    "outputs": [{"id": "exec-out", "label": "", "type": "execution"}],
                },
            },
            {
                "id": "call",
                "type": "llm_call",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "llm_call",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "prompt", "label": "prompt", "type": "string"},
                        {"id": "provider", "label": "provider", "type": "provider_text"},
                        {"id": "model", "label": "model", "type": "model"},
                        {"id": "resp_schema", "label": "resp_schema", "type": "object"},
                    ],
                    "pinDefaults": {"prompt": "Classify the request.", "resp_schema": response_schema},
                    # Blank routing means Gateway/Core defaults. The response schema must still constrain the call.
                    "effectConfig": {"provider": "", "model": "", "temperature": 0.7},
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "call", "targetHandle": "exec-in"},
        ],
    }

    spec = compile_visualflow(raw)
    run = RunState(run_id="r1", workflow_id=spec.workflow_id, status=RunStatus.RUNNING, current_node="start", vars={})

    plan1 = spec.get_node("start")(run, _ctx())
    assert plan1.next_node == "call"
    run.current_node = plan1.next_node

    plan2 = spec.get_node("call")(run, _ctx())
    assert plan2.effect is not None
    payload = dict(plan2.effect.payload or {})
    assert "provider" not in payload
    assert "model" not in payload
    assert payload["response_schema"] == response_schema
    assert payload["response_schema_name"] == "LLM_StructuredOutput"

    llm = CaptureLLM()
    outcome = make_llm_call_handler(llm=llm)(run, plan2.effect, None)
    assert outcome.status == "completed"
    assert isinstance(llm.kwargs, dict)
    params = llm.kwargs.get("params")
    assert isinstance(params, dict)
    response_model = params.get("response_model")
    assert response_model is not None
    assert _model_field_names(response_model) == {"choice", "confidence"}
    model_schema = response_model.model_json_schema()
    assert model_schema["properties"]["choice"]["description"] == "The selected task route."
    assert model_schema["properties"]["confidence"]["description"] == "Confidence in range 0.0 to 1.0."


def test_visualflow_switch_routes_string_enum_values_to_case_branches() -> None:
    from abstractruntime import Runtime
    from abstractruntime.core.models import RunStatus
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
    from abstractruntime.visualflow_compiler import compile_visualflow

    raw = {
        "id": "switch-enum-flow",
        "name": "switch-enum-flow",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "on_flow_start",
                    "outputs": [{"id": "exec-out", "label": "", "type": "execution"}],
                },
            },
            {
                "id": "switch",
                "type": "switch",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "switch",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "value", "label": "value", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "case:approve", "label": "approve", "type": "execution"},
                        {"id": "case:reject", "label": "reject", "type": "execution"},
                        {"id": "default", "label": "default", "type": "execution"},
                    ],
                    "pinDefaults": {"value": "reject"},
                    "switchConfig": {
                        "cases": [
                            {"id": "approve", "value": "approve"},
                            {"id": "reject", "value": "reject"},
                        ]
                    },
                },
            },
            {
                "id": "end",
                "type": "on_flow_end",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "on_flow_end",
                    "inputs": [{"id": "exec-in", "label": "", "type": "execution"}],
                    "outputs": [],
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "switch", "targetHandle": "exec-in"},
            {"id": "e2", "source": "switch", "sourceHandle": "case:reject", "target": "end", "targetHandle": "exec-in"},
        ],
    }

    spec = compile_visualflow(raw)
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    run_id = runtime.start(workflow=spec, vars={})
    state = runtime.tick(workflow=spec, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    assert state.vars["_last_output"]["branch"] == "case:reject"
    assert state.vars["_last_output"]["matched"] == "reject"


def test_visualflow_structured_data_pin_feeds_answer_user_and_switch() -> None:
    from abstractruntime import EffectType, Runtime
    from abstractruntime.core.models import RunStatus
    from abstractruntime.core.runtime import EffectOutcome
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
    from abstractruntime.visualflow_compiler import compile_visualflow

    response_schema = {
        "type": "object",
        "properties": {
            "output": {"type": "string", "enum": ["summarize", "extract"]},
            "confidence": {"type": "number"},
        },
        "required": ["output", "confidence"],
    }
    raw = {
        "id": "structured-answer-switch-flow",
        "name": "structured-answer-switch-flow",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "on_flow_start",
                    "outputs": [
                        {"id": "exec-out", "label": "", "type": "execution"},
                        {"id": "query", "label": "query", "type": "string"},
                    ],
                },
            },
            {
                "id": "call",
                "type": "llm_call",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "llm_call",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "prompt", "label": "prompt", "type": "string"},
                        {"id": "resp_schema", "label": "resp_schema", "type": "object"},
                    ],
                    "outputs": [
                        {"id": "exec-out", "label": "", "type": "execution"},
                        {"id": "response", "label": "response", "type": "string"},
                        {"id": "data", "label": "data", "type": "object"},
                    ],
                    "pinDefaults": {"resp_schema": response_schema},
                    "effectConfig": {"provider": "lmstudio", "model": "unit-test-model", "temperature": 0.0},
                },
            },
            {
                "id": "break",
                "type": "break_object",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "break_object",
                    "inputs": [{"id": "object", "label": "object", "type": "object"}],
                    "outputs": [
                        {"id": "output", "label": "output", "type": "string"},
                        {"id": "confidence", "label": "confidence", "type": "number"},
                    ],
                    "breakConfig": {"selectedPaths": ["output", "confidence"]},
                },
            },
            {
                "id": "answer-before-switch",
                "type": "answer_user",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "answer_user",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "message", "label": "message", "type": "string"},
                        {"id": "level", "label": "level", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "exec-out", "label": "", "type": "execution"},
                        {"id": "message", "label": "message", "type": "string"},
                    ],
                    "pinDefaults": {"message": "fallback", "level": "message"},
                },
            },
            {
                "id": "switch",
                "type": "switch",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "switch",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "value", "label": "value", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "case:summarize", "label": "summarize", "type": "execution"},
                        {"id": "default", "label": "default", "type": "execution"},
                    ],
                    "switchConfig": {"cases": [{"id": "summarize", "value": "summarize"}]},
                },
            },
            {
                "id": "summary-answer",
                "type": "answer_user",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "answer_user",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "message", "label": "message", "type": "string"},
                    ],
                    "outputs": [{"id": "exec-out", "label": "", "type": "execution"}],
                    "pinDefaults": {"message": "Summarize pathway"},
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "call", "targetHandle": "exec-in"},
            {"id": "e2", "source": "start", "sourceHandle": "query", "target": "call", "targetHandle": "prompt"},
            {"id": "e3", "source": "call", "sourceHandle": "data", "target": "break", "targetHandle": "object"},
            {"id": "e4", "source": "call", "sourceHandle": "exec-out", "target": "answer-before-switch", "targetHandle": "exec-in"},
            {"id": "e5", "source": "break", "sourceHandle": "output", "target": "answer-before-switch", "targetHandle": "message"},
            {"id": "e6", "source": "answer-before-switch", "sourceHandle": "exec-out", "target": "switch", "targetHandle": "exec-in"},
            {"id": "e7", "source": "break", "sourceHandle": "output", "target": "switch", "targetHandle": "value"},
            {"id": "e8", "source": "switch", "sourceHandle": "case:summarize", "target": "summary-answer", "targetHandle": "exec-in"},
        ],
    }

    def _llm_stub(run, effect, default_next_node):
        del run, default_next_node
        assert (effect.payload or {}).get("response_schema") == response_schema
        return EffectOutcome.completed(
            {"content": None, "data": {"output": "summarize", "confidence": 1}, "tool_calls": None}
        )

    spec = compile_visualflow(raw)
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: _llm_stub},
    )

    run_id = runtime.start(workflow=spec, vars={"query": "who are you?"})
    state = runtime.tick(workflow=spec, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    temp = state.vars["_temp"]
    assert temp["effects"]["call"]["data"] == {"output": "summarize", "confidence": 1}
    assert temp["effects"]["answer-before-switch"]["message"] == "summarize"
    assert temp["effects"]["summary-answer"]["message"] == "Summarize pathway"
    assert state.vars["_last_output"]["branch"] == "case:summarize"
    assert state.vars["_last_output"]["matched"] == "summarize"
