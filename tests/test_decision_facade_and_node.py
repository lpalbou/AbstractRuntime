"""`decide`: the run facade method and the VisualFlow node, with fakes only.

A decision is an `LLM_CALL` whose output spec has `modality: "decision"`. The
fake client below returns the normalized result the AbstractCore client
produces (`content`, `decision`, `answers`, `provider`, `model`, `metadata`);
no model is ever loaded.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict

import pytest

from abstractruntime import Runtime, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.core.models import EffectType, RunState
from abstractruntime.integrations.abstractcore import AbstractCoreRunFacade, get_abstractcore_run_facade
from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow


QUESTIONS: Dict[str, Any] = {
    "action": {
        "type": "choice",
        "instructions": "What should happen next?",
        "criteria": ["reply", "escalate", "ignore"],
    },
    "urgent": "Is the message urgent?",
}

ANSWERS: Dict[str, Any] = {
    "action": {
        "type": "choice",
        "choice": "escalate",
        "confidence": 0.91,
        "probabilities": {"reply": 0.07, "escalate": 0.91, "ignore": 0.02},
    },
    "urgent": {"type": "noul", "noul": True},
}


def _decision_body(*, model: str, truncated: bool = False) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "answers": copy.deepcopy(ANSWERS),
        "usage": {"input_tokens": 212, "output_tokens": 2},
        "latency_ms": 41,
    }
    if truncated:
        body["truncated"] = True
    return body


def _decision_result(*, provider: str, model: str) -> Dict[str, Any]:
    """The normalized effect result the AbstractCore client returns for a decision."""

    body = _decision_body(model=f"{provider}/{model}")
    return {
        "content": json.dumps(ANSWERS, separators=(",", ":")),
        "decision": body,
        "answers": copy.deepcopy(ANSWERS),
        "outputs": {},
        "resources": [],
        "warnings": [],
        "errors": [],
        "provider": provider,
        "model": model,
        "runtime_provider": "lmstudio",
        "runtime_model": "chat-model",
        "usage": dict(body["usage"]),
        "metadata": {"decision": copy.deepcopy(body), "content": json.dumps(ANSWERS)},
    }


class _FakeDecisionClient:
    """Runtime LLM client stand-in: records the call, answers with the contract's result."""

    def __init__(self, *, provider: str = "mlx", model: str = "kev-4b") -> None:
        self.calls: list[Dict[str, Any]] = []
        self._provider = provider
        self._model = model

    def default_prompt_cache_identity(self):
        return "lmstudio", "chat-model"

    def generate(self, **kwargs: Any) -> Dict[str, Any]:
        # Shallow: the runtime's params carry non-copyable host objects.
        params = kwargs.get("params")
        self.calls.append({**kwargs, "params": dict(params) if isinstance(params, dict) else params})
        return _decision_result(provider=self._provider, model=self._model)


class _NoopTools:
    def execute(self, *, tool_calls):
        return {"mode": "executed", "results": []}


def _completed_parent_workflow() -> WorkflowSpec:
    def done(run, ctx):
        _ = (run, ctx)
        return StepPlan(node_id="done", complete_output={"ok": True})

    return WorkflowSpec("wf_parent", "done", {"done": done})


def _runtime(llm: Any) -> Runtime:
    return Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers=build_effect_handlers(llm=llm, tools=_NoopTools()),
    )


def _facade_with_parent(llm: Any) -> tuple[AbstractCoreRunFacade, str]:
    runtime = _runtime(llm)
    parent = _completed_parent_workflow()
    parent_run_id = runtime.start(workflow=parent, session_id="sess-decide")
    runtime.tick(workflow=parent, run_id=parent_run_id)
    return get_abstractcore_run_facade(runtime), parent_run_id


# --------------------------------------------------------------------------
# Facade
# --------------------------------------------------------------------------


def test_run_facade_decide_creates_durable_child_llm_call_with_decision_spec() -> None:
    llm = _FakeDecisionClient()
    facade, parent_run_id = _facade_with_parent(llm)

    child = facade.decide(
        parent_run_id,
        state="Customer writes: the invoice is wrong and I need it fixed today.",
        questions=QUESTIONS,
        model="mlx/kev-4b",
        truncate=True,
    )

    assert child.status == RunStatus.COMPLETED
    assert child.parent_run_id == parent_run_id
    assert child.session_id == "sess-decide"
    assert child.workflow_id == "wf_abstractcore_run_facade_decision"

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["prompt"] == "Customer writes: the invoice is wrong and I need it fixed today."
    spec = call["params"]["output"]
    assert spec["modality"] == "decision"
    assert spec["questions"] == QUESTIONS
    assert spec["truncate"] is True
    assert "provider" not in spec
    assert spec["model"] == "mlx/kev-4b"
    # A decision produces no bytes: no artifact store was given and none was needed.

    result = child.output["result"]
    assert result["answers"] == ANSWERS
    assert result["decision"]["answers"] == ANSWERS
    assert result["decision"]["model"] == "mlx/kev-4b"
    assert result["provider"] == "mlx"
    assert result["model"] == "kev-4b"
    assert json.loads(result["content"]) == ANSWERS


def test_run_facade_decide_passes_a_provider_slash_model_handle_whole() -> None:
    """AbstractCore splits `provider/model` handles (decision.pool.split_handle); the
    runtime keeps no provider vocabulary of its own."""
    llm = _FakeDecisionClient(provider="huggingface", model="kev-9b")
    facade, parent_run_id = _facade_with_parent(llm)

    child = facade.decide(parent_run_id, state="state", questions=QUESTIONS, model="hf/kev-9b")

    assert child.status == RunStatus.COMPLETED
    spec = llm.calls[0]["params"]["output"]
    assert "provider" not in spec
    assert spec["model"] == "hf/kev-9b"
    assert "truncate" not in spec


def test_run_facade_decide_keeps_a_namespaced_model_handle_whole() -> None:
    llm = _FakeDecisionClient()
    facade, parent_run_id = _facade_with_parent(llm)

    facade.decide(parent_run_id, state="state", questions=QUESTIONS, model="mlx-community/kev-4b-8bit")

    spec = llm.calls[0]["params"]["output"]
    assert "provider" not in spec
    assert spec["model"] == "mlx-community/kev-4b-8bit"


def test_run_facade_decide_without_model_leaves_the_route_to_core() -> None:
    llm = _FakeDecisionClient()
    facade, parent_run_id = _facade_with_parent(llm)

    facade.decide(parent_run_id, state="state", questions=QUESTIONS)

    spec = llm.calls[0]["params"]["output"]
    assert spec["modality"] == "decision"
    assert "provider" not in spec
    assert "model" not in spec


def test_run_facade_decide_validates_state_and_questions_before_runtime_start() -> None:
    class _StubRuntime:
        def start(self, **kwargs):
            raise AssertionError("invalid decide input should be rejected before runtime start")

        def tick(self, **kwargs):
            raise AssertionError("invalid decide input should be rejected before runtime tick")

        def get_state(self, run_id: str):
            raise AssertionError(f"unexpected get_state({run_id})")

    facade = AbstractCoreRunFacade(_StubRuntime())

    with pytest.raises(ValueError, match="state"):
        facade.decide("run-1", state="", questions=QUESTIONS)
    with pytest.raises(ValueError, match="state"):
        facade.decide("run-1", state="   ", questions=QUESTIONS)
    with pytest.raises(ValueError, match="state"):
        facade.decide("run-1", state=None, questions=QUESTIONS)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="questions"):
        facade.decide("run-1", state="state", questions={})
    with pytest.raises(ValueError, match="questions"):
        facade.decide("run-1", state="state", questions=["yes/no?"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="questions"):
        facade.decide("run-1", state="state", questions="is it urgent?")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# VisualFlow node
# --------------------------------------------------------------------------


def _plan_for_node(node_type: str, effect_config: dict, input_data: dict | None = None) -> object:
    spec = compile_visualflow(
        {
            "id": f"vf_{node_type}",
            "name": f"vf_{node_type}",
            "entryNode": "node",
            "nodes": [
                {
                    "id": "node",
                    "type": node_type,
                    "data": {"nodeType": node_type, "effectConfig": effect_config},
                }
            ],
            "edges": [],
        }
    )
    run = RunState(
        run_id="run",
        workflow_id=str(spec.workflow_id),
        status=RunStatus.RUNNING,
        current_node="node",
        vars={"_temp": {}},
    )
    if isinstance(input_data, dict):
        run.vars["_last_output"] = dict(input_data)
    return spec.nodes["node"](run, input_data)


def test_decide_node_compiles_to_llm_call_decision_selector() -> None:
    plan = _plan_for_node(
        "decide",
        {"state": "The build failed twice.", "questions": QUESTIONS, "model": "mlx/kev-4b", "truncate": True},
    )

    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = dict(plan.effect.payload or {})
    assert payload["prompt"] == "The build failed twice."
    assert "provider" not in payload
    assert "model" not in payload
    output = dict(payload.get("output") or {})
    assert output == {
        "modality": "decision",
        "questions": QUESTIONS,
        "truncate": True,
        "model": "mlx/kev-4b",
    }


def test_decide_node_input_pins_override_config_and_bare_model_stays_bare() -> None:
    plan = _plan_for_node(
        "decide",
        {"state": "config state", "questions": {"x": "config?"}, "truncate": True},
        {"state": "pin state", "questions": QUESTIONS, "model": "kev-9b", "truncate": False},
    )

    assert plan.effect is not None
    payload = dict(plan.effect.payload or {})
    assert payload["prompt"] == "pin state"
    output = dict(payload.get("output") or {})
    assert output["questions"] == QUESTIONS
    assert output["truncate"] is False
    assert output["model"] == "kev-9b"
    assert "provider" not in output


def test_decide_node_without_model_or_truncate_sends_only_the_decision_spec() -> None:
    plan = _plan_for_node("decide", {"state": "s", "questions": QUESTIONS})

    assert plan.effect is not None
    output = dict((plan.effect.payload or {}).get("output") or {})
    assert output == {"modality": "decision", "questions": QUESTIONS}


def test_decide_node_provider_pin_is_the_decision_provider_never_the_text_runtime() -> None:
    """The node's `provider` / `model` pins pick the DECISION model (the editor scopes the
    pickers to decision providers); the text runtime never reaches the spec, and without pins
    the spec names no route (the output.decision default applies)."""
    plan = _plan_for_node("decide", {"state": "s", "questions": QUESTIONS}, {"provider": "mlx", "model": "jaredpalmer/kev-0.8b"})
    output = dict(plan.effect.payload["output"])
    assert output["provider"] == "mlx" and output["model"] == "jaredpalmer/kev-0.8b"
    assert "provider" not in plan.effect.payload

    plan = _plan_for_node("decide", {"state": "s", "questions": QUESTIONS, "provider": "lmstudio", "model_text": "chat-model"})
    output = dict(plan.effect.payload["output"])
    assert "provider" not in output and "model" not in output


def test_decide_node_rejects_missing_state_or_questions() -> None:
    with pytest.raises(ValueError, match="state"):
        _plan_for_node("decide", {"questions": QUESTIONS})
    with pytest.raises(ValueError, match="questions"):
        _plan_for_node("decide", {"state": "s"})
    with pytest.raises(ValueError, match="questions"):
        _plan_for_node("decide", {"state": "s", "questions": "is it urgent?"})


def _decide_flow() -> Dict[str, Any]:
    return {
        "id": "vf-decide-e2e",
        "name": "vf-decide-e2e",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "data": {
                    "nodeType": "on_flow_start",
                    "inputs": [],
                    "outputs": [
                        {"id": "exec-out", "label": "", "type": "execution"},
                        {"id": "state", "label": "state", "type": "string"},
                        {"id": "questions", "label": "questions", "type": "object"},
                    ],
                },
            },
            {
                "id": "decide",
                "type": "decide",
                "data": {
                    "nodeType": "decide",
                    "effectConfig": {"model": "mlx/kev-4b"},
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "state", "label": "state", "type": "string"},
                        {"id": "questions", "label": "questions", "type": "object"},
                        {"id": "model", "label": "model", "type": "string"},
                        {"id": "truncate", "label": "truncate", "type": "boolean"},
                    ],
                    "outputs": [
                        {"id": "exec-out", "label": "", "type": "execution"},
                        {"id": "answers", "label": "answers", "type": "object"},
                        {"id": "decision", "label": "decision", "type": "object"},
                        {"id": "model", "label": "model", "type": "string"},
                        {"id": "content", "label": "content", "type": "string"},
                    ],
                },
            },
            {
                "id": "end",
                "type": "on_flow_end",
                "data": {
                    "nodeType": "on_flow_end",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "answers", "label": "answers", "type": "object"},
                        {"id": "decision", "label": "decision", "type": "object"},
                        {"id": "model", "label": "model", "type": "string"},
                        {"id": "content", "label": "content", "type": "string"},
                    ],
                    "outputs": [],
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "decide", "targetHandle": "exec-in"},
            {"id": "e2", "source": "start", "sourceHandle": "state", "target": "decide", "targetHandle": "state"},
            {"id": "e3", "source": "start", "sourceHandle": "questions", "target": "decide", "targetHandle": "questions"},
            {"id": "e4", "source": "decide", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"},
            {"id": "e5", "source": "decide", "sourceHandle": "answers", "target": "end", "targetHandle": "answers"},
            {"id": "e6", "source": "decide", "sourceHandle": "decision", "target": "end", "targetHandle": "decision"},
            {"id": "e7", "source": "decide", "sourceHandle": "model", "target": "end", "targetHandle": "model"},
            {"id": "e8", "source": "decide", "sourceHandle": "content", "target": "end", "targetHandle": "content"},
        ],
    }


def test_decide_node_executes_end_to_end_and_answers_reach_the_output_pins() -> None:
    llm = _FakeDecisionClient()
    runtime = _runtime(llm)
    spec = compile_visualflow(_decide_flow())

    run_id = runtime.start(
        workflow=spec,
        vars={"state": "Ticket: the invoice total is wrong.", "questions": copy.deepcopy(QUESTIONS)},
    )
    state = runtime.tick(workflow=spec, run_id=run_id, max_steps=20)

    assert state.status == RunStatus.COMPLETED, state.error
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["prompt"] == "Ticket: the invoice total is wrong."
    assert call["params"]["output"]["modality"] == "decision"
    assert call["params"]["output"]["questions"] == QUESTIONS
    assert "provider" not in call["params"]["output"]
    assert call["params"]["output"]["model"] == "mlx/kev-4b"

    effect_result = state.vars["_temp"]["effects"]["decide"]
    assert effect_result["answers"] == ANSWERS

    out = state.output
    assert isinstance(out, dict), out
    assert out["answers"] == ANSWERS
    assert out["decision"]["answers"] == ANSWERS
    assert out["decision"]["model"] == "mlx/kev-4b"
    assert out["model"] == "kev-4b"
    assert json.loads(out["content"]) == ANSWERS


def test_decide_result_sync_fills_the_declared_output_pins() -> None:
    from abstractruntime.visualflow_compiler.compiler import _sync_effect_results_to_node_outputs
    from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
    from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

    vf = load_visualflow_json(
        {
            "id": "vf-decide-sync",
            "name": "vf-decide-sync",
            "entryNode": "decide",
            "nodes": [
                {
                    "id": "decide",
                    "type": "decide",
                    "data": {"nodeType": "decide", "effectConfig": {"state": "s", "questions": QUESTIONS}},
                }
            ],
            "edges": [],
        }
    )
    flow = visual_to_flow(vf)
    run = RunState.new(workflow_id=flow.flow_id, entry_node="decide")
    generated = _decision_result(provider="typesafe", model="jev")
    run.vars["_temp"] = {"effects": {"decide": generated}}

    _sync_effect_results_to_node_outputs(run, flow)

    out = flow._node_outputs["decide"]  # type: ignore[attr-defined]
    assert out["answers"] == ANSWERS
    assert out["decision"] == generated["decision"]
    assert out["model"] == "jev"
    assert out["content"] == generated["content"]
    assert out["success"] is True
    assert out["meta"]["schema"] == "abstractflow.decide.v1.meta"
    assert out["meta"]["modality"] == "decision"
    assert out["meta"]["provider"] == "typesafe"
    assert out["meta"]["usage"] == {"input_tokens": 212, "output_tokens": 2}
    assert "_pending_effect" not in out


def test_decide_result_sync_without_answers_is_not_a_success() -> None:
    from abstractruntime.visualflow_compiler.compiler import _sync_effect_results_to_node_outputs
    from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
    from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

    vf = load_visualflow_json(
        {
            "id": "vf-decide-sync-empty",
            "name": "vf-decide-sync-empty",
            "entryNode": "decide",
            "nodes": [
                {
                    "id": "decide",
                    "type": "decide",
                    "data": {"nodeType": "decide", "effectConfig": {"state": "s", "questions": QUESTIONS}},
                }
            ],
            "edges": [],
        }
    )
    flow = visual_to_flow(vf)
    run = RunState.new(workflow_id=flow.flow_id, entry_node="decide")
    run.vars["_temp"] = {"effects": {"decide": {"content": "", "errors": [{"message": "no decision model"}]}}}

    _sync_effect_results_to_node_outputs(run, flow)

    out = flow._node_outputs["decide"]  # type: ignore[attr-defined]
    assert out["answers"] == {}
    assert out["decision"] is None
    assert out["model"] is None
    assert out["success"] is False


# --------------------------------------------------------------------------
# Single-question form (`question` / `question_type` / `options` / `levels`)
# --------------------------------------------------------------------------

SINGLE_ANSWERS: Dict[str, Dict[str, Any]] = {
    "choice": {
        "type": "choice",
        "choice": "shipping",
        "confidence": 0.8,
        "probabilities": {"returns": 0.15, "shipping": 0.8, "billing": 0.05},
    },
    "noul": {"type": "noul", "noul": 0.72},
    "score": {
        "type": "score",
        "score": 1.4,
        "confidence": 0.6,
        "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
        "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
    },
}

SINGLE_CONFIG: Dict[str, Dict[str, Any]] = {
    "choice": {"question": "Which team?", "question_type": "choice", "options": ["returns", "shipping", "billing"]},
    "noul": {"question": "Is the customer angry?", "question_type": "noul"},
    "score": {"question": "How angry?", "question_type": "score", "levels": ["Calm", "Frustrated", "Very angry"]},
}

TYPED_PINS: Dict[str, list[str]] = {
    "choice": ["choice", "confidence", "probabilities"],
    "noul": ["noul", "verdict"],
    "score": ["score", "level", "confidence", "probabilities"],
}


class _FakeSingleDecisionClient(_FakeDecisionClient):
    """Answers with one `answer` of the given type (the single-question form)."""

    def __init__(self, answers: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._answers = answers

    def generate(self, **kwargs: Any) -> Dict[str, Any]:
        result = super().generate(**kwargs)
        answers = copy.deepcopy(self._answers)
        result["answers"] = answers
        result["decision"]["answers"] = copy.deepcopy(answers)
        result["content"] = json.dumps(answers, separators=(",", ":"))
        return result


def _single_question_payload(effect_config: dict, input_data: dict | None = None) -> Dict[str, Any]:
    plan = _plan_for_node("decide", effect_config, input_data)
    assert plan.effect is not None and plan.effect.type == EffectType.LLM_CALL
    return dict(plan.effect.payload or {})


def test_decide_single_choice_with_option_names_compiles_to_an_answer_question() -> None:
    payload = _single_question_payload({"state": "The parcel never arrived.", **SINGLE_CONFIG["choice"]})

    assert payload["prompt"] == "The parcel never arrived."
    assert payload["output"] == {
        "modality": "decision",
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"returns": None, "shipping": None, "billing": None},
            }
        },
    }


def test_decide_single_choice_with_described_options_keeps_the_descriptions() -> None:
    options = {"returns": "refunds and exchanges", "shipping": "delivery problems"}
    payload = _single_question_payload(
        {"state": "s", "question": "Which team?", "question_type": "choice", "options": options, "model": "mlx/kev-4b"}
    )

    assert payload["output"]["questions"] == {
        "answer": {"type": "choice", "instructions": "Which team?", "criteria": options}
    }
    assert payload["output"]["model"] == "mlx/kev-4b"


def test_decide_single_question_type_defaults_to_choice() -> None:
    payload = _single_question_payload({"state": "s", "question": "Which team?", "options": ["a", "b"]})

    assert payload["output"]["questions"]["answer"]["type"] == "choice"


def test_decide_single_noul_compiles_without_criteria() -> None:
    payload = _single_question_payload({"state": "s", **SINGLE_CONFIG["noul"], "truncate": True})

    assert payload["output"] == {
        "modality": "decision",
        "questions": {"answer": {"type": "noul", "instructions": "Is the customer angry?"}},
        "truncate": True,
    }


def test_decide_single_score_compiles_levels_as_ordered_criteria() -> None:
    payload = _single_question_payload({"state": "s", **SINGLE_CONFIG["score"]})

    assert payload["output"]["questions"] == {
        "answer": {"type": "score", "instructions": "How angry?", "criteria": ["Calm", "Frustrated", "Very angry"]}
    }


def test_decide_single_form_reads_question_pins_from_input_over_config() -> None:
    payload = _single_question_payload(
        {"state": "s", "question": "config question", "question_type": "score", "levels": ["lo", "hi"]},
        {"question": "pin question", "question_type": "choice", "options": ["x"]},
    )

    assert payload["output"]["questions"] == {
        "answer": {"type": "choice", "instructions": "pin question", "criteria": {"x": None}}
    }


def test_decide_single_form_validation_errors_name_the_pin() -> None:
    with pytest.raises(ValueError, match="`question`"):
        _plan_for_node("decide", {"state": "s", "question_type": "noul"})
    with pytest.raises(ValueError, match="`options`"):
        _plan_for_node("decide", {"state": "s", "question": "Which?", "question_type": "choice"})
    with pytest.raises(ValueError, match="`options`"):
        _plan_for_node("decide", {"state": "s", "question": "Which?", "question_type": "choice", "options": []})
    with pytest.raises(ValueError, match="`options`"):  # only blank rows: nothing to choose from
        _plan_for_node("decide", {"state": "s", "question": "Which?", "options": ["", " "]})
    with pytest.raises(ValueError, match="`levels`"):
        _plan_for_node("decide", {"state": "s", "question": "How?", "question_type": "score", "levels": ["only"]})
    with pytest.raises(ValueError, match="`levels`"):
        _plan_for_node("decide", {"state": "s", "question": "How?", "question_type": "score"})
    with pytest.raises(ValueError, match="`question_type`"):
        _plan_for_node("decide", {"state": "s", "question": "How?", "question_type": "rank", "options": ["a"]})
    with pytest.raises(ValueError, match="`state`"):
        _plan_for_node("decide", {"question": "How?", "question_type": "noul"})


def test_decide_wired_questions_object_takes_precedence_over_the_single_form() -> None:
    payload = _single_question_payload({"state": "s", **SINGLE_CONFIG["choice"]}, {"questions": QUESTIONS})

    assert payload["output"]["questions"] == QUESTIONS


def _single_decide_flow(question_type: str) -> Dict[str, Any]:
    """Editor-shaped flow: the question pins live in `pinDefaults`, the typed outputs are wired to the end."""
    config = SINGLE_CONFIG[question_type]
    out_pins = ["answer", *TYPED_PINS[question_type], "answers", "model", "success"]
    input_pins = [{"id": "exec-in", "label": "", "type": "execution"}, {"id": "state", "label": "state", "type": "string"}] + [
        {"id": key, "label": key, "type": "string"} for key in config
    ]
    return {
        "id": f"vf-decide-{question_type}",
        "name": f"vf-decide-{question_type}",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "data": {
                    "nodeType": "on_flow_start",
                    "inputs": [],
                    "outputs": [
                        {"id": "exec-out", "label": "", "type": "execution"},
                        {"id": "state", "label": "state", "type": "string"},
                    ],
                },
            },
            {
                "id": "decide",
                "type": "decide",
                "data": {
                    "nodeType": "decide",
                    "effectConfig": {"model": "mlx/kev-4b"},
                    "pinDefaults": dict(config),
                    "inputs": input_pins,
                    "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]
                    + [{"id": pin, "label": pin, "type": "any"} for pin in out_pins],
                },
            },
            {
                "id": "end",
                "type": "on_flow_end",
                "data": {
                    "nodeType": "on_flow_end",
                    "inputs": [{"id": "exec-in", "label": "", "type": "execution"}]
                    + [{"id": pin, "label": pin, "type": "any"} for pin in out_pins],
                    "outputs": [],
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "decide", "targetHandle": "exec-in"},
            {"id": "e2", "source": "start", "sourceHandle": "state", "target": "decide", "targetHandle": "state"},
            {"id": "e3", "source": "decide", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"},
        ]
        + [
            {"id": f"o-{pin}", "source": "decide", "sourceHandle": pin, "target": "end", "targetHandle": pin}
            for pin in out_pins
        ],
    }


def _run_single(question_type: str, answer: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    llm = _FakeSingleDecisionClient({"answer": answer})
    runtime = _runtime(llm)
    spec = compile_visualflow(_single_decide_flow(question_type))
    run_id = runtime.start(workflow=spec, vars={"state": "Customer: where is my parcel?!"})
    state = runtime.tick(workflow=spec, run_id=run_id, max_steps=20)
    assert state.status == RunStatus.COMPLETED, state.error
    assert len(llm.calls) == 1
    assert llm.calls[0]["prompt"] == "Customer: where is my parcel?!"
    assert isinstance(state.output, dict), state.output
    return llm.calls[0]["params"]["output"], state.output


def test_decide_single_choice_runs_end_to_end_into_the_typed_pins() -> None:
    answer = SINGLE_ANSWERS["choice"]
    spec, out = _run_single("choice", answer)

    assert spec["questions"]["answer"]["criteria"] == {"returns": None, "shipping": None, "billing": None}
    assert out["answer"] == answer
    assert out["choice"] == "shipping"
    assert out["confidence"] == 0.8
    assert out["probabilities"] == answer["probabilities"]
    assert out["answers"] == {"answer": answer}
    assert out["model"] == "kev-4b"
    assert out["success"] is True


def test_decide_single_noul_runs_end_to_end_into_noul_and_verdict() -> None:
    answer = SINGLE_ANSWERS["noul"]
    spec, out = _run_single("noul", answer)

    assert spec["questions"]["answer"] == {"type": "noul", "instructions": "Is the customer angry?"}
    assert out["answer"] == answer
    assert out["noul"] == 0.72
    assert out["verdict"] is True
    assert out["success"] is True


def test_decide_single_score_runs_end_to_end_into_score_and_level() -> None:
    answer = SINGLE_ANSWERS["score"]
    spec, out = _run_single("score", answer)

    assert spec["questions"]["answer"]["criteria"] == ["Calm", "Frustrated", "Very angry"]
    assert out["answer"] == answer
    assert out["score"] == 1.4
    assert out["level"] == "Frustrated"
    assert out["confidence"] == 0.6
    assert out["probabilities"] == answer["probabilities"]
    assert out["success"] is True


def _sync_single(answers: Dict[str, Any] | None) -> Dict[str, Any]:
    from abstractruntime.visualflow_compiler.compiler import _sync_effect_results_to_node_outputs
    from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
    from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

    vf = load_visualflow_json(
        {
            "id": "vf-decide-sync-single",
            "name": "vf-decide-sync-single",
            "entryNode": "decide",
            "nodes": [{"id": "decide", "type": "decide", "data": {"nodeType": "decide", "effectConfig": {"state": "s", **SINGLE_CONFIG["noul"]}}}],
            "edges": [],
        }
    )
    flow = visual_to_flow(vf)
    run = RunState.new(workflow_id=flow.flow_id, entry_node="decide")
    generated = _decision_result(provider="mlx", model="kev-4b")
    if answers is None:
        generated = {"content": "", "errors": [{"message": "no decision model"}]}
    else:
        generated["answers"] = copy.deepcopy(answers)
        generated["decision"]["answers"] = copy.deepcopy(answers)
    run.vars["_temp"] = {"effects": {"decide": generated}}
    _sync_effect_results_to_node_outputs(run, flow)
    return flow._node_outputs["decide"]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("noul", "verdict"),
    [(0.0, False), (0.49, False), (0.5, True), (0.51, True), (1.0, True), (True, True), (False, False)],
)
def test_decide_verdict_is_noul_at_or_above_one_half(noul: Any, verdict: bool) -> None:
    out = _sync_single({"answer": {"type": "noul", "noul": noul}})

    assert out["noul"] == noul
    assert out["verdict"] is verdict
    for pin in ("choice", "confidence", "probabilities", "score", "level"):
        assert out[pin] is None


@pytest.mark.parametrize(("score", "level"), [(0.0, "Calm"), (0.49, "Calm"), (1.4, "Frustrated"), (1.6, "Very angry"), (2.0, "Very angry")])
def test_decide_level_is_the_legend_label_nearest_the_score(score: float, level: str) -> None:
    answer = {**SINGLE_ANSWERS["score"], "score": score}
    out = _sync_single({"answer": answer})

    assert out["score"] == score
    assert out["level"] == level
    assert out["noul"] is None
    assert out["verdict"] is None
    assert out["choice"] is None


def test_decide_sync_for_the_advanced_form_fills_answers_and_answer_from_the_first_answer() -> None:
    out = _sync_single(ANSWERS)

    assert out["answers"] == ANSWERS
    assert out["answer"] == ANSWERS["action"]
    assert out["choice"] == "escalate"
    assert out["confidence"] == 0.91
    assert out["probabilities"] == ANSWERS["action"]["probabilities"]
    assert out["noul"] is None
    assert out["verdict"] is None
    assert out["score"] is None
    assert out["level"] is None
    assert out["success"] is True


def test_decide_sync_without_answers_leaves_answer_and_typed_pins_none() -> None:
    out = _sync_single(None)

    assert out["answer"] is None
    for pin in ("choice", "confidence", "probabilities", "noul", "verdict", "score", "level"):
        assert out[pin] is None
    assert out["success"] is False


# --------------------------------------------------------------------------
# Editor parity: cleaned options, object-list options, casing, level rounding
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options, expected",
    [
        (["returns", "", "shipping", "returns"], {"returns": None, "shipping": None}),  # blank row being typed, repeat
        ([{"name": "a", "description": "A"}, {"name": "b"}], {"a": "A", "b": None}),   # object-list form
        ({" a ": "A", "b": ""}, {"a": "A", "b": None}),                               # {name: description}
    ],
)
def test_decide_options_are_cleaned_like_the_editor(options, expected) -> None:
    plan = _plan_for_node("decide", {"state": "s", "question": "q?", "question_type": "Choice", "options": options})
    assert plan.effect.payload["output"]["questions"]["answer"]["criteria"] == expected
    assert plan.effect.payload["output"]["questions"]["answer"]["type"] == "choice"


def test_decide_levels_drop_blank_rows_and_keep_order() -> None:
    plan = _plan_for_node("decide", {"state": "s", "question": "q?", "question_type": "score", "levels": ["Calm", "", " Angry "]})
    assert plan.effect.payload["output"]["questions"]["answer"]["criteria"] == ["Calm", "Angry"]


@pytest.mark.parametrize("score, level", [(0.5, "Frustrated"), (1.5, "Very angry"), (1.49, "Frustrated"), (0.49, "Calm")])
def test_level_is_the_nearest_label_with_half_rounding_up(score, level) -> None:
    from abstractruntime.visualflow_compiler.compiler import _decide_level_for

    assert _decide_level_for(score, {"0": "Calm", "1": "Frustrated", "2": "Very angry"}) == level


def test_decide_provider_pin_travels_beside_the_model() -> None:
    # the provider PIN (wired or a pin default) selects the decision provider ...
    plan = _plan_for_node("decide", {"state": "s", "question": "q?", "question_type": "noul"},
                          {"provider": "MLX", "model": "jaredpalmer/kev-0.8b"})
    output = plan.effect.payload["output"]
    assert output["provider"] == "mlx" and output["model"] == "jaredpalmer/kev-0.8b"
    # ... while effectConfig.provider (the text runtime of llm_call-style nodes) never does
    plan = _plan_for_node("decide", {"state": "s", "question": "q?", "question_type": "noul", "provider": "lmstudio", "model": "typesafe/jev-latest"})
    assert "provider" not in plan.effect.payload["output"] and plan.effect.payload["output"]["model"] == "typesafe/jev-latest"
