"""The visual llm_call node forwards provider/model INDEPENDENTLY (flow's
adversary P0, commons c884; the 2026-06-10 preflight ruling: "provider and
model are independently optional in the LLM effect handler; connected pins
resolve at runtime").

The killed bug: the old both-or-neither branch built the pending effect
with NEITHER key when only one resolved, so a model-only override (the
model-pool-through-loop-item pattern flow's authoring stack teaches)
SILENTLY executed every call on the gateway default model — the graph
looked right, readiness passed, and nothing downstream could detect it.
A workflow that runs and lies.
"""

from __future__ import annotations

from types import SimpleNamespace

from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.visualflow_compiler import compile_visualflow


def _ctx():
    return SimpleNamespace(now_iso=lambda: "2026-01-16T00:00:00Z")


def _flow(effect_config: dict) -> dict:
    return {
        "id": "vf",
        "name": "vf",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "position": {"x": 0, "y": 0},
                "data": {"outputs": [
                    {"id": "exec-out", "label": "", "type": "execution"},
                    {"id": "prompt", "label": "prompt", "type": "string"},
                ]},
            },
            {
                "id": "call",
                "type": "llm_call",
                "position": {"x": 0, "y": 0},
                "data": {"effectConfig": effect_config},
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out",
             "target": "call", "targetHandle": "exec-in"},
        ],
    }


def _payload_for(effect_config: dict) -> dict:
    spec = compile_visualflow(_flow(effect_config))
    run = RunState(run_id="r1", workflow_id=spec.workflow_id,
                   status=RunStatus.RUNNING, current_node="start", vars={})
    run.vars.update({"prompt": "Hello"})
    plan1 = spec.get_node("start")(run, _ctx())
    run.current_node = plan1.next_node
    plan2 = spec.get_node("call")(run, _ctx())
    assert plan2.effect is not None and plan2.effect.type.value == "llm_call"
    return dict(plan2.effect.payload or {})


def test_model_only_override_reaches_the_effect() -> None:
    """The model-pool pattern: model pinned, provider blank — the MODEL must
    ride the effect (the provider resolves from run/gateway defaults)."""
    payload = _payload_for({"model": "pool-model-b", "temperature": 0.0})
    assert payload.get("model") == "pool-model-b"
    assert "provider" not in payload  # absent, never an empty string
    assert "error" not in payload


def test_provider_only_override_reaches_the_effect() -> None:
    payload = _payload_for({"provider": "lmstudio", "temperature": 0.0})
    assert payload.get("provider") == "lmstudio"
    assert "model" not in payload


def test_both_present_and_both_blank_unchanged() -> None:
    both = _payload_for({"provider": "lmstudio", "model": "unit-test-model"})
    assert both.get("provider") == "lmstudio" and both.get("model") == "unit-test-model"
    neither = _payload_for({"temperature": 0.0})
    assert "provider" not in neither and "model" not in neither
