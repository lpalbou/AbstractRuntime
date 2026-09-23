"""MTP controls cross run trees and every LLM emitter without owning policy."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from abstractruntime import (
    Effect, EffectType, InMemoryLedgerStore, InMemoryRunStore, RunState,
    RunStatus, Runtime, StepPlan, WorkflowRegistry, WorkflowSpec,
)
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
from abstractruntime.visualflow_compiler.compiler import compile_visualflow
from abstractruntime.visualflow_compiler.adapters.effect_adapter import create_llm_call_handler

D2 = {"mode": "native_mtp", "num_draft_tokens": 2}
D4 = {"mode": "native_mtp", "num_draft_tokens": 4}
_CTX = SimpleNamespace(now_iso=lambda: "2026-09-20T00:00:00Z")


def _child(parent_ns, child_ns=None):
    child = WorkflowSpec(workflow_id="child", entry_node="end", nodes={"end": lambda r, c: StepPlan(node_id="end", complete_output={})})
    parent = WorkflowSpec(workflow_id="parent", entry_node="spawn", nodes={
        "spawn": lambda r, c: StepPlan(node_id="spawn", effect=Effect(
            type=EffectType.START_SUBWORKFLOW,
            payload={"workflow_id": "child", "vars": {"_runtime": deepcopy(child_ns or {})}, "async": True, "wait": True},
            result_key="child",
        )),
    })
    registry = WorkflowRegistry()
    registry.register(child)
    registry.register(parent)
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=registry)
    run_id = runtime.start(workflow=parent, vars={"_runtime": deepcopy(parent_ns)})
    state = runtime.tick(workflow=parent, run_id=run_id, max_steps=1)
    child_id = state.waiting.wait_key.split(":", 1)[1]
    return runtime.get_state(child_id).vars["_runtime"], runtime.get_state(run_id).vars["_runtime"]


@pytest.mark.parametrize("parent,child,expected", [(D2, None, D2), (D2, False, False), (False, D4, D4), (False, None, False), (D2, {}, {}), (None, None, None)])
def test_subworkflow_policy_cascade_and_two_hops(parent, child, expected):
    result, original = _child({"speculation": deepcopy(parent)}, {"speculation": deepcopy(child)})
    assert result.get("speculation") == expected
    grandchild, _ = _child(result)
    assert grandchild.get("speculation") == expected
    if isinstance(result.get("speculation"), dict):
        result["speculation"]["num_draft_tokens"] = 9
        assert original.get("speculation") == parent


class _LLM:
    def __init__(self, content="ok"):
        self.content, self.calls = content, []

    def generate(self, **kwargs):
        self.calls.append(deepcopy(kwargs.get("params") or {}))
        return {"content": self.content, "finish_reason": "stop", "metadata": {}}


@pytest.mark.parametrize("run_value,override,expected", [(D2, None, D2), (D2, False, False), (False, D4, D4), (False, None, False), (D2, {}, {}), (None, None, None)])
@pytest.mark.parametrize("repair", [False, True])
def test_every_effect_attempt_carries_policy(run_value, override, expected, repair):
    llm = _LLM("not json" if repair else "ok")
    run = RunState(run_id="r", workflow_id="wf", status=RunStatus.RUNNING, current_node="n", vars={
        "_runtime": {"speculation": deepcopy(run_value)}, "_limits": {"max_tokens": 65536},
    })
    params = {"speculation": deepcopy(override)}
    payload = {"prompt": "hello", "params": params}
    if repair:
        payload.update(response_schema={"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}, structured_output_fallback=True)
    make_llm_call_handler(llm=llm, artifact_store=None)(run, Effect(type=EffectType.LLM_CALL, payload=payload), None)
    assert len(llm.calls) >= (2 if repair else 1)
    for call in llm.calls:
        assert call.get("speculation") == expected
        if expected is None:
            assert "speculation" not in call
    assert params == {"speculation": override}
    assert run.vars["_runtime"]["speculation"] == run_value


def _visual(node_type, config, *, pin=None):
    return compile_visualflow({
        "id": "spec-test", "entryNode": "start", "nodes": [
            {"id": "start", "type": "on_flow_start", "data": {"outputs": [
                {"id": "exec-out", "type": "execution"}, {"id": "speculation", "type": "any"},
            ]}},
            {"id": "call", "type": node_type, "data": {"agentConfig" if node_type == "agent" else "effectConfig": {
                "provider": "mlx", "model": "test", **config,
            }}},
        ], "edges": [
            {"source": "start", "sourceHandle": "exec-out", "target": "call", "targetHandle": "exec-in"},
            {"source": "start", "sourceHandle": "speculation", "target": "call", "targetHandle": "speculation"},
        ],
    })


@pytest.mark.parametrize("node_type", ["agent", "llm_call"])
@pytest.mark.parametrize("config,pin,expected", [(D2, False, False), (False, D4, D4), (D2, None, D2), (None, None, None), (D2, {}, {})])
def test_visual_node_pin_over_config_policy(node_type, config, pin, expected):
    flow = _visual(node_type, {"speculation": deepcopy(config)})
    run = RunState.new(workflow_id=flow.workflow_id, entry_node="start", vars={
        "speculation": deepcopy(pin), "_runtime": {},
    })
    flow.nodes["start"](run, _CTX)
    run.current_node = "call"
    plan = flow.nodes["call"](run, _CTX)
    assert plan.effect is not None
    controls = plan.effect.payload["vars"]["_runtime"] if node_type == "agent" else plan.effect.payload["params"]
    assert controls.get("speculation") == expected
    if expected is None:
        assert "speculation" not in controls


@pytest.mark.parametrize("policy", [False, D2, D4])
def test_visual_agent_inherits_run_and_structured_postpass_keeps_node_override(policy):
    flow = _visual("agent", {"speculation": deepcopy(policy), "outputSchema": {"enabled": True, "jsonSchema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}}})
    run = RunState.new(workflow_id=flow.workflow_id, entry_node="call", vars={"_runtime": {"speculation": deepcopy(D2)}})
    handler = flow.nodes["call"]
    first = handler(run, _CTX)
    assert first.effect.payload["vars"]["_runtime"]["speculation"] == policy
    # Drive the same compiler's completion branch, not just a helper.
    bucket = run.vars["_temp"]["agent"]["call"]
    bucket["sub"] = {"sub_run_id": "child", "output": {"answer": "done", "iterations": 1}, "node_traces": {}}
    second = handler(run, _CTX)
    assert second.effect is not None and second.effect.type == EffectType.LLM_CALL
    assert second.effect.payload["params"]["speculation"] == policy


@pytest.mark.parametrize("policy", [False, D2, None])
def test_legacy_llm_emitter_transports_policy(policy):
    handler = create_llm_call_handler(node_id="n", next_node=None, speculation=deepcopy(policy))
    run = RunState.new(workflow_id="wf", entry_node="n", vars={"prompt": "hello"})
    params = handler(run, None).effect.payload["params"]
    assert params.get("speculation") == policy
    if policy is None:
        assert "speculation" not in params


def test_visual_agent_without_node_override_inherits_parent():
    flow = _visual("agent", {})
    run = RunState.new(workflow_id=flow.workflow_id, entry_node="call", vars={"_runtime": {"speculation": deepcopy(D4)}})
    controls = flow.nodes["call"](run, _CTX).effect.payload["vars"]["_runtime"]
    assert controls["speculation"] == D4
    controls["speculation"]["num_draft_tokens"] = 1
    assert run.vars["_runtime"]["speculation"] == D4


@pytest.mark.parametrize("thinking", [False, "high"])
def test_visual_agent_reasoning_pin_reaches_compiler_too(thinking):
    raw = {
        "id": "reasoning-pin", "entryNode": "start", "nodes": [
            {"id": "start", "type": "on_flow_start", "data": {"outputs": [
                {"id": "exec-out", "type": "execution"}, {"id": "thinking", "type": "any"},
            ]}},
            {"id": "agent", "type": "agent", "data": {"agentConfig": {"provider": "mlx", "model": "test", "thinking": "low"}}},
        ], "edges": [
            {"source": "start", "sourceHandle": "exec-out", "target": "agent", "targetHandle": "exec-in"},
            {"source": "start", "sourceHandle": "thinking", "target": "agent", "targetHandle": "thinking"},
        ],
    }
    flow = compile_visualflow(raw)
    run = RunState.new(workflow_id=flow.workflow_id, entry_node="start", vars={"thinking": thinking})
    flow.nodes["start"](run, _CTX)
    run.current_node = "agent"
    result = flow.nodes["agent"](run, _CTX)
    assert result.effect.payload["vars"]["_runtime"]["thinking"] == thinking
