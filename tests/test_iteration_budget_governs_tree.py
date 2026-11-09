"""One iteration budget governs the whole run tree (operator 2026-11-08: "it doesn't look like the
agent is following the max iterations I request ... By default, set it to 50 everywhere").

- An On Flow Start `max_iterations` pin DEFAULT (the shipped basic-agent authors 20) never stands
  in for the run's `_limits.max_iterations`: that default used to ride the data edge into the
  Agent node and overwrite the requested budget in the react child's vars.
- START_SUBWORKFLOW carries the parent's `_limits.max_iterations` into the child (an explicit
  child value wins).
- No request = DEFAULT_MAX_ITERATIONS (50), exported by name; no reader carries a copy.

ONE order everywhere (review 2026-11-09 finding 2): the person's request > the host's stored
default (both a DECLARED `_limits.max_iterations`) > the workflow's own value (a start-pin or
Agent-node pin default, an Agent node's config) > DEFAULT_MAX_ITERATIONS. The reviewer's probes
(rt_probe/probe.py, probe2.py, probe3.py) are the Agent-node and start-pin cases below."""

from __future__ import annotations

from typing import Any, Dict

import pytest

import abstractruntime
from abstractruntime import (
    DEFAULT_MAX_ITERATIONS,
    Effect,
    EffectType,
    InMemoryLedgerStore,
    InMemoryRunStore,
    RunState,
    RunStatus,
    Runtime,
    StepPlan,
    WorkflowRegistry,
    WorkflowSpec,
)
from abstractruntime.core.config import RuntimeConfig
from abstractruntime.core.vars import MAX_ITERATIONS_DECLARED_KEY, declared_max_iterations, ensure_limits
from abstractruntime.visualflow_compiler.compiler import compile_visualflow

pytestmark = pytest.mark.basic


def _flow() -> Dict[str, Any]:
    """On Flow Start (max_iterations pin, authored default 20) -> Agent node, as basic-agent."""
    return {
        "id": "budget-flow",
        "name": "budget",
        "entryNode": "start",
        "nodes": [
            {"id": "start", "type": "on_flow_start", "data": {
                "nodeType": "on_flow_start",
                "outputs": [
                    {"id": "exec-out", "type": "execution"},
                    {"id": "prompt", "type": "string"},
                    {"id": "max_iterations", "type": "number"},
                ],
                "pinDefaults": {"max_iterations": 20},
            }},
            {"id": "agent", "type": "agent", "data": {
                "nodeType": "agent",
                "agentConfig": {"provider": "stub", "model": "stub"},
                "inputs": [{"id": "exec-in", "type": "execution"}, {"id": "prompt", "type": "string"},
                           {"id": "max_iterations", "type": "number"}],
            }},
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "agent", "targetHandle": "exec-in"},
            {"id": "e2", "source": "start", "sourceHandle": "prompt", "target": "agent", "targetHandle": "prompt"},
            {"id": "e3", "source": "start", "sourceHandle": "max_iterations", "target": "agent",
             "targetHandle": "max_iterations"},
        ],
    }


def _child_budget(vars: Dict[str, Any], flow: Any = None) -> tuple[Any, Any]:
    spec = compile_visualflow(flow if flow is not None else _flow())
    seen: Dict[str, Any] = {}

    def react_stub(run: RunState, ctx) -> StepPlan:  # stands in for the react child: records what it got
        seen["limits"] = (run.vars.get("_limits") or {}).get("max_iterations")
        seen["scratchpad"] = (run.vars.get("scratchpad") or {}).get("max_iterations")
        return StepPlan(node_id="s", complete_output={"answer": "ok"})

    reg = WorkflowRegistry()
    reg.register(spec)
    reg.register(WorkflowSpec(workflow_id="visual_react_agent_budget-flow_agent", entry_node="s", nodes={"s": react_stub}))
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=reg)
    rid = rt.start(workflow=spec, vars={"prompt": "go", **vars})
    for _ in range(10):
        st = rt.tick(workflow=spec, run_id=rid, max_steps=1)
        if st.status == RunStatus.WAITING and str(st.waiting.wait_key or "").startswith("subworkflow:"):
            child_id = str(st.waiting.wait_key).split(":", 1)[1]
            child = rt.run_store.load(child_id)
            rt.tick(workflow=reg.get(child.workflow_id), run_id=child_id, max_steps=1)
        if seen or st.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert seen, ("the Agent node never started its react child", st.status, st.error)
    return seen["limits"], seen["scratchpad"]


@pytest.mark.parametrize("requested", [50, 10])
def test_a_requested_limits_budget_beats_the_start_pin_default(requested: int) -> None:
    assert _child_budget({"_limits": {"max_iterations": requested}}) == (requested, requested)


def test_the_flat_pin_still_counts_as_the_request() -> None:
    assert _child_budget({"max_iterations": 7}) == (7, 7)


def _start_pin(value: Any) -> Dict[str, Any]:
    """The flow with the start pin's own default set to `value` (None = no default authored)."""
    f = _flow()
    defaults = f["nodes"][0]["data"]["pinDefaults"]
    if value is None:
        defaults.pop("max_iterations")
    else:
        defaults["max_iterations"] = value
    return f


def _agent_own(*, pin_default: Any = None, config: Any = None) -> Dict[str, Any]:
    """The Agent node with its own budget and NO edge from the start pin (as the Flow editor saves it)."""
    f = _flow()
    agent = f["nodes"][1]["data"]
    f["edges"] = [e for e in f["edges"] if e["targetHandle"] != "max_iterations"]
    if pin_default is not None:
        agent["pinDefaults"] = {"max_iterations": pin_default}
    if config is not None:
        agent["inputs"] = [p for p in agent["inputs"] if p["id"] != "max_iterations"]
        agent["agentConfig"]["max_iterations"] = config
    return f


def test_nothing_anywhere_gets_the_framework_default() -> None:
    assert _child_budget({}, _start_pin(None)) == (DEFAULT_MAX_ITERATIONS, DEFAULT_MAX_ITERATIONS)


@pytest.mark.parametrize("pin", [5, 20])
def test_a_deliberate_start_pin_value_applies_when_nothing_is_requested(pin: int) -> None:
    """rt_probe/probe.py: a start pin of 5 used to give 50 (the seeded default always won)."""
    assert _child_budget({}, _start_pin(pin)) == (pin, pin)


@pytest.mark.parametrize("own", [{"pin_default": 20}, {"config": 20}])
def test_the_request_beats_the_agent_nodes_own_value(own: Dict[str, Any]) -> None:
    """rt_probe/probe3.py: an Agent node's saved pin default 20 beat a requested 100."""
    assert _child_budget({"_limits": {"max_iterations": 100}}, _agent_own(**own)) == (100, 100)


@pytest.mark.parametrize("own", [{"pin_default": 20}, {"config": 20}])
def test_the_agent_nodes_own_value_beats_the_seeded_default(own: Dict[str, Any]) -> None:
    assert _child_budget({}, _agent_own(**own)) == (20, 20)


def test_start_stamps_whether_the_budget_was_declared() -> None:
    reg = WorkflowRegistry()
    spec = WorkflowSpec(workflow_id="w", entry_node="n", nodes={"n": lambda run, ctx: StepPlan(node_id="n", complete_output={})})
    reg.register(spec)
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=reg)
    asked = rt.run_store.load(rt.start(workflow=spec, vars={"_limits": {"max_iterations": 9}})).vars["_limits"]
    seeded = rt.run_store.load(rt.start(workflow=spec, vars={})).vars["_limits"]
    assert asked[MAX_ITERATIONS_DECLARED_KEY] is True and declared_max_iterations(asked) == 9
    assert seeded[MAX_ITERATIONS_DECLARED_KEY] is False and declared_max_iterations(seeded) is None
    assert seeded["max_iterations"] == DEFAULT_MAX_ITERATIONS


def test_the_default_is_50_and_exported_by_name() -> None:
    assert DEFAULT_MAX_ITERATIONS == 50
    assert "DEFAULT_MAX_ITERATIONS" in abstractruntime.__all__
    assert RuntimeConfig().max_iterations == 50
    assert ensure_limits({})["max_iterations"] == 50


def _spawn(parent_limits: Dict[str, Any], child_vars: Dict[str, Any]) -> Dict[str, Any]:
    def child_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="c", complete_output={"ok": True})

    def parent_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="p", effect=Effect(
            type=EffectType.START_SUBWORKFLOW,
            payload={"workflow_id": "child_wf", "vars": dict(child_vars), "async": True, "wait": True},
            result_key="r"), next_node="p")

    child = WorkflowSpec(workflow_id="child_wf", entry_node="c", nodes={"c": child_node})
    parent = WorkflowSpec(workflow_id="parent_wf", entry_node="p", nodes={"p": parent_node})
    reg = WorkflowRegistry()
    reg.register(child)
    reg.register(parent)
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=reg)
    rid = rt.start(workflow=parent, vars={"_limits": dict(parent_limits)})
    st = rt.tick(workflow=parent, run_id=rid, max_steps=1)
    child_id = str(st.waiting.wait_key).split(":", 1)[1]
    return rt.run_store.load(child_id).vars["_limits"]


def test_a_child_flow_inherits_the_parents_budget() -> None:
    assert _spawn({"max_iterations": 37}, {})["max_iterations"] == 37


def test_an_explicit_child_budget_wins_over_the_parents() -> None:
    """A delegated sub-agent's narrower `_limits` budget is kept (abstractagent's min(explicit, parent))."""
    assert _spawn({"max_iterations": 37}, {"_limits": {"max_iterations": 5}})["max_iterations"] == 5


def test_a_declared_parent_budget_beats_a_subflow_pin_value() -> None:
    """A flat `max_iterations` a subflow pin passes is a workflow value: the request governs."""
    child = _spawn({"max_iterations": 37}, {"max_iterations": 5})
    assert child["max_iterations"] == 37 and child[MAX_ITERATIONS_DECLARED_KEY] is True


def test_a_seeded_parent_budget_rides_as_seeded() -> None:
    """The parent's seeded default reaches the child still marked seeded: a child workflow's own
    value beats it there."""
    child = _spawn({"max_iterations": DEFAULT_MAX_ITERATIONS, MAX_ITERATIONS_DECLARED_KEY: False}, {})
    assert child["max_iterations"] == DEFAULT_MAX_ITERATIONS and child[MAX_ITERATIONS_DECLARED_KEY] is False
