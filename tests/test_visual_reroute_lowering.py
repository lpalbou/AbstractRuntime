"""Reroute (relay) nodes are visual only: chains are rewired source -> target before compiling.

Operator 2026-10-11: "it's a question (during execution and packaging) of duplicating
(output1 -> pin -> input1) into (output1 -> input1) like if the pin never existed visually".
"""

from __future__ import annotations

import pytest

from abstractruntime.core.runtime import Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler.compiler import compile_flow
from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json
from abstractruntime.visualflow_compiler.visual.reroute_lowering import lower_reroutes

X = {"id": "exec-in", "label": "", "type": "execution"}
O = {"id": "exec-out", "label": "", "type": "execution"}


def _node(nid, t, inputs=(), outputs=(), **extra):
    data = {"nodeType": t, "inputs": list(inputs), "outputs": list(outputs)}
    data.update(extra)
    return {"id": nid, "type": t, "data": data}


def _relay(nid):
    return _node(nid, "reroute", [{"id": "in", "type": "string"}], [{"id": "out", "type": "string"}])


def _set_var(nid, name):
    return _node(nid, "set_var", [X, {"id": "name", "type": "string"}, {"id": "value", "type": "any"}],
                 [O, {"id": "value", "type": "any"}], pinDefaults={"name": name, "value": "default"})


def _edge(s, sh, t, th):
    return {"source": s, "sourceHandle": sh, "target": t, "targetHandle": th}


def _flow(nodes, edges):
    return load_visualflow_json({"id": "reroutes", "name": "reroutes", "entryNode": "start", "nodes": nodes, "edges": edges})


def _run(vf):
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    wf = compile_flow(visual_to_flow(vf))
    rid = rt.start(workflow=wf, vars={})
    state = rt.tick(workflow=wf, run_id=rid, max_steps=50)
    assert state.status.value == "completed", state.error
    return state.vars


def _fan_out_flow(with_source=True):
    nodes = [
        _node("start", "on_flow_start", outputs=[O]),
        _node("text", "literal_string", outputs=[{"id": "value", "type": "string"}], literalValue="hello"),
        _relay("r1"),
        _relay("r2"),
        _set_var("a", "a"),
        _set_var("b", "b"),
        _set_var("c", "c"),
    ]
    edges = [
        _edge("start", "exec-out", "a", "exec-in"),
        _edge("a", "exec-out", "b", "exec-in"),
        _edge("b", "exec-out", "c", "exec-in"),
        _edge("r1", "out", "a", "value"),  # fan-out from one relay
        _edge("r1", "out", "b", "value"),
        _edge("r1", "out", "r2", "in"),  # a chain of relays
        _edge("r2", "out", "c", "value"),
    ]
    if with_source:
        edges.append(_edge("text", "value", "r1", "in"))
    return _flow(nodes, edges)


def test_a_relay_feeds_every_target_as_if_wired_directly():
    out = _run(_fan_out_flow())
    assert (out["a"], out["b"], out["c"]) == ("hello", "hello", "hello")


def test_lowering_removes_relays_and_rewires_each_target_to_the_source():
    lowered = lower_reroutes(_fan_out_flow())
    assert not any(n.type == "reroute" for n in lowered.nodes)
    data = sorted((e.source, e.sourceHandle, e.target, e.targetHandle) for e in lowered.edges if e.targetHandle == "value")
    assert data == [("text", "value", "a", "value"), ("text", "value", "b", "value"), ("text", "value", "c", "value")]


def test_a_relay_without_a_source_feeds_nothing_and_targets_keep_their_defaults():
    out = _run(_fan_out_flow(with_source=False))
    assert (out["a"], out["b"], out["c"]) == ("default", "default", "default")


def test_a_flow_without_relays_is_left_untouched():
    vf = _flow([_node("start", "on_flow_start", outputs=[O])], [])
    assert lower_reroutes(vf) is vf


def test_per_path_values_from_a_relay_point_at_the_real_source():
    vf = _flow(
        [_relay("r1"), _node("text", "literal_string", outputs=[{"id": "value", "type": "string"}], literalValue="x"),
         _node("t", "set_var", [X], [O], inputRouteOverrides={"value": {"p::exec-out": {"sourceNodeId": "r1", "sourceHandle": "out"}}})],
        [_edge("text", "value", "r1", "in")],
    )
    t = next(n for n in lower_reroutes(vf).nodes if n.id == "t")
    assert t.data["inputRouteOverrides"] == {"value": {"p::exec-out": {"sourceNodeId": "text", "sourceHandle": "value"}}}


@pytest.mark.parametrize(
    "edges, message",
    [
        ([_edge("start", "exec-out", "r1", "in")], "execution wires through a reroute"),
        ([_edge("r1", "out", "r2", "in"), _edge("r2", "out", "r1", "in"), _edge("r1", "out", "a", "value")], "cycle"),
        ([_edge("text", "value", "r1", "in"), _edge("text", "value", "r1", "in")], "more than one incoming"),
    ],
)
def test_malformed_relays_are_refused_loudly(edges, message):
    vf = _flow(
        [_node("start", "on_flow_start", outputs=[O]), _node("text", "literal_string", outputs=[{"id": "value", "type": "string"}]),
         _relay("r1"), _relay("r2"), _set_var("a", "a")],
        edges,
    )
    with pytest.raises(ValueError, match=message):
        lower_reroutes(vf)
