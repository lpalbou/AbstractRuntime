"""An execution wire that loops back out of a ForEach / For / While body is refused when the flow
compiles (AbstractFramework backlog 1018; operator ruling 2026-10-15). The body's frame would stay
on the control stack and the run would never end; a Loop Limit on that wire never reaches its max.
"""

from __future__ import annotations

import pytest

from abstractruntime.visualflow_compiler.compiler import compile_flow
from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

X = {"id": "exec-in", "type": "execution"}
O = {"id": "exec-out", "type": "execution"}
LOOP_OUT = [{"id": "loop", "type": "execution"}, {"id": "done", "type": "execution"}, {"id": "item", "type": "any"}, {"id": "index", "type": "number"}]
FOR_OUT = [{"id": "loop", "type": "execution"}, {"id": "done", "type": "execution"}, {"id": "i", "type": "number"}, {"id": "index", "type": "number"}]
IF_IN = [X, {"id": "condition", "type": "boolean"}]
IF_OUT = [{"id": "true", "type": "execution"}, {"id": "false", "type": "execution"}]


def _node(nid, ntype, inputs=(), outputs=(), **data):
    return {"id": nid, "type": ntype, "data": {"nodeType": ntype, "inputs": list(inputs), "outputs": list(outputs), **data}}


def _edge(s, sh, t, th, **data):
    e = {"source": s, "sourceHandle": sh, "target": t, "targetHandle": th}
    if data:
        e["data"] = data
    return e


def _compile(nodes, edges):
    vf = load_visualflow_json({"id": "t", "name": "t", "entryNode": "start", "nodes": nodes, "edges": edges})
    return compile_flow(visual_to_flow(vf))


START = _node("start", "on_flow_start", outputs=[O])
STEP = lambda nid, label=None: _node(nid, "set_var", [X, {"id": "name", "type": "string"}, {"id": "value", "type": "any"}], [O], pinDefaults={"name": nid, "value": 1}, **({"label": label} if label else {}))  # noqa: E731


def _foreach(nid, label=None, **pins):
    return _node(nid, "loop", [X, {"id": "items", "type": "array"}], LOOP_OUT, pinDefaults={"items": [1, 2], **pins}, **({"label": label} if label else {}))


def _for(nid, label=None):
    return _node(nid, "for", [X, {"id": "start", "type": "number"}, {"id": "end", "type": "number"}, {"id": "step", "type": "number"}], FOR_OUT, pinDefaults={"start": 0, "end": 2, "step": 1}, **({"label": label} if label else {}))


def _if(nid, condition=False, label=None):
    return _node(nid, "if", IF_IN, IF_OUT, pinDefaults={"condition": condition}, **({"label": label} if label else {}))


def test_a_wire_from_an_inner_body_back_to_the_outer_loop_is_refused_naming_the_wire_and_the_loop():
    nodes = [START, _for("outer", "Outer"), _for("inner", "Inner"), _if("check", label="Check"), STEP("after")]
    edges = [
        _edge("start", "exec-out", "outer", "exec-in"),
        _edge("outer", "loop", "inner", "exec-in"),
        _edge("inner", "loop", "check", "exec-in"),
        _edge("check", "true", "after", "exec-in"),
        _edge("check", "false", "outer", "exec-in"),
    ]
    with pytest.raises(ValueError) as err:
        _compile(nodes, edges)
    msg = str(err.value)
    assert msg.startswith("Wire Check.false -> Outer goes back out of Outer's loop body: the run would never end.")
    assert "to repeat the whole loop, wire the loop-back after the loop's done" in msg


def test_a_wire_out_of_a_body_to_a_node_before_the_inner_loop_is_refused_too():
    # The target sits inside the outer body but outside the inner one: the inner frame would never pop.
    nodes = [START, _for("outer"), STEP("prepare"), _foreach("inner"), _if("check")]
    edges = [
        _edge("start", "exec-out", "outer", "exec-in"),
        _edge("outer", "loop", "prepare", "exec-in"),
        _edge("prepare", "exec-out", "inner", "exec-in"),
        _edge("inner", "loop", "check", "exec-in"),
        _edge("check", "false", "prepare", "exec-in"),
    ]
    with pytest.raises(ValueError, match=r"Wire check\.false -> prepare goes back out of inner's loop body"):
        _compile(nodes, edges)


def test_the_operators_shape_is_accepted_the_loop_back_goes_through_done():
    # ForEach -> body; done -> review -> If; false back to the ForEach (its frame was popped at done).
    nodes = [START, _foreach("each"), STEP("work"), STEP("review"), _if("check", True), STEP("report")]
    edges = [
        _edge("start", "exec-out", "each", "exec-in"),
        _edge("each", "loop", "work", "exec-in"),
        _edge("each", "done", "review", "exec-in"),
        _edge("review", "exec-out", "check", "exec-in"),
        _edge("check", "true", "report", "exec-in"),
        _edge("check", "false", "each", "exec-in", loopLimit={"max": 2}),
    ]
    _compile(nodes, edges)


def test_a_repeat_inside_the_body_is_accepted():
    nodes = [START, _foreach("each"), STEP("step1"), _if("check", True)]
    edges = [
        _edge("start", "exec-out", "each", "exec-in"),
        _edge("each", "loop", "step1", "exec-in"),
        _edge("step1", "exec-out", "check", "exec-in"),
        _edge("check", "false", "step1", "exec-in", loopLimit={"max": 3}),
    ]
    _compile(nodes, edges)


def test_a_limit_on_the_offending_wire_does_not_make_it_acceptable():
    nodes = [START, _for("outer"), _if("check")]
    edges = [
        _edge("start", "exec-out", "outer", "exec-in"),
        _edge("outer", "loop", "check", "exec-in"),
        _edge("check", "false", "outer", "exec-in", loopLimit={"max": 1}),
    ]
    with pytest.raises(ValueError, match="goes back out of outer's loop body"):
        _compile(nodes, edges)
