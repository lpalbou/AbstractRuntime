"""A limit on a wire (`edge.data.loopLimit`), operator 2026-10-14: "the loop can only happen X times..
and then it should get out of the loop and resume where it was" — no Loop Limit node to place, no
exit pin to wire. Lowered into a loop_limit node on the wire before compile
(visual/loop_limit_lowering.py).
"""

from __future__ import annotations

import pytest

from abstractruntime.core.runtime import Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler.compiler import compile_flow
from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
from abstractruntime.visualflow_compiler.visual.loop_limit_lowering import lower_loop_limits
from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

X = {"id": "exec-in", "type": "execution"}
O = {"id": "exec-out", "type": "execution"}
ADD_PINS = [X, {"id": "name", "type": "string"}, {"id": "item", "type": "any"}, {"id": "items", "type": "array"}]


def _node(nid, ntype, inputs=(), outputs=(), **data):
    return {"id": nid, "type": ntype, "data": {"nodeType": ntype, "inputs": list(inputs), "outputs": list(outputs), **data}}


def _edge(s, sh, t, th, **data):
    e = {"source": s, "sourceHandle": sh, "target": t, "targetHandle": th}
    if data:
        e["data"] = data
    return e


def _run(nodes, edges, start_vars=None):
    vf = load_visualflow_json({"id": "t", "name": "t", "entryNode": "start", "nodes": nodes, "edges": edges})
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    wf = compile_flow(visual_to_flow(vf))
    rid = rt.start(workflow=wf, vars=dict(start_vars or {}))
    return rt.tick(workflow=wf, run_id=rid, max_steps=300)


START = _node("start", "on_flow_start", outputs=[O])
IF_PINS = ([X, {"id": "condition", "type": "boolean"}], [{"id": "true", "type": "execution"}, {"id": "false", "type": "execution"}])


def _review_loop(max_passes, condition=False):
    """ForEach-less shape of the operator's flow: head -> review -> If; false loops back (limited), true -> report."""
    head = _node("head", "add_to_list", ADD_PINS, [O], pinDefaults={"name": "passes", "item": "x"})
    iff = _node("if", "if", IF_PINS[0], IF_PINS[1], pinDefaults={"condition": condition})
    report = _node("report", "set_var", [X, {"id": "name", "type": "string"}, {"id": "value", "type": "any"}], [O], pinDefaults={"name": "report", "value": "done"})
    nodes = [START, head, iff, report]
    edges = [
        _edge("start", "exec-out", "head", "exec-in"),
        _edge("head", "exec-out", "if", "exec-in"),
        _edge("if", "true", "report", "exec-in"),
        _edge("if", "false", "head", "exec-in", loopLimit={"max": max_passes}),
    ]
    return nodes, edges


def test_the_loop_happens_at_most_max_times_then_resumes_on_the_other_branch():
    state = _run(*_review_loop(3))
    assert state.status.value == "completed", state.error
    assert len(state.vars["passes"]) == 4  # the first pass, then 3 repeats
    assert state.vars["report"] == "done"  # got out of the loop: continued where If/Else true goes


def test_a_loop_that_ends_by_itself_is_not_affected():
    state = _run(*_review_loop(3, condition=True))
    assert len(state.vars["passes"]) == 1 and state.vars["report"] == "done"


def test_the_per_path_wire_keyed_on_the_limited_wire_keeps_working():
    # head is entered from start and from If/Else false; on the way back, `item` comes from a per-path wire.
    lit = _node("lit", "literal_string", outputs=[{"id": "value", "type": "string"}], literalValue="again")
    head = _node("head", "add_to_list", ADD_PINS, [O], pinDefaults={"name": "passes", "item": "first"},
                 inputRouteOverrides={"item": {"if::false": {"sourceNodeId": "lit", "sourceHandle": "value"}}})
    iff = _node("if", "if", IF_PINS[0], IF_PINS[1], pinDefaults={"condition": False})
    nodes = [START, lit, head, iff]
    edges = [
        _edge("start", "exec-out", "head", "exec-in"),
        _edge("head", "exec-out", "if", "exec-in"),
        _edge("if", "false", "head", "exec-in", loopLimit={"max": 2}),
    ]
    state = _run(nodes, edges)
    assert state.status.value == "completed", state.error
    assert state.vars["passes"] == ["first", "again", "again"]


def test_without_an_other_branch_the_path_ends_at_the_limit():
    nodes, edges = _review_loop(2)
    edges = [e for e in edges if e["sourceHandle"] != "true"]  # If/Else true unwired
    state = _run(nodes, edges)
    assert state.status.value == "completed"
    assert len(state.vars["passes"]) == 3 and "report" not in state.vars


def test_a_limit_on_a_data_wire_is_refused():
    nodes, edges = _review_loop(2)
    edges.append(_edge("head", "list", "if", "condition", loopLimit={"max": 1}))
    nodes[1]["data"]["outputs"].append({"id": "list", "type": "array"})
    with pytest.raises(ValueError, match="not an execution wire"):
        lower_loop_limits(load_visualflow_json({"id": "t", "name": "t", "nodes": nodes, "edges": edges}))


def test_a_wire_without_a_limit_is_left_alone():
    nodes, edges = _review_loop(2)
    del edges[3]["data"]
    vf = load_visualflow_json({"id": "t", "name": "t", "nodes": nodes, "edges": edges})
    assert lower_loop_limits(vf) is vf


def test_the_editors_entry_lists_are_kept_in_step_with_the_rewritten_wires():
    # Exactly what the editor saves for the operator's shape: the ForEach-like head is entered from
    # start and from If/Else false (entryRoutes names both); the report is entered from If/Else true
    # and, once the limit is lowered, from the limit too (its entryRoutes must gain that way in).
    lit = _node("lit", "literal_string", outputs=[{"id": "value", "type": "string"}], literalValue="again")
    head = _node("head", "add_to_list", ADD_PINS, [O], pinDefaults={"name": "passes", "item": "first"},
                 entryRoutes=[{"key": "start::exec-out", "sourceNodeId": "start", "sourceHandle": "exec-out", "label": "Entry 1"},
                              {"key": "if::false", "sourceNodeId": "if", "sourceHandle": "false", "label": "false"}],
                 inputRouteOverrides={"item": {"if::false": {"sourceNodeId": "lit", "sourceHandle": "value"}}})
    iff = _node("if", "if", IF_PINS[0], IF_PINS[1], pinDefaults={"condition": False})
    report = _node("report", "set_var", [X, {"id": "name", "type": "string"}, {"id": "value", "type": "any"}], [O],
                   pinDefaults={"name": "report", "value": "done"},
                   entryRoutes=[{"key": "if::true", "sourceNodeId": "if", "sourceHandle": "true", "label": "true"},
                                {"key": "start::exec-out", "sourceNodeId": "start", "sourceHandle": "exec-out", "label": "Entry 2"}])
    other = _node("other", "set_var", [X, {"id": "name", "type": "string"}, {"id": "value", "type": "any"}], [O], pinDefaults={"name": "other", "value": 1})
    seq = _node("seq", "sequence", [X], [{"id": "then:0", "type": "execution"}, {"id": "then:1", "type": "execution"}])
    nodes = [START, lit, seq, head, iff, report, other]
    edges = [
        _edge("start", "exec-out", "seq", "exec-in"),
        _edge("seq", "then:0", "head", "exec-in"),
        _edge("head", "exec-out", "if", "exec-in"),
        _edge("if", "true", "report", "exec-in"),
        _edge("if", "false", "head", "exec-in", loopLimit={"max": 2}),
        _edge("seq", "then:1", "report", "exec-in"),
    ]
    # the editor keys head's first entry on the Sequence
    head["data"]["entryRoutes"][0] = {"key": "seq::then:0", "sourceNodeId": "seq", "sourceHandle": "then:0", "label": "Entry 1"}
    report["data"]["entryRoutes"][1] = {"key": "seq::then:1", "sourceNodeId": "seq", "sourceHandle": "then:1", "label": "Entry 2"}
    vf = load_visualflow_json({"id": "t", "name": "t", "entryNode": "start", "nodes": nodes, "edges": edges})
    lowered = lower_loop_limits(vf)
    by_id = {n.id: n for n in lowered.nodes}
    limit_id = "loop-limit:if.false->head"
    assert [r["key"] for r in by_id["head"].data["entryRoutes"]] == ["seq::then:0", f"{limit_id}::again"]
    assert [r["key"] for r in by_id["report"].data["entryRoutes"]] == ["if::true", "seq::then:1", f"{limit_id}::limit_reached"]
    state = _run(nodes, edges)
    assert state.status.value == "completed", state.error
    assert state.vars["passes"] == ["first", "again", "again"] and state.vars["report"] == "done"


def test_both_branches_looping_back_end_at_the_limit():
    # Review 2026-10-14: the exit of one wire's limit used to skip the other wire's limit; handing
    # over to it instead never ends, so such a flow ends at the limit.
    head = _node("head", "add_to_list", ADD_PINS, [O], pinDefaults={"name": "passes", "item": "x"})
    iff = _node("if", "if", IF_PINS[0], IF_PINS[1], pinDefaults={"condition": False})
    nodes = [START, head, iff]
    edges = [
        _edge("start", "exec-out", "head", "exec-in"),
        _edge("head", "exec-out", "if", "exec-in"),
        _edge("if", "true", "head", "exec-in", loopLimit={"max": 2}),
        _edge("if", "false", "head", "exec-in", loopLimit={"max": 2}),
    ]
    lowered = lower_loop_limits(load_visualflow_json({"id": "t", "name": "t", "nodes": nodes, "edges": edges}))
    # No way forward exists (each limit would hand over to the other for ever): no exits, the path ends.
    assert not [e for e in lowered.edges if e.sourceHandle == "limit_reached"]
    state = _run(nodes, edges)
    assert state.status.value == "completed", state.error
    assert len(state.vars["passes"]) == 3  # the first pass, then false's two repeats, then it ends


def test_a_wire_keeps_its_identity_without_its_data():
    from abstractruntime.visualflow_compiler.visual.models import VisualEdge

    a = VisualEdge(source="s", target="t", sourceHandle="exec-out", targetHandle="exec-in", data={"loopLimit": {"max": 1}})
    b = VisualEdge(source="s", target="t", sourceHandle="exec-out", targetHandle="exec-in")
    assert a == b and hash(a) == hash(b)
