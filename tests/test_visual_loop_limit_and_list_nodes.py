"""Visual nodes that replace pin-expression code in review loops (operator 2026-10-11: "i want a
visual understanding for non expert... so no code"):

- Add to List Variable (`add_to_list`): add one item and/or a list's items to a list variable.
- Loop Limit (`loop_limit`): `again` at most `max` times, then `limit_reached`; `count` = passes.
- String Template: a placeholder reads an input pin of its name, else the run variable.
- Set Variables / Add to List see only their own pins, never same-named keys of the previous step.
"""

from __future__ import annotations

from abstractruntime.core.runtime import Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler.compiler import compile_flow
from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

X = {"id": "exec-in", "type": "execution"}
O = {"id": "exec-out", "type": "execution"}


def _edge(s, sh, t, th):
    return {"source": s, "sourceHandle": sh, "target": t, "targetHandle": th}


def _node(nid, ntype, inputs=(), outputs=(), **data):
    return {"id": nid, "type": ntype, "data": {"nodeType": ntype, "inputs": list(inputs), "outputs": list(outputs), **data}}


def _run(nodes, edges, start_vars=None):
    vf = load_visualflow_json({"id": "t", "name": "t", "entryNode": "start", "nodes": nodes, "edges": edges})
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    wf = compile_flow(visual_to_flow(vf))
    rid = rt.start(workflow=wf, vars=dict(start_vars or {}))
    return rt.tick(workflow=wf, run_id=rid, max_steps=200)


START = _node("start", "on_flow_start", outputs=[O])
ADD_PINS = [X, {"id": "name", "type": "string"}, {"id": "item", "type": "any"}, {"id": "items", "type": "array"}]
CODE_OUT = [O, {"id": "result", "type": "any"}]


def _ambient(body):
    # A code step whose output has keys named like the next node's pins.
    return _node("code", "code", inputs=[X], outputs=CODE_OUT, codeBody=body)


# ---- Add to List Variable ---------------------------------------------------------------------


def _add(defaults, start_vars=None, extra_nodes=(), extra_edges=()):
    nodes = [START, _node("add", "add_to_list", ADD_PINS, [O, {"id": "list", "type": "array"}], pinDefaults=defaults), *extra_nodes]
    edges = [_edge("start", "exec-out", "add", "exec-in"), *extra_edges]
    return _run(nodes, edges, start_vars)


def test_item_is_one_element_items_are_each_added():
    state = _add({"name": "knowledge", "item": [1, 2], "items": ["k3", "k4"]}, {"knowledge": ["k1"]})
    assert state.status.value == "completed", state.error
    assert state.vars["knowledge"] == ["k1", [1, 2], "k3", "k4"]


def test_a_missing_variable_starts_empty_and_null_inputs_add_nothing():
    state = _add({"name": "history", "item": "first", "items": None})
    assert state.vars["history"] == ["first"]
    state = _add({"name": "history", "item": None}, {"history": ["a"]})
    assert state.vars["history"] == ["a"]


def test_a_variable_that_is_not_a_list_fails_the_step_naming_it():
    state = _add({"name": "history", "item": "x"}, {"history": "text"})
    assert "'history' holds a str, not a list" in str(state.vars.get("_flow_error"))
    state = _add({"name": "history", "items": "x"})
    assert "items must be a list" in str(state.vars.get("_flow_error"))


def test_the_new_list_is_an_output_and_the_variable_is_read_in_the_same_step():
    sink = _node("sv", "set_var", [X, {"id": "name", "type": "string"}, {"id": "value", "type": "any"}], [O], pinDefaults={"name": "copy"})
    state = _add(
        {"name": "history", "item": "b"},
        {"history": ["a"]},
        extra_nodes=[sink],
        extra_edges=[_edge("add", "exec-out", "sv", "exec-in"), _edge("add", "list", "sv", "value")],
    )
    assert state.vars["history"] == ["a", "b"] and state.vars["copy"] == ["a", "b"]


def test_add_to_list_ignores_same_named_keys_of_the_previous_step():
    nodes = [START, _ambient("return {'item': 'stray', 'items': ['stray']}"), _node("add", "add_to_list", ADD_PINS, [O], pinDefaults={"name": "history"})]
    edges = [_edge("start", "exec-out", "code", "exec-in"), _edge("code", "exec-out", "add", "exec-in")]
    state = _run(nodes, edges, {"history": ["a"]})
    assert state.vars["history"] == ["a"]


def test_set_variables_writes_only_its_rows_not_the_previous_steps_keys():
    sv = _node("sv", "set_vars", [X, {"id": "a", "type": "number"}], [O], pinDefaults={"a": 1})
    nodes = [START, _ambient("return {'stray': 1, 'a': 5}"), sv]
    edges = [_edge("start", "exec-out", "code", "exec-in"), _edge("code", "exec-out", "sv", "exec-in")]
    state = _run(nodes, edges)
    assert state.status.value == "completed", state.error
    assert state.vars["a"] == 1 and "stray" not in state.vars


# ---- Loop Limit -------------------------------------------------------------------------------


def _limited_loop(max_passes):
    limit = _node(
        "limit",
        "loop_limit",
        [X, {"id": "max", "type": "number"}],
        [{"id": "again", "type": "execution"}, {"id": "limit_reached", "type": "execution"}, {"id": "count", "type": "number"}],
        pinDefaults={"max": max_passes},
    )
    body = _node("add", "add_to_list", ADD_PINS, [O], pinDefaults={"name": "seen"})
    done = _node("done", "set_var", [X, {"id": "name", "type": "string"}, {"id": "value", "type": "any"}], [O], pinDefaults={"name": "final"})
    nodes = [START, limit, body, done]
    edges = [
        _edge("start", "exec-out", "limit", "exec-in"),
        _edge("limit", "again", "add", "exec-in"),
        _edge("limit", "count", "add", "item"),
        _edge("add", "exec-out", "limit", "exec-in"),
        _edge("limit", "limit_reached", "done", "exec-in"),
        _edge("limit", "count", "done", "value"),
    ]
    return _run(nodes, edges)


def test_loop_limit_lets_max_passes_through_then_takes_limit_reached():
    state = _limited_loop(3)
    assert state.status.value == "completed", state.error
    assert state.vars["seen"] == [1, 2, 3]
    assert state.vars["final"] == 3
    assert state.vars["_temp"]["loop_limits"]["limit"]["count"] == 0  # counts again from 0 next time


def test_loop_limit_zero_goes_straight_to_limit_reached():
    state = _limited_loop(0)
    assert "seen" not in state.vars and state.vars["final"] == 0


# ---- String Template --------------------------------------------------------------------------


def _template(text, pins=(), defaults=None, start_vars=None, wires=(), extra_nodes=()):
    tpl = _node("tpl", "string_template", [{"id": "template", "type": "string"}, *pins], [{"id": "result", "type": "string"}], pinDefaults={"template": text, **(defaults or {})})
    out = _node("out", "set_var", [X, {"id": "name", "type": "string"}, {"id": "value", "type": "any"}], [O], pinDefaults={"name": "text"})
    nodes = [START, tpl, out, *extra_nodes]
    edges = [_edge("start", "exec-out", "out", "exec-in"), _edge("tpl", "result", "out", "value"), *wires]
    return _run(nodes, edges, start_vars).vars["text"]


def test_placeholders_read_their_input_pin_then_the_run_variable():
    lit = _node("lit", "literal_string", outputs=[{"id": "value", "type": "string"}], literalValue="write tests")
    text = _template(
        "Goal: {{prompt}} / Task: {{task}} / Knowledge: {{knowledge}} / Done: {{knowledge | join(\", \")}}",
        pins=[{"id": "task", "type": "string"}],
        start_vars={"prompt": "ship it", "knowledge": ["a", "b"], "task": "not this"},
        wires=[_edge("lit", "value", "tpl", "task")],
        extra_nodes=[lit],
    )
    assert text == 'Goal: ship it / Task: write tests / Knowledge: ["a", "b"] / Done: a, b'


def test_the_vars_object_still_works_and_reserved_run_variables_stay_hidden():
    text = _template("{{who}} {{_temp}}", pins=[{"id": "vars", "type": "object"}], defaults={"vars": {"who": "Ada"}}, start_vars={"who": "Bob"})
    assert text == "Ada "


LIMIT_PINS_IN = [X, {"id": "max", "type": "number"}]
LIMIT_PINS_OUT = [{"id": "again", "type": "execution"}, {"id": "limit_reached", "type": "execution"}, {"id": "count", "type": "number"}]


def test_each_iteration_of_an_enclosing_loop_gets_its_own_count():
    # Review (adversarial, 2026-10-11): the count leaked from one outer iteration into the next.
    loop = _node(
        "for",
        "for",
        [X, {"id": "start", "type": "number"}, {"id": "end", "type": "number"}, {"id": "step", "type": "number"}],
        [{"id": "loop", "type": "execution"}, {"id": "done", "type": "execution"}, {"id": "i", "type": "number"}, {"id": "index", "type": "number"}],
        pinDefaults={"start": 0, "end": 3, "step": 1},
    )
    limit = _node("limit", "loop_limit", LIMIT_PINS_IN, LIMIT_PINS_OUT, pinDefaults={"max": 1})
    add = _node("add", "add_to_list", ADD_PINS, [O], pinDefaults={"name": "seen"})
    state = _run(
        [START, loop, limit, add],
        [
            _edge("start", "exec-out", "for", "exec-in"),
            _edge("for", "loop", "limit", "exec-in"),
            _edge("limit", "again", "add", "exec-in"),
            _edge("for", "i", "add", "item"),
        ],
    )
    assert state.status.value == "completed", state.error
    assert state.vars["seen"] == [0, 1, 2]


def test_loop_limit_ignores_a_max_key_of_the_previous_step():
    limit = _node("limit", "loop_limit", LIMIT_PINS_IN, LIMIT_PINS_OUT)  # no max set: 3
    took = _node("sv", "set_var", [X, {"id": "name", "type": "string"}, {"id": "value", "type": "any"}], [O], pinDefaults={"name": "took", "value": "again"})
    state = _run(
        [START, _ambient("return {'max': 0}"), limit, took],
        [_edge("start", "exec-out", "code", "exec-in"), _edge("code", "exec-out", "limit", "exec-in"), _edge("limit", "again", "sv", "exec-in")],
    )
    assert state.vars.get("took") == "again"


def test_a_value_json_cannot_write_is_written_as_text():
    import datetime

    text = _template("{{when}}", start_vars={"when": [datetime.date(2026, 10, 11)]})
    assert text == '["2026-10-11"]'


# ---- Get Variables ----------------------------------------------------------------------------


def test_get_variables_outputs_each_variable_by_its_pin():
    # Operator 2026-10-12: "in the same way we created set variables, ... a get variables that
    # enable us to get as many variables as we want".
    getter = _node("gv", "get_vars", outputs=[{"id": "history", "type": "array"}, {"id": "state.round", "type": "number"}, {"id": "missing", "type": "any"}])
    sink = _node("sv", "set_vars", [X, {"id": "copy", "type": "array"}, {"id": "round", "type": "number"}, {"id": "gone", "type": "any"}], [O])
    state = _run(
        [START, getter, sink],
        [
            _edge("start", "exec-out", "sv", "exec-in"),
            _edge("gv", "history", "sv", "copy"),
            _edge("gv", "state.round", "sv", "round"),
            _edge("gv", "missing", "sv", "gone"),
        ],
        {"history": ["a"], "state": {"round": 2}},
    )
    assert state.status.value == "completed", state.error
    assert (state.vars["copy"], state.vars["round"], state.vars["gone"]) == (["a"], 2, None)


def test_get_variables_reads_the_current_value_each_time():
    getter = _node("gv", "get_vars", outputs=[{"id": "seen", "type": "array"}])
    limit = _node("limit", "loop_limit", LIMIT_PINS_IN, LIMIT_PINS_OUT, pinDefaults={"max": 2})
    add = _node("add", "add_to_list", ADD_PINS, [O], pinDefaults={"name": "seen"})
    snap = _node("snap", "add_to_list", ADD_PINS, [O], pinDefaults={"name": "sizes"})
    state = _run(
        [START, limit, add, snap, getter],
        [
            _edge("start", "exec-out", "limit", "exec-in"),
            _edge("limit", "again", "add", "exec-in"),
            _edge("limit", "count", "add", "item"),
            _edge("add", "exec-out", "snap", "exec-in"),
            _edge("gv", "seen", "snap", "item"),
            _edge("snap", "exec-out", "limit", "exec-in"),
        ],
    )
    assert state.status.value == "completed", state.error
    assert state.vars["sizes"] == [[1], [1, 2]]
