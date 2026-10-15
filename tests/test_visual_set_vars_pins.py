"""Set Variables: one input pin per variable (operator 2026-10-11, "set multiple variables at once ...
just defining name and type ... and optionally a default value").

A data input pin other than `updates` is one variable: its id is the name, its value (a wire, else
the pin default) is written; pins win over the same name in the `updates` object.
"""

from __future__ import annotations

from abstractruntime.core.runtime import Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler.compiler import compile_flow
from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

X = {"id": "exec-in", "type": "execution"}
O = {"id": "exec-out", "type": "execution"}


def _run(pins, pin_defaults, edges=(), extra_nodes=()):
    vf = load_visualflow_json(
        {
            "id": "set-vars-pins",
            "name": "set-vars-pins",
            "entryNode": "start",
            "nodes": [
                {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start", "outputs": [O]}},
                {
                    "id": "sv",
                    "type": "set_vars",
                    "data": {
                        "nodeType": "set_vars",
                        "inputs": [X, {"id": "updates", "type": "object"}, *pins],
                        "outputs": [O, {"id": "updates", "type": "object"}],
                        "pinDefaults": pin_defaults,
                    },
                },
                *extra_nodes,
            ],
            "edges": [{"source": "start", "sourceHandle": "exec-out", "target": "sv", "targetHandle": "exec-in"}, *edges],
        }
    )
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    wf = compile_flow(visual_to_flow(vf))
    rid = rt.start(workflow=wf, vars={})
    return rt.tick(workflow=wf, run_id=rid, max_steps=20)


def test_each_variable_pin_writes_its_default_or_its_wire():
    text = {"id": "txt", "type": "literal_string", "data": {"nodeType": "literal_string", "outputs": [{"id": "value", "type": "string"}], "literalValue": "Global Task: x"}}
    state = _run(
        [{"id": "history", "type": "array"}, {"id": "review_round", "type": "number"}, {"id": "task", "type": "string"}, {"id": "state.flags", "type": "object"}],
        {"history": [], "review_round": 0, "state.flags": {"done": False}},
        edges=[{"source": "txt", "sourceHandle": "value", "target": "sv", "targetHandle": "task"}],
        extra_nodes=[text],
    )
    assert state.status.value == "completed", state.error
    assert state.vars["history"] == [] and state.vars["review_round"] == 0
    assert state.vars["task"] == "Global Task: x"
    assert state.vars["state"]["flags"] == {"done": False}  # dotted names nest


def test_variable_pins_win_over_the_updates_object():
    state = _run([{"id": "a", "type": "number"}], {"updates": {"a": 1, "b": 2}, "a": 9})
    assert (state.vars["a"], state.vars["b"]) == (9, 2)


def test_a_reserved_name_is_refused():
    state = _run([{"id": "_secret", "type": "string"}], {"_secret": "x"})
    assert "reserved" in str(state.vars.get("_flow_error") or state.output)
