"""Inline pin expressions (tier 1, 2026-07-25) — the load-bearing behaviors.

Each test pins one clause of the design contract:
- expressions REPLACE the pin's resolved value; wire/default arrive as `value`
- while-conditions re-read `vars.*` FRESH each iteration (volatile parity
  with the get_var chains they replace) and terminate
- an unparseable expression fails the BUILD naming node+pin (compile is the
  honesty boundary)
- a raising expression fails the consumer's STEP naming node+pin (attribution
  from a zero-trace baseline)
- `vars` is read-only and unknown names raise helpfully
- the skew-safe encoding: an ABSENT expression leaves the pin on its default
  (the old-compiler degrade is bounded-by-falsy, never truthy-dict spin)
- parse_json/to_json exist in BOTH the expression env and code-node bodies
"""

from __future__ import annotations

import pytest

from abstractruntime import Runtime
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow
from abstractruntime.visualflow_compiler.visual.pin_expressions import (
    PinExpressionError,
    compile_pin_expression,
)


def _run(flow: dict, vars: dict | None = None):
    spec = compile_visualflow(flow)
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    run_id = runtime.start(workflow=spec, vars=vars or {})
    state = runtime.tick(workflow=spec, run_id=run_id, max_steps=200)
    return state


def _start_end_flow(end_extra: dict, end_pins: list[dict]) -> dict:
    return {
        "id": "fx-test",
        "name": "fx-test",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "end",
                "type": "on_flow_end",
                "data": {
                    "label": "The End",
                    "inputs": [{"id": "exec-in", "label": "", "type": "execution"}, *end_pins],
                    "outputs": [],
                    **end_extra,
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "end", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
        ],
    }


def test_expression_replaces_pin_value_reading_vars() -> None:
    flow = _start_end_flow(
        {"pinExpressions": {"answer": "vars.count * 2"}},
        [{"id": "answer", "label": "answer", "type": "number"}],
    )
    state = _run(flow, vars={"count": 21})
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("answer") == 42


def test_expression_sees_wire_value_and_default_as_value() -> None:
    # Wired pin: the wire's value arrives as `value` and the expression transforms it.
    flow = {
        "id": "fx-value",
        "name": "fx-value",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "obj",
                "type": "make_object",
                "data": {
                    "inputs": [{"id": "phase", "label": "phase", "type": "string"}],
                    "outputs": [{"id": "result", "label": "result", "type": "object"}],
                    "pinDefaults": {"phase": "fixing"},
                },
            },
            {
                "id": "end",
                "type": "on_flow_end",
                "data": {
                    "label": "End",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "wired", "label": "wired", "type": "string"},
                        {"id": "defaulted", "label": "defaulted", "type": "string"},
                    ],
                    "outputs": [],
                    "pinDefaults": {"defaulted": "hello"},
                    "pinExpressions": {
                        "wired": 'value["phase"] + "!"',
                        "defaulted": "value.upper()",
                    },
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "end", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e2", "source": "obj", "target": "end", "sourceHandle": "result", "targetHandle": "wired"},
        ],
    }
    state = _run(flow)
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("wired") == "fixing!"
    assert state.output.get("defaulted") == "HELLO"


def test_while_condition_expression_reads_vars_fresh_and_terminates() -> None:
    # while(vars.i < 3) { i = i + 1 } — the condition is a pin expression;
    # each iteration must see the updated var (volatile parity) and the loop
    # must exit at i == 3.
    flow = {
        "id": "fx-while",
        "name": "fx-while",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "loop",
                "type": "while",
                "data": {
                    "label": "Loop",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "condition", "label": "condition", "type": "boolean"},
                    ],
                    "outputs": [
                        {"id": "loop", "label": "loop", "type": "execution"},
                        {"id": "done", "label": "done", "type": "execution"},
                    ],
                    "pinExpressions": {"condition": "vars.i < 3"},
                },
            },
            {
                "id": "inc",
                "type": "code",
                "data": {
                    "label": "inc",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "i", "label": "i", "type": "number"},
                    ],
                    "outputs": [
                        {"id": "exec-out", "label": "", "type": "execution"},
                        {"id": "next", "label": "next", "type": "number"},
                    ],
                    "code": "def transform(_input):\n    return {\"next\": (_input.get(\"i\") or 0) + 1}\n",
                    "pinExpressions": {"i": "vars.i"},
                },
            },
            {
                "id": "setv",
                "type": "set_var",
                "data": {
                    "label": "set i",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "name", "label": "name", "type": "string"},
                        {"id": "value", "label": "value", "type": "number"},
                    ],
                    "outputs": [{"id": "exec-out", "label": "", "type": "execution"}],
                    "pinDefaults": {"name": "i"},
                },
            },
            {
                "id": "end",
                "type": "on_flow_end",
                "data": {
                    "label": "End",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "final_i", "label": "final_i", "type": "number"},
                    ],
                    "outputs": [],
                    "pinExpressions": {"final_i": "vars.i"},
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "loop", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e2", "source": "loop", "target": "inc", "sourceHandle": "loop", "targetHandle": "exec-in"},
            {"id": "e3", "source": "inc", "target": "setv", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e4", "source": "inc", "target": "setv", "sourceHandle": "next", "targetHandle": "value"},
            {"id": "e5", "source": "loop", "target": "end", "sourceHandle": "done", "targetHandle": "exec-in"},
        ],
    }
    state = _run(flow, vars={"i": 0})
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("final_i") == 3


def test_unparseable_expression_fails_the_build_naming_node_and_pin() -> None:
    flow = _start_end_flow(
        {"pinExpressions": {"answer": "vars.count <"}},
        [{"id": "answer", "label": "answer", "type": "number"}],
    )
    with pytest.raises(PinExpressionError) as exc:
        compile_visualflow(flow)
    msg = str(exc.value)
    assert "The End" in msg and "answer" in msg


def test_raising_expression_on_end_node_folds_into_output_error() -> None:
    # DESIGNED end-node semantics (pre-existing): a flow that reaches its end
    # reports resolution failures IN-BAND (success: False + error) so parent
    # flows read one contract. The expression error must arrive ATTRIBUTED.
    flow = _start_end_flow(
        {"pinExpressions": {"answer": 'vars.build_state["missing"]'}},
        [{"id": "answer", "label": "answer", "type": "number"}],
    )
    state = _run(flow, vars={"build_state": {}})
    assert state.status == RunStatus.COMPLETED
    out = state.output if isinstance(state.output, dict) else {}
    assert out.get("success") is False
    err = str(out.get("error") or "")
    assert "The End" in err and "answer" in err
    assert "vars.build_state" in err  # the expression preview rides the error


def test_raising_expression_mid_flow_stops_the_flow_with_attribution() -> None:
    # The visual-flow error contract (pre-existing, uniform): a step handler
    # exception COMPLETES the run with {"success": False, "error", "node"} and
    # stops execution there — the same in-band contract guard nodes and
    # subflow callers read for code-body failures. Expressions join it; the
    # requirement this test pins is ATTRIBUTION (node + pin in the error)
    # and STOPPAGE (downstream never ran).
    flow = {
        "id": "fx-fail",
        "name": "fx-fail",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "worker",
                "type": "code",
                "data": {
                    "label": "Worker",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "n", "label": "n", "type": "number"},
                    ],
                    "outputs": [{"id": "exec-out", "label": "", "type": "execution"}],
                    "code": "def transform(_input):\n    return {}\n",
                    "pinExpressions": {"n": 'vars.state["missing"]'},
                },
            },
            {
                "id": "end",
                "type": "on_flow_end",
                "data": {
                    "label": "End",
                    "inputs": [{"id": "exec-in", "label": "", "type": "execution"}],
                    "outputs": [],
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "worker", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e2", "source": "worker", "target": "end", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
        ],
    }
    state = _run(flow, vars={"state": {}})
    assert state.status == RunStatus.COMPLETED
    out = state.output if isinstance(state.output, dict) else {}
    assert out.get("success") is False
    assert out.get("node") == "worker", "execution stopped AT the raising node"
    err = str(out.get("error") or "")
    assert "Worker" in err and ".n" in err


def test_unknown_var_raises_helpfully() -> None:
    flow = _start_end_flow(
        {"pinExpressions": {"answer": "vars.nope"}},
        [{"id": "answer", "label": "answer", "type": "number"}],
    )
    state = _run(flow, vars={"count": 1})
    out = state.output if isinstance(state.output, dict) else {}
    assert out.get("success") is False
    assert "unknown run var 'nope'" in str(out.get("error") or "")


def test_absent_expression_leaves_default_semantics_untouched() -> None:
    # The skew-encoding property: without the pinExpressions key the pin
    # resolves to its default — a falsy while-condition stays bounded. This
    # is the exact behavior an old compiler exhibits when it ignores the key.
    flow = _start_end_flow(
        {},
        [{"id": "answer", "label": "answer", "type": "number"}],
    )
    state = _run(flow, vars={"count": 21})
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("answer") in (None, ""), "no expression, no default: pin resolves empty"


def test_parse_json_available_in_expressions_and_code_nodes() -> None:
    flow = {
        "id": "fx-json",
        "name": "fx-json",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "codey",
                "type": "code",
                "data": {
                    "label": "codey",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "raw", "label": "raw", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "exec-out", "label": "", "type": "execution"},
                        {"id": "parsed", "label": "parsed", "type": "object"},
                    ],
                    "code": 'def transform(_input):\n    return {"parsed": parse_json(_input.get("raw"))}\n',
                    "pinDefaults": {"raw": '{"ok": true}'},
                },
            },
            {
                "id": "end",
                "type": "on_flow_end",
                "data": {
                    "label": "End",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "ok", "label": "ok", "type": "boolean"},
                        {"id": "roundtrip", "label": "roundtrip", "type": "string"},
                    ],
                    "outputs": [],
                    "pinExpressions": {
                        "ok": 'parse_json(vars.blob)["ok"]',
                        "roundtrip": "to_json(value)",
                    },
                    "pinDefaults": {"roundtrip": {"x": 1}},
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "codey", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e2", "source": "codey", "target": "end", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
        ],
    }
    state = _run(flow, vars={"blob": '{"ok": true}'})
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("ok") is True
    assert state.output.get("roundtrip") == '{"x": 1}'


def test_expression_on_pure_node_applies_in_the_pure_resolution_lane() -> None:
    # THE SECOND APPLY SITE (cycle-1 review): pure (no-exec-pin) consumers
    # resolve inputs in `_ensure_node_output`, not in the data-aware exec
    # wrapper. An expression on a pure node's pin must apply there too — this
    # is the drift the shared `apply_pin_expressions` exists to prevent, so
    # both lanes stay test-pinned. Also exercises the `.get` and `in`
    # surfaces of the read-only vars view.
    flow = {
        "id": "fx-pure",
        "name": "fx-pure",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "obj",
                "type": "make_object",
                "data": {
                    "inputs": [{"id": "phase", "label": "phase", "type": "string"}],
                    "outputs": [{"id": "result", "label": "result", "type": "object"}],
                    "pinExpressions": {"phase": 'vars.get("phase_name", "?") if "phase_name" in vars else "absent"'},
                },
            },
            {
                "id": "end",
                "type": "on_flow_end",
                "data": {
                    "label": "End",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "answer", "label": "answer", "type": "object"},
                    ],
                    "outputs": [],
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "end", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e2", "source": "obj", "target": "end", "sourceHandle": "result", "targetHandle": "answer"},
        ],
    }
    state = _run(flow, vars={"phase_name": "fixing"})
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("answer") == {"phase": "fixing"}


def test_starred_call_in_expression_works() -> None:
    # `max(*vars.nums)` compiles to `_apply_(...)` under RestrictedPython even
    # in eval mode; the expression env must carry the guard or a legal
    # expression dies with a bare NameError naming sandbox internals.
    flow = _start_end_flow(
        {"pinExpressions": {"answer": "max(*vars.nums)"}},
        [{"id": "answer", "label": "answer", "type": "number"}],
    )
    state = _run(flow, vars={"nums": [3, 7, 5]})
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("answer") == 7


def test_bare_vars_expression_yields_plain_dict_not_the_view() -> None:
    # The read-only view is an evaluation-time artifact: `vars` alone (pass
    # the whole state through this pin) must hand the consumer a plain,
    # JSON-serializable dict — never the view object (which would alias live
    # run vars and break run-state persistence).
    import json

    flow = _start_end_flow(
        {"pinExpressions": {"answer": "vars"}},
        [{"id": "answer", "label": "answer", "type": "object"}],
    )
    state = _run(flow, vars={"count": 2, "name": "x"})
    assert state.status == RunStatus.COMPLETED
    answer = state.output.get("answer")
    assert isinstance(answer, dict)
    assert answer.get("count") == 2 and answer.get("name") == "x"
    json.dumps({k: v for k, v in answer.items() if not k.startswith("_")})  # must not raise


def test_vars_read_only_via_method_mutation_refused() -> None:
    # Eval-mode expressions cannot assign, but method calls could mutate the
    # TOP mapping if it were a plain dict; the read-only view has no mutating
    # surface — .pop simply does not exist on it (attribute access resolves
    # var names, so 'pop' reads as an unknown var).
    flow = _start_end_flow(
        {"pinExpressions": {"answer": 'vars.pop("count")'}},
        [{"id": "answer", "label": "answer", "type": "number"}],
    )
    state = _run(flow, vars={"count": 1})
    out = state.output if isinstance(state.output, dict) else {}
    assert out.get("success") is False
    assert "unknown run var 'pop'" in str(out.get("error") or "")


# ---------------------------------------------------------------------------
# Cycle 2 (adversarial robustness/security), 2026-07-25.
# ---------------------------------------------------------------------------


def _eval(expr: str, value=None, run_vars=None):
    """Compile + evaluate one expression directly (unit boundary)."""
    fn = compile_pin_expression(expr, node_label="N", pin_id="p")
    return fn(value, run_vars or {})


def _refused(expr: str, run_vars=None) -> bool:
    """True if the expression is refused at COMPILE or EVAL (sandbox held)."""
    try:
        fn = compile_pin_expression(expr, node_label="N", pin_id="p")
    except PinExpressionError:
        return True
    try:
        fn(None, run_vars or {})
    except PinExpressionError:
        return True
    return False


def test_view_never_escapes_when_nested_in_a_container() -> None:
    # THE cycle-2 hole: the top-level `vars` unwrap missed a view returned
    # INSIDE a container. `[vars]`, `{"v": vars}`, `(vars,)` all used to
    # persist the _ReadOnlyVars object into node outputs — not JSON
    # serializable, aliasing every run var. Strip it wherever it lands.
    import json

    rv = {"count": 2, "name": "x", "nested": {"k": 1}}

    in_list = _eval("[vars]", run_vars=rv)
    assert isinstance(in_list, list) and isinstance(in_list[0], dict)
    assert in_list[0].get("count") == 2

    in_dict = _eval('{"v": vars}', run_vars=rv)
    assert isinstance(in_dict["v"], dict) and in_dict["v"].get("name") == "x"

    in_tuple = _eval("(vars, 1)", run_vars=rv)
    assert isinstance(in_tuple[0], dict) and in_tuple[0].get("count") == 2

    deep = _eval('{"outer": [vars]}', run_vars=rv)
    assert isinstance(deep["outer"][0], dict)

    # None of the results carry the view object, so all serialize (the
    # persistence-integrity property: node outputs are saved into run vars).
    for result in (in_list, in_dict, list(in_tuple), deep):
        json.dumps(result, default=str)  # must not raise


def test_view_stripping_preserves_object_identity_for_viewless_containers() -> None:
    # Hot-path guarantee: a legitimate container result with NO view inside is
    # returned UNCHANGED (not defensively re-copied every resolution). We prove
    # it by observing that the returned list aliases the live nested list — the
    # documented get_var-parity posture — i.e. the strip walked but did not
    # rebuild.
    rv = {"items": [3, 1, 2]}
    result = _eval("vars.items", run_vars=rv)
    assert result is rv["items"], "viewless result must not be re-copied"


def test_nested_mutation_parity_between_expression_and_code_node_lanes() -> None:
    # POSTURE PIN (cycle 2): nested method-mutation is REACHABLE and is
    # deliberate parity with the get_var -> code-node lane (both hand out the
    # live object). This test asserts the two lanes behave IDENTICALLY so a
    # future one-sided "fix" (harden expressions but not code nodes, or vice
    # versa) breaks the parity loudly and forces both to move together.
    def _mutating_flow(node: dict, edges_extra: list[dict]) -> dict:
        return {
            "id": "fx-mut",
            "name": "fx-mut",
            "entryNode": "start",
            "nodes": [
                {"id": "start", "type": "on_flow_start",
                 "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]}},
                node,
                {"id": "end", "type": "on_flow_end",
                 "data": {"label": "End",
                          "inputs": [{"id": "exec-in", "label": "", "type": "execution"},
                                     {"id": "final", "label": "final", "type": "object"}],
                          "outputs": [],
                          "pinExpressions": {"final": "vars.items"}}},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": node["id"], "sourceHandle": "exec-out", "targetHandle": "exec-in"},
                {"id": "e2", "source": node["id"], "target": "end", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
                *edges_extra,
            ],
        }

    # Expression lane: an expression that appends to the live nested list.
    expr_node = {
        "id": "mut", "type": "code",
        "data": {"label": "mut",
                 "inputs": [{"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "sink", "label": "sink", "type": "any"}],
                 "outputs": [{"id": "exec-out", "label": "", "type": "execution"}],
                 "code": "def transform(_input):\n    return {}\n",
                 "pinExpressions": {"sink": "vars.items.append(99) or 0"}},
    }
    expr_state = _run(_mutating_flow(expr_node, []), vars={"items": [1, 2]})
    assert expr_state.status == RunStatus.COMPLETED
    assert expr_state.output.get("final") == [1, 2, 99], "expression mutated the live list"

    # Code-node lane via get_var: identical mutation reachable today.
    code_node = {
        "id": "mut", "type": "code",
        "data": {"label": "mut",
                 "inputs": [{"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "items", "label": "items", "type": "array"}],
                 "outputs": [{"id": "exec-out", "label": "", "type": "execution"}],
                 "code": "def transform(_input):\n    _input['items'].append(99)\n    return {}\n"},
    }
    getvar = {
        "id": "gv", "type": "get_var",
        "data": {"label": "gv",
                 "inputs": [{"id": "name", "label": "name", "type": "string"}],
                 "outputs": [{"id": "value", "label": "value", "type": "any"}],
                 "pinDefaults": {"name": "items"}},
    }
    code_flow = _mutating_flow(code_node, [
        {"id": "e3", "source": "gv", "target": "mut", "sourceHandle": "value", "targetHandle": "items"},
    ])
    code_flow["nodes"].append(getvar)
    code_state = _run(code_flow, vars={"items": [1, 2]})
    assert code_state.status == RunStatus.COMPLETED
    assert code_state.output.get("final") == [1, 2, 99], "code node mutated the live list identically"


@pytest.mark.parametrize(
    "expr",
    [
        "().__class__",                                   # dunder traversal (parse-refused)
        "().__class__.__bases__",                         # deeper traversal
        "type(vars).__mro__",                             # mro pivot
        "type('').__mro__[-1].__subclasses__()",          # subclasses gadget
        "vars.__class__.__init__.__globals__",            # globals pivot
        '"{0.__class__}".format(())',                     # format-string pivot (eval-refused)
        '"{x.__class__}".format_map({"x": ()})',          # format_map pivot
        'getattr((), "__class__")',                       # getattr not in builtins
        "__import__('os')",                               # import
        "open('/etc/passwd')",                            # file open
        "(_getattr_ := 5)",                               # walrus rebind of a guard name
        "(__builtins__ := {})",                           # walrus rebind of builtins
        "eval('1')",                                      # eval not in builtins
        "exec('x=1')",                                    # exec not in builtins
        "vars._vars",                                     # reach the view's backing store
    ],
)
def test_security_escape_attempts_are_refused(expr: str) -> None:
    # Each probe attempts a known sandbox-escape shape; the guarded eval (the
    # SAME _CodeNodePolicy + safer_getattr + safe_builtins the code-node lane
    # uses) must refuse it at compile or eval. A regression here is a real
    # sandbox escape — pin every refusal.
    assert _refused(expr, run_vars={"count": 1}), f"expected refusal for: {expr}"


def test_legitimate_expression_forms_still_evaluate() -> None:
    # The eval-mode allowances the design keeps: walrus, comprehension,
    # lambda, starred call. If a security tightening ever kills these, this
    # goes red — the sandbox must stay usable, not just safe.
    assert _eval("(z := vars.n) + z", run_vars={"n": 3}) == 6
    assert _eval("[x * 2 for x in vars.items]", run_vars={"items": [1, 2]}) == [2, 4]
    assert _eval("(lambda a: a + 1)(vars.n)", run_vars={"n": 4}) == 5
    assert _eval("max(*vars.nums)", run_vars={"nums": [2, 9, 4]}) == 9


def test_bounded_range_and_sum_available_resource_posture_is_shared() -> None:
    # range/sum are granted exactly as in the code-node sandbox; a bounded use
    # works. The UNBOUNDED case (sum(range(10**9))) blocks the tick thread
    # identically to a code body — a SHARED, documented posture (module
    # docstring), deliberately left to a future shared watchdog rather than an
    # expression-only guard. We do not run the unbounded case (it would hang);
    # we pin that the primitives are present so the posture claim is grounded.
    assert _eval("sum(range(101))", run_vars={}) == 5050


def test_expression_runs_even_when_a_wire_delivers_none() -> None:
    # A pin with BOTH a wire and an expression: the wire may deliver None, and
    # the expression must still run (value = None). Precedence is uniform —
    # the expression applies LAST and replaces whatever resolution produced.
    flow = {
        "id": "fx-wire-none", "name": "fx-wire-none", "entryNode": "start",
        "nodes": [
            {"id": "start", "type": "on_flow_start",
             "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]}},
            {"id": "worker", "type": "code",
             "data": {"label": "worker",
                      "inputs": [{"id": "exec-in", "label": "", "type": "execution"}],
                      "outputs": [{"id": "exec-out", "label": "", "type": "execution"},
                                  {"id": "out", "label": "out", "type": "any"}],
                      "code": "def transform(_input):\n    return {\"out\": None}\n"}},
            {"id": "end", "type": "on_flow_end",
             "data": {"label": "End",
                      "inputs": [{"id": "exec-in", "label": "", "type": "execution"},
                                 {"id": "answer", "label": "answer", "type": "string"}],
                      "outputs": [],
                      "pinExpressions": {"answer": '"got none" if value is None else "got " + str(value)'}}},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "worker", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e2", "source": "worker", "target": "end", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e3", "source": "worker", "target": "end", "sourceHandle": "out", "targetHandle": "answer"},
        ],
    }
    state = _run(flow)
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("answer") == "got none"


def test_expression_returning_none_on_a_pin_completes() -> None:
    # An expression may legitimately compute None; there is no runtime
    # required-pin enforcement (required-ness is a preflight/UI concept), so
    # None is simply the pin's value — the flow completes, pin is None.
    flow = _start_end_flow(
        {"pinExpressions": {"answer": "None"}},
        [{"id": "answer", "label": "answer", "type": "number"}],
    )
    state = _run(flow, vars={})
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("answer") is None


def test_expression_on_unknown_pin_id_is_harmless_at_runtime() -> None:
    # STALE expression (dynamic-pin rename/delete): pinExpressions carries a
    # pin id with no matching input pin. The evaluator still runs and writes
    # resolved_input[ghost]; that extra key is inert (on_flow_end reads only
    # declared pins). The run must COMPLETE, not choke. (Preflight warns the
    # author about the stale id before any run — editor half.)
    flow = _start_end_flow(
        {"pinExpressions": {"ghost": "vars.count"}},
        [{"id": "answer", "label": "answer", "type": "number"}],
    )
    state = _run(flow, vars={"count": 5})
    assert state.status == RunStatus.COMPLETED


def test_pin_named_value_does_not_collide_with_the_env_value() -> None:
    # Reserved-name pin id 'value': the env `value` is the pin's own resolved
    # value (its default here), so `value.upper()` on a pin literally named
    # 'value' works — no collision between the pin id and the eval binding.
    flow = _start_end_flow(
        {"pinDefaults": {"value": "hi"}, "pinExpressions": {"value": "value.upper()"}},
        [{"id": "value", "label": "value", "type": "string"}],
    )
    state = _run(flow, vars={})
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("value") == "HI"


def test_pin_named_vars_is_shadowed_by_the_run_vars_view_in_expressions() -> None:
    # Reserved-name pin id 'vars': inside an expression, `vars` ALWAYS means
    # the run-vars view — a pin default named 'vars' never shadows it. The
    # expression reads run vars (count=7), not the {"x":5} pin default.
    flow = _start_end_flow(
        {"pinDefaults": {"vars": {"x": 5}},
         "pinExpressions": {"answer": "vars.count"}},
        [{"id": "vars", "label": "vars", "type": "object"},
         {"id": "answer", "label": "answer", "type": "number"}],
    )
    state = _run(flow, vars={"count": 7})
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("answer") == 7


def test_unicode_and_emoji_in_expressions() -> None:
    # Non-ASCII identifiers and string literals survive compile + eval.
    assert _eval("vars.café", run_vars={"café": "☕"}) == "☕"
    assert _eval('"🎉 " + str(vars.n)', run_vars={"n": 1}) == "🎉 1"


def test_large_flat_expression_compiles_and_evaluates() -> None:
    # A ~10KB FLAT expression (a giant string literal, well past chip
    # truncation) must not break the compiler or evaluator — truncation is a
    # UI concern only. Flat size is fine; only DEPTH is a problem (below).
    big = "x" * 10_000
    assert _eval(f'"{big}" + str(vars.count)', run_vars={"count": 1}) == big + "1"


def test_view_inside_a_set_strips_to_plain_values() -> None:
    # The set/frozenset branch of the strip walk: a stripped view becomes an
    # UNHASHABLE dict, so a set-of-views cannot be rebuilt as a set — the
    # documented degenerate normalization is a list of plain values, never a
    # crash and never a leaked view. A viewless set stays a set (identity
    # branch).
    import json

    result = _eval("{vars}", run_vars={"count": 2})
    assert isinstance(result, list) and len(result) == 1
    assert isinstance(result[0], dict) and result[0].get("count") == 2
    json.dumps(result)  # must not raise: no view escaped

    untouched = _eval("{1, 2}", run_vars={})
    assert isinstance(untouched, (set, frozenset)) and untouched == {1, 2}


def test_junk_pin_expression_entries_behave_as_absent() -> None:
    # The normalize tolerance contract: null / blank / non-string entries are
    # DROPPED (never compiled, never raised) — the designed degrade, identical
    # to the old-compiler skew path: the pin simply resolves to its default
    # and the flow completes. A non-dict pinExpressions value is ignored the
    # same way.
    flow = _start_end_flow(
        {"pinExpressions": {"answer": None, "ghost": "   ", "blank": ""}},
        [{"id": "answer", "label": "answer", "type": "number"}],
    )
    state = _run(flow, vars={"count": 21})
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("answer") in (None, "")

    flow2 = _start_end_flow(
        {"pinExpressions": "vars.count"},
        [{"id": "answer", "label": "answer", "type": "number"}],
    )
    state2 = _run(flow2, vars={"count": 21})
    assert state2.status == RunStatus.COMPLETED
    assert state2.output.get("answer") in (None, "")


def test_pathologically_nested_expression_fails_build_attributed() -> None:
    # Cycle-2 finding: a deeply-nested expression (`a + 0 + 0 + ...` thousands
    # deep) overflows RestrictedPython's AST transformer with a RecursionError
    # — which used to escape the SyntaxError-only guard and crash the flow
    # BUILD with a bare stack trace. The compile boundary must fail LOUDLY and
    # ATTRIBUTED (node+pin) for ANY compile failure, not just parse errors.
    deep = "vars.count" + " + 0" * 5000
    with pytest.raises(PinExpressionError) as exc:
        compile_pin_expression(deep, node_label="Deep Node", pin_id="answer")
    msg = str(exc.value)
    assert "Deep Node" in msg and "answer" in msg
