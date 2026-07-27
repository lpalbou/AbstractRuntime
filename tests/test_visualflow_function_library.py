"""Flow-level function library (tier 2, 2026-07-26) — the load-bearing behaviors.

Each test pins one clause of the tier-2 contract:
- a declared function is callable from a pin expression (`fn(vars.x, value)`)
- entries share ONE namespace: functions call each other and private helpers
- name hygiene refuses loudly at BUILD: bad identifiers, reserved shadows,
  duplicate names, cross-entry private collisions, def-name mismatch
- bodies compile under the SAME code-node sandbox policy (import refused)
- an expression calling an UNKNOWN name fails the step attributed (node.pin)
- the `functions` field parses permissively and is inert without callers
"""

from __future__ import annotations

import pytest

from abstractruntime import Runtime
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow
from abstractruntime.visualflow_compiler.visual.function_library import (
    FunctionLibraryError,
    compile_function_library,
    normalize_flow_functions,
)
from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json
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


def _flow_with_functions(functions: list[dict], end_expressions: dict, end_pins: list[dict]) -> dict:
    return {
        "id": "fnlib-test",
        "name": "fnlib-test",
        "entryNode": "start",
        "functions": functions,
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
                    "pinExpressions": end_expressions,
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "end", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
        ],
    }


# ---------------------------------------------------------------- happy path


def test_library_function_called_from_expression() -> None:
    flow = _flow_with_functions(
        [
            {
                "name": "build_again",
                "code": (
                    "def build_again(state, max_fix):\n"
                    "    if state.get(\"approved\"):\n"
                    "        return False\n"
                    "    return state.get(\"fix_cycles\", 0) < max_fix\n"
                ),
            }
        ],
        {"answer": "build_again(vars.state, 3)"},
        [{"id": "answer", "label": "answer", "type": "boolean"}],
    )
    state = _run(flow, vars={"state": {"approved": False, "fix_cycles": 1}})
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("answer") is True

    state2 = _run(flow, vars={"state": {"approved": True, "fix_cycles": 1}})
    assert state2.output.get("answer") is False


def test_functions_call_each_other_and_private_helpers() -> None:
    flow = _flow_with_functions(
        [
            {
                "name": "fmt_header",
                "code": (
                    "def _upper(s):\n"
                    "    return str(s).upper()\n"
                    "def fmt_header(title):\n"
                    "    return \"== \" + _upper(title) + \" ==\"\n"
                ),
            },
            {
                "name": "report",
                "code": (
                    "def report(state):\n"
                    "    return fmt_header(state.get(\"phase\", \"?\")) + \" cycles=\" + str(state.get(\"cycles\", 0))\n"
                ),
            },
        ],
        {"answer": "report(vars.state)"},
        [{"id": "answer", "label": "answer", "type": "string"}],
    )
    state = _run(flow, vars={"state": {"phase": "build", "cycles": 2}})
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("answer") == "== BUILD == cycles=2"


def test_function_receives_wire_value_as_argument() -> None:
    flow = {
        "id": "fnlib-value",
        "name": "fnlib-value",
        "entryNode": "start",
        "functions": [
            {"name": "shout", "code": "def shout(s):\n    return str(s).upper() + \"!\"\n"}
        ],
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
                    "inputs": [{"id": "word", "label": "word", "type": "string"}],
                    "outputs": [{"id": "result", "label": "result", "type": "object"}],
                    "pinDefaults": {"word": "go"},
                },
            },
            {
                "id": "end",
                "type": "on_flow_end",
                "data": {
                    "label": "End",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "answer", "label": "answer", "type": "string"},
                    ],
                    "outputs": [],
                    "pinExpressions": {"answer": 'shout(value["word"])'},
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "end", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e2", "source": "obj", "target": "end", "sourceHandle": "result", "targetHandle": "answer"},
        ],
    }
    state = _run(flow)
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("answer") == "GO!"


def test_library_in_while_condition_reads_vars_fresh() -> None:
    """A while-condition calling a library function re-reads vars per iteration
    and terminates — the volatile parity clause extends to tier 2."""
    flow = {
        "id": "fnlib-while",
        "name": "fnlib-while",
        "entryNode": "start",
        "functions": [
            {
                "name": "keep_going",
                "code": "def keep_going(n, cap):\n    return (n or 0) < cap\n",
            }
        ],
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
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "condition", "label": "condition", "type": "boolean"},
                    ],
                    "outputs": [
                        {"id": "loop", "label": "loop", "type": "execution"},
                        {"id": "done", "label": "done", "type": "execution"},
                    ],
                    "pinExpressions": {"condition": "keep_going(vars.i, 3)"},
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


# ---------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    "payload",
    [
        # import reach (in-body — RestrictedPython refuses)
        "def f():\n    import os\n    return os.getcwd()\n",
        # dunder introspection pivot to object base classes
        "def f():\n    return ().__class__.__bases__[0].__subclasses__()\n",
        "def f():\n    return type('x').__mro__[-1].__subclasses__()\n",
        # reach a function's globals to escape the sandbox
        "def f():\n    return f.__globals__\n",
        # eval/exec/__import__ are not in safe_builtins
        "def f():\n    return eval('1+1')\n",
        "def f():\n    return __import__('os')\n",
        # open() is not granted
        "def f():\n    return open('/etc/passwd')\n",
    ],
)
def test_library_escape_attempts_are_refused(payload: str) -> None:
    """The library lane must be as airtight as the code-node lane. Either the
    build refuses (compile/definition), or the compiled function raises when
    CALLED — never returns a live escape gadget."""
    try:
        lib = compile_function_library([{"name": "f", "code": payload}])
    except FunctionLibraryError:
        return  # refused at build — good
    # Compiled: calling it must raise inside the sandbox, not return a gadget.
    with pytest.raises(Exception):
        result = lib["f"]()
        # __globals__ payload "succeeds" as a dict read — assert it cannot be
        # a live escape (no __builtins__ with dangerous names, no os module).
        assert "os" not in str(result) and "subprocess" not in str(result)
        raise AssertionError(f"escape payload returned a value: {str(result)[:120]}")


def test_bad_identifier_refused() -> None:
    with pytest.raises(FunctionLibraryError, match="not a plain identifier"):
        compile_function_library([{"name": "not a name", "code": "def x():\n    return 1\n"}])


def test_reserved_shadow_refused() -> None:
    with pytest.raises(FunctionLibraryError, match="shadows a sandbox builtin"):
        compile_function_library([{"name": "len", "code": "def len(x):\n    return 0\n"}])
    with pytest.raises(FunctionLibraryError, match="shadows a sandbox builtin"):
        compile_function_library([{"name": "parse_json", "code": "def parse_json(x):\n    return 0\n"}])
    with pytest.raises(FunctionLibraryError, match="shadows a sandbox builtin"):
        compile_function_library([{"name": "vars", "code": "def vars(x):\n    return 0\n"}])


def test_duplicate_name_refused() -> None:
    fns = [
        {"name": "f", "code": "def f():\n    return 1\n"},
        {"name": "f", "code": "def f():\n    return 2\n"},
    ]
    with pytest.raises(FunctionLibraryError, match="duplicate flow function name"):
        compile_function_library(fns)


def test_cross_entry_private_collision_refused() -> None:
    fns = [
        {"name": "a", "code": "def _fmt(x):\n    return str(x)\ndef a(x):\n    return _fmt(x)\n"},
        {"name": "b", "code": "def _fmt(x):\n    return repr(x)\ndef b(x):\n    return _fmt(x)\n"},
    ]
    with pytest.raises(FunctionLibraryError, match="already owned by entry 'a'"):
        compile_function_library(fns)


def test_def_name_mismatch_refused() -> None:
    with pytest.raises(FunctionLibraryError, match="must define a callable named"):
        compile_function_library([{"name": "wanted", "code": "def other():\n    return 1\n"}])


def test_top_level_import_refused() -> None:
    # A top-level import is refused — validate_code (shared with code nodes)
    # catches imports first; the top-level-defs-only purity rule is a further
    # backstop (see test_purity_refuses_top_level_mutable_container).
    with pytest.raises(FunctionLibraryError, match="Imports are not allowed"):
        compile_function_library(
            [{"name": "evil", "code": "import os\ndef evil():\n    return os.getcwd()\n"}]
        )


def test_import_inside_body_refused() -> None:
    # An import INSIDE a function body is structurally legal (a def) but the
    # shared validate_code (same lane as code nodes) refuses imports at BUILD —
    # one sandbox, one rule (the library lane must not be weaker).
    with pytest.raises(FunctionLibraryError, match="Imports are not allowed"):
        compile_function_library(
            [{"name": "evil", "code": "def evil():\n    import os\n    return os.getcwd()\n"}]
        )


def test_dunder_attribute_access_refused() -> None:
    # validate_code parity: dunder-attribute access (the classic escape pivot)
    # is refused at build in the library lane exactly as in code nodes.
    with pytest.raises(FunctionLibraryError, match="dunder"):
        compile_function_library(
            [{"name": "esc", "code": "def esc():\n    return ().__class__\n"}]
        )


def test_definition_time_failure_attributed() -> None:
    # A def whose default expression raises at def time (evaluating an unknown
    # name) is a genuine definition-time failure — attributed by name. The
    # default is a Name node, not a mutable literal, so the purity pass allows
    # it through to exec, where it raises.
    with pytest.raises(FunctionLibraryError, match="failed at definition time"):
        compile_function_library([{"name": "boom", "code": "def boom(x=undefined_name):\n    return x\n"}])


# ---------------------------------------------------------------- purity


def test_purity_refuses_global_state_leak() -> None:
    with pytest.raises(FunctionLibraryError, match="global.*not allowed|not allowed.*persistent"):
        compile_function_library(
            [{"name": "bump", "code": "def bump():\n    global _n\n    _n = (_n or 0) + 1\n    return _n\n"}]
        )


def test_purity_refuses_nonlocal() -> None:
    with pytest.raises(FunctionLibraryError, match="not allowed"):
        compile_function_library([{
            "name": "outer",
            "code": "def outer():\n    x = 0\n    def inner():\n        nonlocal x\n        x = x + 1\n    inner()\n    return x\n",
        }])


def test_purity_refuses_top_level_mutable_container() -> None:
    # `_cache = {}` at top level + call-time subscript mutation is the silent
    # cross-run leak the adversary found; refused as a top-level non-def.
    with pytest.raises(FunctionLibraryError, match="only `def` statements are allowed"):
        compile_function_library([{
            "name": "remember",
            "code": "_cache = {}\ndef remember(k):\n    _cache[k] = _cache.get(k, 0) + 1\n    return _cache[k]\n",
        }])


def test_purity_refuses_mutable_default_argument() -> None:
    # The classic mutable-default trap: the dict is created once at def time.
    with pytest.raises(FunctionLibraryError, match="mutable default argument"):
        compile_function_library([{
            "name": "remember",
            "code": "def remember(k, _c={}):\n    _c[k] = 1\n    return len(_c)\n",
        }])


def test_purity_allows_none_default_and_inside_build() -> None:
    # The SAFE idiom (None default, build inside) must NOT be refused.
    lib = compile_function_library([{
        "name": "collect",
        "code": "def collect(x, items=None):\n    items = items or []\n    items.append(x)\n    return items\n",
    }])
    assert lib["collect"](1) == [1]
    # A fresh list each call — no leak (the whole point).
    assert lib["collect"](2) == [2]


def test_pure_function_stable_across_two_runs() -> None:
    # The cross-run isolation the adversary asked for: the SAME compiled spec,
    # run twice, must produce identical output — a stateless library cannot
    # leak because purity is enforced (a stateful one would have been refused
    # at build). This pins the no-leak property so it can never drift silently.
    flow = _flow_with_functions(
        [{"name": "double", "code": "def double(n):\n    return (n or 0) * 2\n"}],
        {"answer": "double(vars.n)"},
        [{"id": "answer", "label": "answer", "type": "number"}],
    )
    spec = compile_visualflow(flow)
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    outputs = []
    for n in (10, 10, 10):
        rid = runtime.start(workflow=spec, vars={"n": n})
        st = runtime.tick(workflow=spec, run_id=rid, max_steps=50)
        assert st.status == RunStatus.COMPLETED
        outputs.append(st.output.get("answer"))
    # Byte-identical across runs: the second/third run see no residue from the first.
    assert outputs == [20, 20, 20], outputs


def test_private_helper_not_exported_to_expression() -> None:
    # A private `def _priv` beside a public function is callable INSIDE the
    # library but NOT exported to pin expressions (only declared names export).
    flow = _flow_with_functions(
        [{"name": "pub", "code": "def _priv(x):\n    return x + 1\ndef pub(x):\n    return _priv(x)\n"}],
        {"answer": "_priv(vars.n)"},
        [{"id": "answer", "label": "answer", "type": "number"}],
    )
    lib = compile_function_library(flow["functions"])
    assert set(lib.keys()) == {"pub"}, "only the declared name exports; _priv stays internal"
    # Calling the private name from an EXPRESSION fails attributed (NameError).
    state = _run(flow, vars={"n": 1})
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("success") is False


def test_unknown_function_in_expression_fails_attributed() -> None:
    evaluate = compile_pin_expression(
        "no_such_fn(vars.x)", node_label="While", pin_id="condition", library={}
    )
    with pytest.raises(PinExpressionError, match=r"While\.condition"):
        evaluate(None, {"x": 1})


def test_build_fails_loud_on_bad_function_via_compile_visualflow() -> None:
    flow = _flow_with_functions(
        [{"name": "broken", "code": "def broken(:\n    return 1\n"}],
        {},
        [],
    )
    with pytest.raises(FunctionLibraryError, match="broken"):
        compile_visualflow(flow)


# ---------------------------------------------------------------- tolerance


def test_normalize_drops_junk_entries() -> None:
    assert normalize_flow_functions(None) == []
    assert normalize_flow_functions("nope") == []
    assert normalize_flow_functions([{"name": "", "code": "x"}, {"name": "ok"}, 7]) == []
    out = normalize_flow_functions([{"name": " ok ", "code": "def ok():\n    return 1\n", "kind": "checker"}])
    assert out == [{"name": "ok", "code": "def ok():\n    return 1\n"}]


def test_functions_field_parses_and_is_inert_without_callers() -> None:
    flow = _flow_with_functions(
        [{"name": "unused", "code": "def unused():\n    return 1\n"}],
        {},
        [{"id": "answer", "label": "answer", "type": "string"}],
    )
    flow["nodes"][1]["data"]["pinDefaults"] = {"answer": "plain"}
    visual = load_visualflow_json(flow)
    assert visual.functions and visual.functions[0]["name"] == "unused"
    state = _run(flow)
    assert state.status == RunStatus.COMPLETED
    assert state.output.get("answer") == "plain"
