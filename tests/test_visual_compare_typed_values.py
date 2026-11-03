"""Compare executes values by their types; text next to a number or true/false is read exactly
or refused loudly (ADR 0049, amended 2026-11-03).

The flow fixtures are AbstractFlow's (`src/utils/__fixtures__/typed_literals_flow*.json`): the
editor saves Compare `b` typed next to a wired number as the JSON number; a flow saved by Flow
0.13.0 stored the text. Both compare 5 == 5 and 10 > 9. `compare_text_operands.json` is the table
of the conversion grammar (the editor's `parseLiteralText`): what converts, and what is refused.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from abstractruntime import Runtime, RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow
from abstractruntime.visualflow_compiler.visual.builtins import control_compare

pytestmark = pytest.mark.basic

FIXTURES = Path(__file__).parent / "fixtures" / "visualflow"
TEXT_OPERANDS = json.loads((FIXTURES / "compare_text_operands.json").read_text(encoding="utf-8"))["cases"]


def _run(name: str):
    spec = compile_visualflow(json.loads((FIXTURES / name).read_text(encoding="utf-8")))
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    run_id = runtime.start(workflow=spec, vars={})
    return runtime.tick(workflow=spec, run_id=run_id)


def test_typed_flow_compares_numbers() -> None:
    state = _run("typed_literals_flow.json")
    assert state.status == RunStatus.COMPLETED
    assert state.output == {"same": True, "above": True, "success": True}


def test_flow_saved_by_flow_0_13_with_text_runs_like_the_typed_flow() -> None:
    state = _run("typed_literals_flow.legacy.json")
    assert state.status == RunStatus.COMPLETED
    assert state.output == {"same": True, "above": True, "success": True}


@pytest.mark.parametrize("case", TEXT_OPERANDS, ids=lambda c: f"{c['a']!r} {c['op']} {c['text']!r}")
def test_text_next_to_a_typed_value_converts_exactly_or_is_refused(case: Dict[str, Any]) -> None:
    a, text, op, expected = case["a"], case["text"], case["op"], case["expected"]
    mirrored = {"==": "==", "!=": "!=", "<": ">", "<=": ">=", ">": "<", ">=": "<="}[op]
    for inputs in ({"a": a, "b": text, "op": op}, {"a": text, "b": a, "op": mirrored}):
        if expected == "refused":
            with pytest.raises(ValueError, match=r"is text \(.*\), not (a number|true/false);"):
                control_compare(inputs)
        else:
            assert control_compare(inputs) is expected


def test_the_refusal_names_the_node_the_text_and_the_type() -> None:
    with pytest.raises(ValueError) as info:
        control_compare({"a": 5, "b": "abc", "op": "=="}, node="'Is five' (node same)")
    assert str(info.value) == "Compare 'Is five' (node same) ==: b is text ('abc'), not a number; a is a number (5)."
    with pytest.raises(ValueError) as info:
        control_compare({"a": "yes", "b": True})
    assert str(info.value) == "Compare ==: a is text ('yes'), not true/false; b is true/false (True)."


def test_a_refused_text_stops_the_run_naming_the_node() -> None:
    flow = json.loads((FIXTURES / "typed_literals_flow.legacy.json").read_text(encoding="utf-8"))
    for node in flow["nodes"]:
        if node["id"] == "same":
            data = node["data"]
            data["pinDefaults"]["b"] = "five"
    spec = compile_visualflow(flow)
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    state = runtime.tick(workflow=spec, run_id=runtime.start(workflow=spec, vars={}))
    out: Dict[str, Any] = dict(state.output or {})
    assert out.get("success") is False
    assert "Compare 'Is five' (node same) ==: b is text ('five'), not a number; a is a number (5)." in str(out.get("error"))
    assert "same" not in out  # the run stopped at the refusal: no silent False was produced


@pytest.mark.parametrize(
    "a,b,op,expected",
    [
        (5, 5, "==", True),
        (5, 5.0, "==", True),
        ("5", "5", "==", True),
        ("abc", "abd", "<", True),
        ("10", "9", ">", False),  # text against text is never converted: "10" < "9" as text
        ("5", "5.0", "==", False),
        (10, 9, ">", True),
        (None, None, "==", True),
        ([1], [1], "==", True),
        ({"a": 1}, {"a": 2}, "!=", True),
        (None, 5, "==", False),  # Python's equality across other types is unchanged
        (True, 1, "==", True),  # true/false is a number to Python: True == 1, False < 1
        (False, 1, "<", True),
    ],
)
def test_same_type_and_python_equality_unchanged(a: Any, b: Any, op: str, expected: bool) -> None:
    assert control_compare({"a": a, "b": b, "op": op}) is expected


def test_no_operator_is_equality() -> None:
    assert control_compare({"a": 3, "b": 3}) is True


@pytest.mark.parametrize("a,b", [(None, 5), ([1], 2), ({"a": 1}, {"a": 2})])
def test_an_order_between_values_python_cannot_order_is_refused(a: Any, b: Any) -> None:
    with pytest.raises(ValueError, match="cannot be ordered"):
        control_compare({"a": a, "b": b, "op": ">"})


def test_unknown_operator_still_refused() -> None:
    with pytest.raises(ValueError, match="Unknown comparison operator"):
        control_compare({"a": 1, "b": 2, "op": "<>"})
