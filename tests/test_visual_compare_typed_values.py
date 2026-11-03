"""Compare executes stored values by their types and refuses a mismatch loudly (ADR 0049).

The fixtures are AbstractFlow's (`src/utils/__fixtures__/typed_literals_flow*.json`): the editor
saves Compare `b` typed next to a wired number as the JSON number; a flow saved by Flow 0.13.0
stored the text. The typed flow compares 5 == 5 and 10 > 9; the 0.13.0 flow is refused with a
run error that names the node and says how to fix it, never answered with a silent False.
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
FIX_HINT = "Open the flow in the editor and save it to store typed values."


def _run(name: str):
    spec = compile_visualflow(json.loads((FIXTURES / name).read_text(encoding="utf-8")))
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    run_id = runtime.start(workflow=spec, vars={})
    return runtime.tick(workflow=spec, run_id=run_id)


def test_typed_flow_compares_numbers() -> None:
    state = _run("typed_literals_flow.json")
    assert state.status == RunStatus.COMPLETED
    assert state.output == {"same": True, "above": True, "success": True}


def test_flow_saved_with_text_is_refused_loudly_naming_the_node() -> None:
    state = _run("typed_literals_flow.legacy.json")
    out: Dict[str, Any] = dict(state.output or {})
    assert out.get("success") is False
    error = str(out.get("error") or "")
    assert "Compare 'Is five' (node same) ==: a is a number (5) and b is text ('5')." in error
    assert "One side is a number and the other is text." in error
    assert error.endswith(FIX_HINT)
    assert "same" not in out  # the run stopped at the refusal: no silent False was produced


@pytest.mark.parametrize("op", ["==", "!=", "<", "<=", ">", ">="])
def test_number_against_text_is_refused_for_every_operator(op: str) -> None:
    for a, b in ((5, "5"), ("9", 10), (2.5, "x")):
        with pytest.raises(ValueError, match="One side is a number and the other is text"):
            control_compare({"a": a, "b": b, "op": op})


def test_true_false_against_text_is_refused() -> None:
    with pytest.raises(ValueError, match="One side is true/false and the other is text"):
        control_compare({"a": True, "b": "true"})


@pytest.mark.parametrize(
    "a,b,op,expected",
    [
        (5, 5, "==", True),
        (5, 5.0, "==", True),
        ("5", "5", "==", True),
        ("abc", "abd", "<", True),
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
