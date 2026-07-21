"""Code-node sandbox under RestrictedPython (regression pins, 2026-07-17).

THE INCIDENT: the visual compiler generates ``def transform(_input):`` for
every codeBody node, but RestrictedPython's default policy refuses ANY
leading-underscore name. The custom allow-single-underscore policy existed
in the original abstractflow executor (2026-02-20) and was lost when the
compiler moved into abstractruntime — invisibly, because RestrictedPython
was not installed here and the ImportError fallback ran the basic handler.
The moment another package's work installed RestrictedPython 8.4, every
code node "completed" its run while silently failing (7 unrelated-looking
test failures: KG ingest empty, event-inbox dead, file nodes writing
nothing).

These pins run only when RestrictedPython is importable — which is exactly
the configuration the incident proved must stay green.
"""

from __future__ import annotations

import pytest

pytest.importorskip("RestrictedPython")

from abstractruntime.visualflow_compiler.visual.code_executor import (  # noqa: E402
    CodeExecutionError,
    create_code_handler,
)


def test_generated_wrapper_shape_compiles_and_runs() -> None:
    """The exact shape _generate_code_from_body emits must work."""
    handler = create_code_handler(
        'def transform(_input):\n    content = _input.get("content")\n    return str(content)'
    )
    assert handler({"content": 42}) == "42"


def test_private_helper_names_are_allowed() -> None:
    code = (
        "def _double(x):\n"
        "    return x * 2\n"
        "\n"
        "def transform(_input):\n"
        "    _tmp = _double(3)\n"
        "    return _tmp\n"
    )
    assert create_code_handler(code)({}) == 6


def test_realistic_flow_code_semantics_hold() -> None:
    """AugAssign on names, tuple unpack, comprehensions, starred calls,
    attribute reads, subscript writes — the everyday code-node diet must
    behave identically to plain Python under the guard set."""
    code = (
        "def transform(_input):\n"
        "    a, b = (1, 2)\n"
        "    total = 0\n"
        "    for i in range(4):\n"
        "        total += i\n"
        "    parts = [str(p) for p in [a, b, total]]\n"
        "    d = {}\n"
        "    d['joined'] = '-'.join(parts)\n"
        "    args = [7, 3]\n"
        "    d['m'] = max(*args)\n"
        "    return d\n"
    )
    assert create_code_handler(code)({}) == {"joined": "1-2-6", "m": 7}


def test_guard_names_are_reserved() -> None:
    with pytest.raises(CodeExecutionError, match="reserved sandbox name"):
        create_code_handler("def transform(_input):\n    _getattr_ = None\n    return 1")


def test_dunder_names_stay_refused() -> None:
    with pytest.raises(CodeExecutionError):
        create_code_handler("def transform(_input):\n    return (1).__class__")
    with pytest.raises(CodeExecutionError):
        create_code_handler("def transform(_input):\n    __x__ = 1\n    return __x__")


def test_runtime_guards_are_bound() -> None:
    """RP 8.4 emits _write_/_inplacevar_/_apply_/_getattr_ call sites; a
    missing guard binding is a NameError at run time, not compile time —
    so exercise each transform once."""
    code = (
        "def transform(_input):\n"
        "    d = {'n': 0}\n"
        "    d['n'] = 5\n"
        "    x = 1\n"
        "    x += 2\n"
        "    vals = [x]\n"
        "    return sum(*[vals]) + d['n']\n"
    )
    assert create_code_handler(code)({}) == 8
