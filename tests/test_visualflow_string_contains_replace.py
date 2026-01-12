from __future__ import annotations

from abstractruntime.visualflow_compiler.visual.builtins import get_builtin_handler


def test_string_contains_builtin() -> None:
    contains = get_builtin_handler("contains")
    assert contains is not None

    assert contains({"text": "hello world", "pattern": "world"}) is True
    assert contains({"text": "hello world", "pattern": "WORLD"}) is False

    # Safer than Python's default behavior where "" is contained in every string.
    assert contains({"text": "hello", "pattern": ""}) is False
    assert contains({"text": "hello", "pattern": None}) is False
    assert contains({"text": None, "pattern": "x"}) is False


def test_string_replace_builtin_modes() -> None:
    replace = get_builtin_handler("replace")
    assert replace is not None

    assert replace({"text": "a-b-c", "pattern": "-", "replacement": "_", "mode": "all"}) == "a_b_c"
    assert replace({"text": "a-b-c", "pattern": "-", "replacement": "_", "mode": "first"}) == "a_b-c"
    assert replace({"text": "a-b-c", "pattern": "-", "replacement": "_", "mode": None}) == "a_b_c"

    # Empty/missing pattern is a no-op.
    assert replace({"text": "a-b-c", "pattern": "", "replacement": "_", "mode": "all"}) == "a-b-c"
    assert replace({"text": "a-b-c", "pattern": None, "replacement": "_", "mode": "all"}) == "a-b-c"

    # Numeric best-effort support: replace first N occurrences.
    assert replace({"text": "a-b-c-d", "pattern": "-", "replacement": "_", "mode": "2"}) == "a_b_c-d"

