from __future__ import annotations

from abstractruntime.rendering import stringify_json


def test_stringify_json_modes() -> None:
    value = {"a": 1, "b": [2, 3]}

    beautify = stringify_json(value, mode="beautify")
    assert beautify.lstrip().startswith("{")
    assert "\n" in beautify
    assert '"a": 1' in beautify

    none = stringify_json(value, mode="none")
    assert none == '{"a": 1, "b": [2, 3]}'

    minified = stringify_json(value, mode="minified")
    assert minified == '{"a":1,"b":[2,3]}'


def test_stringify_json_best_effort_parses_jsonish_strings() -> None:
    raw = "```json\n{'a': 1}\n```"
    out = stringify_json(raw, mode="beautify")
    assert out.lstrip().startswith("{")
    # We should not emit the JSON *string* of the input; we should parse and render an object.
    assert '"a": 1' in out


