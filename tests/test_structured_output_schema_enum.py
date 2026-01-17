from __future__ import annotations

import pytest


def test_pydantic_model_from_json_schema_enforces_enum_literals() -> None:
    """Level A: JSON Schema enums must be enforced for structured outputs."""

    from abstractruntime.integrations.abstractcore.effect_handlers import (  # type: ignore
        _pydantic_model_from_json_schema,
    )

    schema = {
        "type": "object",
        "properties": {
            "predicate": {"type": "string", "enum": ["a", "b"]},
            "maybe_number": {"type": ["number", "null"]},
            "items": {
                "type": "array",
                "items": {"type": "string", "enum": ["x", "y"]},
            },
        },
        "required": ["predicate", "items"],
    }

    Model = _pydantic_model_from_json_schema(schema, name="TestSchema")

    ok = Model(predicate="a", maybe_number=None, items=["x"])
    assert ok.predicate == "a"
    assert ok.maybe_number is None
    assert ok.items == ["x"]

    with pytest.raises(Exception):
        Model(predicate="c", maybe_number=None, items=["x"])

    with pytest.raises(Exception):
        Model(predicate="a", maybe_number=None, items=["z"])

