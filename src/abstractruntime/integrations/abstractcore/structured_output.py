"""Structured-output schema values shared by Runtime's effect and execution clients.

Schemas remain JSON data across client/process boundaries. Only the execution
client creates the Pydantic class required by AbstractCore's Python API.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple, Type


class StructuredOutputRequestError(ValueError):
    """A deterministic structured-request refusal, classified without retries."""

    status_code = 400


def pop_response_schema(params: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], str]]:
    """Remove a schema request from a private per-call parameter dictionary."""
    schema = params.pop("response_schema", None)
    name = params.pop("response_schema_name", None)
    if schema is None:
        return None
    if not isinstance(schema, dict) or not schema:
        raise StructuredOutputRequestError("response_schema must be a non-empty JSON schema object")
    if params.get("response_model") is not None or params.get("response_format") is not None:
        raise StructuredOutputRequestError("response_schema cannot be combined with response_model or response_format")
    if name is None:
        name = "StructuredOutput"
    if not isinstance(name, str) or not name.strip():
        raise StructuredOutputRequestError("response_schema_name must be a non-empty string")
    return schema, name.strip()


def response_model_from_json_schema(schema: Dict[str, Any], *, name: str) -> Type[Any]:
    """Best-effort conversion from a JSON schema dict to a Pydantic model.

    This exists so structured output requests can remain JSON-safe in durable
    effect payloads (we persist the schema, not the Python class).
    """
    try:
        from pydantic import BaseModel, Field, create_model
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Pydantic is required for structured outputs: {e}")

    from typing import Literal, Union

    NoneType = type(None)

    def _python_type(sub_schema: Any, *, nested_name: str) -> Any:
        if not isinstance(sub_schema, dict):
            return Any
        # Enums: represent as Literal[...] so Pydantic can enforce allowed values.
        enum_raw = sub_schema.get("enum")
        if isinstance(enum_raw, list) and enum_raw:
            try:
                return Literal.__getitem__(tuple(enum_raw))  # type: ignore[attr-defined]
            except Exception:
                return Any

        t = sub_schema.get("type")
        if isinstance(t, list) and t:
            # Union types (e.g. ["string","null"]).
            variants: list[Any] = []
            for tt in t:
                if tt == "null":
                    variants.append(NoneType)
                    continue
                if isinstance(tt, str) and tt:
                    variants.append(_python_type(dict(sub_schema, type=tt), nested_name=nested_name))
            # Drop Any from unions to avoid masking concrete variants.
            variants2 = [v for v in variants if v is not Any]
            variants = variants2 or variants
            if not variants:
                return Any
            if len(variants) == 1:
                return variants[0]
            try:
                return Union.__getitem__(tuple(variants))  # type: ignore[attr-defined]
            except Exception:
                return Any
        if t == "string":
            return str
        if t == "integer":
            return int
        if t == "number":
            return float
        if t == "boolean":
            return bool
        if t == "array":
            items = sub_schema.get("items")
            return list[_python_type(items, nested_name=f"{nested_name}Item")]  # type: ignore[index]
        if t == "object":
            props = sub_schema.get("properties")
            if isinstance(props, dict) and props:
                return _model(sub_schema, name=nested_name)
            return Dict[str, Any]
        return Any

    def _model(obj_schema: Dict[str, Any], *, name: str) -> Type[BaseModel]:
        schema_type = obj_schema.get("type")
        if schema_type is None and isinstance(obj_schema.get("properties"), dict):
            schema_type = "object"
        if isinstance(schema_type, list) and "object" in schema_type:
            schema_type = "object"
        if schema_type != "object":
            raise ValueError("response_schema must be a JSON schema object")
        props = obj_schema.get("properties")
        if not isinstance(props, dict) or not props:
            raise ValueError("response_schema must define properties")
        required_raw = obj_schema.get("required")
        required: Set[str] = set()
        if isinstance(required_raw, list):
            required = {str(x) for x in required_raw if isinstance(x, str)}

        fields: Dict[str, Tuple[Any, Any]] = {}
        for prop_name, prop_schema in props.items():
            if not isinstance(prop_name, str) or not prop_name.strip():
                continue
            # Keep things simple: only support identifier-like names to avoid aliasing issues.
            if not prop_name.isidentifier():
                raise ValueError(
                    f"Invalid property name '{prop_name}'. Use identifier-style names (letters, digits, underscore)."
                )
            t = _python_type(prop_schema, nested_name=f"{name}_{prop_name}")
            description = None
            default_value = None
            has_default = False
            if isinstance(prop_schema, dict):
                raw_description = prop_schema.get("description")
                if isinstance(raw_description, str) and raw_description.strip():
                    description = raw_description.strip()
                if "default" in prop_schema:
                    default_value = prop_schema.get("default")
                    has_default = True

            if prop_name in required:
                fields[prop_name] = (t, Field(..., description=description))
            else:
                field_default = default_value if has_default else None
                fields[prop_name] = (Optional[t], Field(field_default, description=description))

        return create_model(name, **fields)  # type: ignore[call-arg]

    return _model(schema, name=name)
