"""The structured-output facade hands hosts AbstractCore's own checker and error."""
import pytest

from abstractcore.structured import json_schema as core_json_schema

from abstractruntime.integrations import abstractcore as integration
from abstractruntime.integrations.abstractcore import structured_facade
from abstractruntime.integrations.abstractcore.structured_facade import (
    ResponseFormatError,
    parse_response_format,
)

pytestmark = pytest.mark.basic


def test_error_class_is_cores_own_and_flat_exports_match():
    assert ResponseFormatError is core_json_schema.ResponseFormatError
    assert integration.ResponseFormatError is core_json_schema.ResponseFormatError
    assert integration.parse_response_format is structured_facade.parse_response_format


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {"type": "text"},
        {"type": "json_object"},
        {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}},
    ],
)
def test_valid_formats_answer_exactly_what_core_answers(raw):
    assert parse_response_format(raw) == core_json_schema.parse_response_format(raw)


@pytest.mark.parametrize(
    "raw, param",
    [
        ("json", "response_format"),
        ({"type": "xml"}, "response_format.type"),
    ],
)
def test_bad_formats_raise_cores_error_with_the_same_message_and_param(raw, param):
    with pytest.raises(core_json_schema.ResponseFormatError) as core_err:
        core_json_schema.parse_response_format(raw)
    with pytest.raises(ResponseFormatError) as err:
        parse_response_format(raw)
    assert type(err.value) is type(core_err.value)
    assert str(err.value) == str(core_err.value)
    assert err.value.param == core_err.value.param == param


def test_unknown_attribute_is_an_attribute_error():
    with pytest.raises(AttributeError):
        structured_facade.NoSuchThing  # noqa: B018
