"""Trigger envelopes in CloudEvents 1.0 names (framework backlog 1039).

`TriggerEnvelope` keeps its original keys (`event_id`, `source_id`, `source_version`, `fired_at`,
`payload`, `binding_id`) and gains, additively, the CloudEvents 1.0 attributes and this framework's
extensions, populated by every built-in source's `normalize`:

    specversion "1.0" | id (= event_id) | type "abstract.<source_id>.v<version>" | source (a URI: the
    connection, the mailbox, or the binding) | time (= fired_at) | subject | datacontenttype
    "application/json" | dataschema (optional)

    extensions: bindingid | connectionid | correlationid | dedupekey | sequence (the inbox seq)
                | recordedtime | expirytime | traceparent | abstracttrust ("trusted" | "untrusted")

`to_cloudevent(envelope)` returns the structured CloudEvent (`data` = the envelope's `payload`),
from a new envelope or from one recorded before 1039 (derived from the original keys);
`from_cloudevent(event)` is the inverse. An occurrence receives `to_cloudevent(envelope)` as
`input_data["event"]` beside its prompt header.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional

SPECVERSION = "1.0"
DATACONTENTTYPE = "application/json"
TYPE_PREFIX = "abstract."
CORE_ATTRIBUTES = ("specversion", "id", "type", "source", "time", "subject", "datacontenttype", "dataschema")
EXTENSIONS = (
    "bindingid", "connectionid", "correlationid", "dedupekey", "sequence", "recordedtime", "expirytime",
    "traceparent", "abstracttrust",
)
TRUST_VALUES = ("trusted", "untrusted")
ENVELOPE_KEYS = ("event_id", "source_id", "source_version", "fired_at", "payload", "binding_id")


def event_type(source_id: str, source_version: int) -> str:
    """CloudEvents `type`: `abstract.<source_id>.v<version>` (e.g. `abstract.email.received.v1`)."""
    return f"{TYPE_PREFIX}{source_id}.v{int(source_version)}"


def parse_event_type(value: str) -> tuple:
    """`abstract.<source_id>.v<version>` -> (source_id, version); ValueError otherwise."""
    if not isinstance(value, str) or not value.startswith(TYPE_PREFIX) or ".v" not in value:
        raise ValueError(f"not an abstract trigger event type: {value!r}")
    body, _, version = value[len(TYPE_PREFIX):].rpartition(".v")
    if not body or not version.isdigit():
        raise ValueError(f"not an abstract trigger event type: {value!r}")
    return body, int(version)


def binding_source_uri(source_id: str, binding_id: str) -> str:
    """CloudEvents `source` of an event no connection produced (a schedule tick, a manual run)."""
    return f"abstract://triggers/{source_id}/bindings/{binding_id}"


def with_cloudevent_attributes(
    envelope: Dict[str, Any],
    *,
    subject: Optional[str],
    dedupekey: Optional[str],
    trust: str,
    source: Optional[str] = None,
    connectionid: Optional[str] = None,
    sequence: Optional[int] = None,
    correlationid: Optional[str] = None,
    expirytime: Optional[str] = None,
    traceparent: Optional[str] = None,
    dataschema: Optional[str] = None,
) -> Dict[str, Any]:
    """Add the CloudEvents attributes to an envelope built with the original keys (in place)."""
    if trust not in TRUST_VALUES:
        raise ValueError(f"abstracttrust must be one of {list(TRUST_VALUES)}")
    envelope.update(
        {
            "specversion": SPECVERSION,
            "id": str(envelope["event_id"]),
            "type": event_type(envelope["source_id"], envelope["source_version"]),
            "source": source or binding_source_uri(str(envelope["source_id"]), str(envelope["binding_id"])),
            "time": str(envelope["fired_at"]),
            "datacontenttype": DATACONTENTTYPE,
            "bindingid": str(envelope["binding_id"]),
            "recordedtime": str(envelope["fired_at"]),
            "abstracttrust": trust,
        }
    )
    for key, value in (
        ("subject", subject),
        ("dedupekey", dedupekey),
        ("connectionid", connectionid),
        ("correlationid", correlationid),
        ("expirytime", expirytime),
        ("traceparent", traceparent),
        ("dataschema", dataschema),
    ):
        if value is not None:
            envelope[key] = str(value)
    if sequence is not None:
        envelope["sequence"] = int(sequence)
    return envelope


def ensure_cloudevent_attributes(envelope: Dict[str, Any], *, trust: str) -> Dict[str, Any]:
    """Add the CloudEvents attributes to an envelope a source normalized without them (schedule@1,
    frozen byte for byte; a third-party source): `subject` = `tick:<n>` / `command:<id>` / the event
    id, `dedupekey` = the event id. An envelope that has them is returned unchanged."""
    if envelope.get("specversion"):
        return envelope
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), Mapping) else {}
    if payload.get("tick") is not None:
        subject = f"tick:{payload['tick']}"
    elif payload.get("command_id"):
        subject = f"command:{payload['command_id']}"
    else:
        subject = str(envelope["event_id"])
    return with_cloudevent_attributes(envelope, subject=subject, dedupekey=str(envelope["event_id"]), trust=trust)


def to_cloudevent(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """The structured CloudEvent of a trigger envelope (`data` = its payload)."""
    out: Dict[str, Any] = {
        "specversion": SPECVERSION,
        "id": str(envelope.get("id") or envelope["event_id"]),
        "type": str(envelope.get("type") or event_type(envelope["source_id"], envelope["source_version"])),
        "source": str(envelope.get("source") or binding_source_uri(str(envelope["source_id"]), str(envelope["binding_id"]))),
        "time": str(envelope.get("time") or envelope["fired_at"]),
        "datacontenttype": str(envelope.get("datacontenttype") or DATACONTENTTYPE),
    }
    for key in ("subject", "dataschema"):
        if envelope.get(key) is not None:
            out[key] = envelope[key]
    out["bindingid"] = str(envelope.get("bindingid") or envelope["binding_id"])
    for key in EXTENSIONS:
        if key != "bindingid" and envelope.get(key) is not None:
            out[key] = envelope[key]
    out["data"] = copy.deepcopy(dict(envelope.get("payload") or {}))
    return out


def from_cloudevent(event: Mapping[str, Any]) -> Dict[str, Any]:
    """The trigger envelope (original keys + CloudEvents attributes) of a structured CloudEvent."""
    if event.get("specversion") != SPECVERSION:
        raise ValueError("not a CloudEvents 1.0 event (specversion)")
    for key in ("id", "type", "source"):
        if not isinstance(event.get(key), str) or not event[key]:
            raise ValueError(f"a CloudEvent needs {key}")
    source_id, version = parse_event_type(event["type"])
    envelope: Dict[str, Any] = {
        "event_id": event["id"],
        "source_id": source_id,
        "source_version": version,
        "fired_at": str(event.get("time") or event.get("recordedtime") or ""),
        "payload": copy.deepcopy(dict(event.get("data") or {})),
        "binding_id": str(event.get("bindingid") or ""),
    }
    for key in CORE_ATTRIBUTES + EXTENSIONS:
        if key in event:
            envelope[key] = copy.deepcopy(event[key])
    return envelope


# The CloudEvents 1.0 JSON format, as a JSON schema (core attributes and extension naming).
CLOUDEVENT_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["specversion", "id", "source", "type"],
    "properties": {
        "specversion": {"type": "string", "const": "1.0"},
        "id": {"type": "string", "minLength": 1},
        "source": {"type": "string", "format": "uri-reference", "minLength": 1},
        "type": {"type": "string", "minLength": 1},
        "datacontenttype": {"type": "string"},
        "dataschema": {"type": "string", "format": "uri", "minLength": 1},
        "subject": {"type": "string", "minLength": 1},
        "time": {"type": "string", "format": "date-time"},
        "data": {},
        "data_base64": {"type": "string"},
    },
    "propertyNames": {"pattern": "^([a-z0-9]{1,20}|data_base64)$"},
}


__all__ = [
    "CLOUDEVENT_JSON_SCHEMA",
    "EXTENSIONS",
    "binding_source_uri",
    "ensure_cloudevent_attributes",
    "event_type",
    "from_cloudevent",
    "parse_event_type",
    "to_cloudevent",
    "with_cloudevent_attributes",
]
