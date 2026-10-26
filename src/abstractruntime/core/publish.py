"""Publish to channel (`EffectType.PUBLISH_EVENT`): a flow tells applications something happened.

Payload (a "Publish to channel" node, or any workflow):

    {channel: str, type?: str, title: str, body?: str, subject?: str, data?: object, alert?: bool = false,
     recipient?: str = "me", channels?: [tray | watch | phone | email | console]}

- `channel` is a free name chosen by the author ("kitchen", "ops", an application's id): applications
  subscribe to the user's event stream filtered by channel. Letters, digits, `.`, `_`, `-`; 1..64.
- `type` is the event's CloudEvents type ("kitchen.order.ready"); the host defaults it to
  `<channel>.message`. 1..200 characters, no spaces.
- `alert: true` also raises an attention item, so the tray, the Assistant and the watch show it.
- `recipient` is whom the alert reaches: `"me"` (the run's owner, the default). The shape is a name so
  another account can be named later; a host that cannot reach it refuses the payload (ValueError).
- `channels` is where the alert reaches the recipient: a non-empty list from `tray`, `watch`, `phone`,
  `email`, `console` (deduplicated, kept in that order); absent (or `"all"`) = every channel. It needs
  `alert: true` (it chooses where an alert rings; an event without an alert rings nowhere).

Defaults are omitted from the normalized payload (`recipient` "me", no `channels`, `alert` false), so a
host written before these fields still accepts every payload that does not use them.

The runtime validates the payload and hands it to the HOST, which appends the CloudEvents event to the
user's stream (extension `channel`, `data = {...data, title, body?}`) and the attention item:
`Runtime.set_publish_event_handler(handler)`, called as `handler(run, payload, step_id)`. `step_id` is the
effect's idempotency key, the same on a replayed step, so the host appends the event once (its id is
`publish:<run_id>:<step_id>`). The handler returns `{event_id, sequence, alerted}`, which is the
effect's result; a payload the host refuses raises ValueError (the step fails with that sentence).
Without a handler the effect fails loudly.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional

CHANNEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
TYPE_MAX = 200
TITLE_MAX = 120
BODY_MAX = 2000
DATA_MAX_BYTES = 64 * 1024
_KEYS = ("channel", "type", "title", "body", "subject", "data", "alert", "recipient", "channels")
# Where an alert can reach its recipient (Notify me's "Where"), in their canonical order.
DELIVERY_CHANNELS = ("tray", "watch", "phone", "email", "console")
DEFAULT_RECIPIENT = "me"
RECIPIENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@:+-]{0,127}$")


class PublishError(ValueError):
    def __init__(self, message: str, *, field: Optional[str] = None) -> None:
        super().__init__(message)
        self.field = field


def validate_publish(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """The normalized payload (unset optional fields omitted); raises PublishError."""
    if not isinstance(payload, Mapping):
        raise PublishError("publish payload must be an object")
    unknown = sorted(k for k in payload if k not in _KEYS)
    if unknown:
        raise PublishError(f"unknown publish field(s): {unknown}", field=unknown[0])
    channel = payload.get("channel")
    if not isinstance(channel, str) or not CHANNEL_RE.match(channel.strip()):
        raise PublishError("channel must be a name of letters, digits, '.', '_' or '-' (1..64), e.g. kitchen", field="channel")
    out: Dict[str, Any] = {"channel": channel.strip()}
    etype = payload.get("type")
    if etype is not None and etype != "":
        if not isinstance(etype, str) or len(etype) > TYPE_MAX or any(c.isspace() for c in etype.strip()) or not etype.strip():
            raise PublishError(f"type must be a name without spaces (1..{TYPE_MAX}), e.g. kitchen.order.ready", field="type")
        out["type"] = etype.strip()
    title = payload.get("title")
    if not isinstance(title, str) or not title.strip() or len(title) > TITLE_MAX:
        raise PublishError(f"title is required (1..{TITLE_MAX} characters)", field="title")
    out["title"] = title.strip()
    for key, limit in (("body", BODY_MAX), ("subject", TITLE_MAX)):
        value = payload.get(key)
        if value is None or value == "":
            continue
        if not isinstance(value, str) or len(value) > limit:
            raise PublishError(f"{key} must be text of at most {limit} characters", field=key)
        out[key] = value
    data = payload.get("data")
    if data is not None:
        if not isinstance(data, Mapping):
            raise PublishError("data must be an object", field="data")
        encoded = json.dumps(data, default=str)
        if len(encoded.encode("utf-8")) > DATA_MAX_BYTES:
            raise PublishError(f"data must be at most {DATA_MAX_BYTES} bytes of JSON", field="data")
        out["data"] = json.loads(encoded)
    alert = payload.get("alert", False)
    if not isinstance(alert, bool):
        raise PublishError("alert must be true or false", field="alert")
    if alert:
        out["alert"] = True
    recipient = payload.get("recipient")
    if recipient is not None and recipient != "":
        if not isinstance(recipient, str) or not RECIPIENT_RE.match(recipient.strip()):
            raise PublishError('recipient must be "me" (later: an account name)', field="recipient")
        if recipient.strip() != DEFAULT_RECIPIENT:
            out["recipient"] = recipient.strip()
    channels = normalize_channels(payload.get("channels"))
    if channels is not None:
        if not alert:
            raise PublishError("channels choose where the alert reaches you: set alert on, or leave channels out", field="channels")
        out["channels"] = channels
    return out


# What the visual compiler adds around a node's payload: the effect type it was baked from (under
# `type`, so a node's own `type` pin travels as `event_type`) and the node to resume at.
VISUAL_ENVELOPE_KEYS = ("resume_to_node",)
VISUAL_EFFECT_TYPE = "publish_event"


def strip_visual_envelope(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """The publish fields of an effect payload built by the visual compiler (a Notify me or
    Publish to channel node): `resume_to_node` dropped, `type: "publish_event"` (the effect type,
    not the event's) dropped, `event_type` (the node's `type` pin) restored as `type`. A payload
    built by hand passes through unchanged."""
    out = {k: v for k, v in dict(payload or {}).items() if k not in VISUAL_ENVELOPE_KEYS}
    if out.get("type") == VISUAL_EFFECT_TYPE:
        out.pop("type")
    if "event_type" in out:
        event_type = out.pop("event_type")
        if event_type is not None and event_type != "":
            out["type"] = event_type
    return out


def normalize_channels(value: Any) -> Optional[List[str]]:
    """The channel set of an alert: None for "every channel" (absent, None, "all"), else the listed
    channels deduplicated in `DELIVERY_CHANNELS` order. Raises PublishError on anything else."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "all")):
        return None
    if not isinstance(value, (list, tuple)):
        raise PublishError(f"channels must be a list of {', '.join(DELIVERY_CHANNELS)} (or left out for all)", field="channels")
    asked = set()
    for item in value:
        name = item.strip().lower() if isinstance(item, str) else None
        if name not in DELIVERY_CHANNELS:
            raise PublishError(f"unknown channel {item!r}: choose from {', '.join(DELIVERY_CHANNELS)}", field="channels")
        asked.add(name)
    if not asked:
        raise PublishError("channels must name at least one channel (or be left out for all)", field="channels")
    return [c for c in DELIVERY_CHANNELS if c in asked]


__all__ = ["CHANNEL_RE", "DEFAULT_RECIPIENT", "DELIVERY_CHANNELS", "PublishError", "normalize_channels", "strip_visual_envelope", "validate_publish"]
