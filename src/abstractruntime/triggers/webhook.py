"""`webhook.received@1`: run an automation when an external system calls a signed webhook
(framework backlog 1042, the runtime half; the HTTP route is the gateway's).

A webhook connection is an endpoint with a signing secret. Callers sign per **Standard Webhooks**
(https://www.standardwebhooks.com): three headers,

    webhook-id:        a unique message id (a retry reuses it)
    webhook-timestamp: integer seconds since the epoch
    webhook-signature: space-separated "v1,<base64 HMAC-SHA256>" entries

over the content `"<webhook-id>.<webhook-timestamp>.<raw body>"`, keyed with the base64-decoded secret
(`whsec_<base64>`; the connection generates one, sealed with the plane's sealing key).

- `verify_standard_webhook(secret, headers, body, now, tolerance_s)`: pure; a typed verdict;
- `append_webhook_event(inbox, connection, body, headers, ...)`: verify, then append ONE event to the
  connection's stream `trigger:<connection_id>`; `webhook-id` is the event's `subject` and `dedupekey`, so a
  replayed id never appends twice. Returns `{status: appended|duplicate|refused, http_status, reason?, ...}`
  for the gateway to answer with (202 / 409 / 401 / 413 / 415);
- the source (`WebhookReceivedTriggerAdapter`, pure, `ConnectionEventAdapter`): every new event is admitted
  promptly (no batching interval); events that arrive together become one occurrence.

Connection config: `{subject_path?: "a.b.c" (dotted path in a JSON body for the event subject; default the
webhook id), tolerance_s?: 300 (60..3600), secret?: "whsec_..." (supply your own; generated otherwise)}`.

Each event's `data`: `{webhook_id, content_type, body}` where `body` is the parsed JSON (any JSON value), or
`{webhook_id, content_type, body_text}` for a UTF-8 non-JSON body. Written by the caller: untrusted.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets as _secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from .connection_events import ConnectionEventAdapter
from .connections import TriggerConnection, append_connection_event
from .protocol import TriggerConfigError, format_timestamp

SOURCE_ID = "webhook.received"
SOURCE_VERSION = 1
SOURCE_REF = f"{SOURCE_ID}@{SOURCE_VERSION}"

SECRET_PREFIX = "whsec_"
SECRET_BYTES = 24
DEFAULT_TOLERANCE_S = 300
MIN_TOLERANCE_S = 60
MAX_TOLERANCE_S = 3600
DEFAULT_MAX_BODY_BYTES = 1024 * 1024
MAX_WEBHOOK_ID = 256
MAX_SUBJECT_PATH = 200


def new_webhook_secret() -> str:
    """A Standard Webhooks secret: `whsec_` + base64 of 24 random bytes."""
    return SECRET_PREFIX + base64.b64encode(_secrets.token_bytes(SECRET_BYTES)).decode("ascii")


def _secret_key(secret: str) -> bytes:
    raw = secret[len(SECRET_PREFIX):] if secret.startswith(SECRET_PREFIX) else secret
    try:
        key = base64.b64decode(raw, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("the webhook secret is not base64 (whsec_<base64>)") from exc
    if len(key) < 16:
        raise ValueError("the webhook secret must decode to at least 16 bytes")
    return key


def sign_standard_webhook(secret: str, *, webhook_id: str, timestamp: int, body: bytes) -> str:
    """The `webhook-signature` value (`v1,<base64>`) a sender puts on a request (tests, examples)."""
    content = f"{webhook_id}.{int(timestamp)}.".encode("utf-8") + bytes(body)
    digest = hmac.new(_secret_key(secret), content, hashlib.sha256).digest()
    return "v1," + base64.b64encode(digest).decode("ascii")


# --- the source -----------------------------------------------------------------------------------


class WebhookReceivedTriggerAdapter(ConnectionEventAdapter):
    descriptor = {
        "id": SOURCE_ID,
        "version": SOURCE_VERSION,
        "label": "When a webhook is called",
        "config_schema": ConnectionEventAdapter.binding_config_schema(),
        "event_schema": ConnectionEventAdapter.batch_event_schema(),
        "capabilities": {"kind": "event", "inbox": "connection", "content_trust": "untrusted"},
        "needs_connection": True,
        "connection_provider": "trigger_connections",
        "connection_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "subject_path": {"type": "string", "description": "dotted path in a JSON body for the event subject (default: the webhook id)"},
                "tolerance_s": {"type": "integer", "minimum": MIN_TOLERANCE_S, "maximum": MAX_TOLERANCE_S, "default": DEFAULT_TOLERANCE_S},
                "secret": {"type": "string", "x-secret": True, "description": "whsec_<base64>; generated when omitted"},
            },
        },
    }

    def occurrence_inputs(self, *, payload: Mapping[str, Any], inputs: Optional[Mapping[str, Any]] = None,
                          index: int, mailbox: Optional[str] = None) -> Dict[str, Any]:
        """Named occurrence inputs (the Flow contract): `calls`, each `{id, received_at, headers, body}`
        (`body`: the parsed JSON, or the text of a non-JSON body)."""
        calls = []
        for ev in payload.get("events") or []:
            if not isinstance(ev, Mapping):
                continue
            data = ev.get("data") if isinstance(ev.get("data"), Mapping) else {}
            is_call = "webhook_id" in data
            calls.append({
                "id": data.get("webhook_id") if is_call else ev.get("subject"),
                "received_at": ev.get("time"),
                "headers": dict(data.get("headers") or {}) if is_call else {},
                "body": (data.get("body") if "body" in data else data.get("body_text")) if is_call else dict(data),
            })
        return {"calls": calls}

    def validate_connection(self, config: Mapping[str, Any], *, now: str) -> Dict[str, Any]:
        unknown = sorted(k for k in config if k not in ("subject_path", "tolerance_s", "secret"))
        if unknown:
            raise TriggerConfigError(f"unknown webhook field(s): {unknown}", field=f"config.{unknown[0]}")
        out: Dict[str, Any] = {}
        path = config.get("subject_path")
        if path is not None:
            if (not isinstance(path, str) or not path.strip() or len(path) > MAX_SUBJECT_PATH
                    or any(not part for part in path.strip().split("."))):
                raise TriggerConfigError("subject_path must be a dotted path like data.id", field="config.subject_path")
            out["subject_path"] = path.strip()
        tolerance = config.get("tolerance_s", DEFAULT_TOLERANCE_S)
        if isinstance(tolerance, bool) or not isinstance(tolerance, int) or not MIN_TOLERANCE_S <= tolerance <= MAX_TOLERANCE_S:
            raise TriggerConfigError(f"tolerance_s must be an integer {MIN_TOLERANCE_S}..{MAX_TOLERANCE_S}", field="config.tolerance_s")
        out["tolerance_s"] = tolerance
        if "secret" in config:
            secret = config["secret"]
            try:
                if not isinstance(secret, str):
                    raise ValueError("not a string")
                _secret_key(secret)
            except ValueError as exc:
                raise TriggerConfigError(f"secret must be whsec_<base64 of at least 16 bytes> ({exc})", field="config.secret") from exc
            out["secret"] = secret
        return out

    def connection_secrets(self, config: Mapping[str, Any]) -> Dict[str, Any]:
        """A signing secret when the creator supplied none (sealed; shown to the owner once)."""
        return {} if config.get("secret") else {"secret": new_webhook_secret()}


# --- verification (pure) ----------------------------------------------------------------------------


@dataclass(frozen=True)
class WebhookVerification:
    ok: bool
    reason: Optional[str] = None  # missing_headers | bad_timestamp | stale | bad_signature
    webhook_id: Optional[str] = None
    timestamp: Optional[int] = None

    @property
    def http_status(self) -> int:
        return 200 if self.ok else 401


def _header(headers: Mapping[str, Any], name: str) -> Optional[str]:
    for key, value in (headers or {}).items():
        if str(key).lower() == name:
            return str(value).strip() if value is not None else None
    return None


def verify_standard_webhook(
    secret: str,
    headers: Mapping[str, Any],
    body: bytes,
    now: Any,
    tolerance_s: int = DEFAULT_TOLERANCE_S,
) -> WebhookVerification:
    """Check a Standard Webhooks signature. `now`: epoch seconds, an aware datetime or an RFC 3339
    string. Valid when ONE `v1` signature matches (constant-time) and the timestamp is within
    `tolerance_s` of `now` either way. Replays (same `webhook-id`) are the inbox's to refuse
    (`dedupekey`), not this function's."""
    webhook_id = _header(headers, "webhook-id")
    ts_raw = _header(headers, "webhook-timestamp")
    sig_raw = _header(headers, "webhook-signature")
    if not webhook_id or not ts_raw or not sig_raw or len(webhook_id) > MAX_WEBHOOK_ID:
        return WebhookVerification(False, "missing_headers", webhook_id=webhook_id)
    try:
        timestamp = int(ts_raw)
    except ValueError:
        return WebhookVerification(False, "bad_timestamp", webhook_id=webhook_id)
    if isinstance(now, datetime):
        now_s = now.timestamp()
    elif isinstance(now, str):
        now_s = datetime.fromisoformat(now.replace("Z", "+00:00")).timestamp()
    else:
        now_s = float(now)
    if abs(now_s - timestamp) > int(tolerance_s):
        return WebhookVerification(False, "stale", webhook_id=webhook_id, timestamp=timestamp)
    expected = sign_standard_webhook(secret, webhook_id=webhook_id, timestamp=timestamp, body=body).split(",", 1)[1]
    for entry in sig_raw.split(" "):
        version, _, value = entry.partition(",")
        if version == "v1" and value and hmac.compare_digest(value.encode("ascii", "ignore"), expected.encode("ascii")):
            return WebhookVerification(True, webhook_id=webhook_id, timestamp=timestamp)
    return WebhookVerification(False, "bad_signature", webhook_id=webhook_id, timestamp=timestamp)


# --- mapping an HTTP request to an event ----------------------------------------------------------


def _subject_from(body_value: Any, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    cur = body_value
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return None
    return str(cur) if isinstance(cur, (str, int, float)) and not isinstance(cur, bool) and str(cur) else None


# Never kept with a stored call: the signature and credentials a caller may send.
WITHHELD_HEADERS = frozenset({"webhook-signature", "authorization", "proxy-authorization", "cookie"})


def public_headers(headers: Mapping[str, Any]) -> Dict[str, str]:
    """The call's headers as stored (lower-case names), without `WITHHELD_HEADERS`."""
    items = headers.items() if hasattr(headers, "items") else []
    return {str(k).lower(): str(v) for k, v in items if str(k).lower() not in WITHHELD_HEADERS}


def webhook_event_data(webhook_id: str, body: bytes, content_type: Optional[str]) -> Dict[str, Any]:
    """`{webhook_id, content_type, body}` (JSON) or `{..., body_text}` (UTF-8 text); ValueError otherwise."""
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    try:
        text = bytes(body).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("the webhook body is not UTF-8 text") from exc
    if ctype in ("application/json", "") or ctype.endswith("+json"):
        try:
            return {"webhook_id": webhook_id, "content_type": ctype or "application/json", "body": json.loads(text) if text.strip() else None}
        except ValueError:
            if ctype:
                raise ValueError("the webhook body is not valid JSON")
    return {"webhook_id": webhook_id, "content_type": ctype or "text/plain", "body_text": text}


def append_webhook_event(
    inbox: Any,
    connection: TriggerConnection,
    body: bytes,
    headers: Mapping[str, Any],
    *,
    secret: Optional[str] = None,
    store: Any = None,
    now: Optional[Any] = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    runtime: Any = None,
) -> Dict[str, Any]:
    """Verify one webhook request and append it to the connection's stream.

    `secret`: the connection's signing secret (default: read from `store.secrets(...)`). Returns
    `{status, http_status, reason?, event_id?, seq?, webhook_id?}`: `appended` (202), `duplicate`
    (409, the `webhook-id` was already received), `refused` (401 `missing_headers` / `bad_timestamp` /
    `stale` / `bad_signature`; 413 `too_large`; 415 `unsupported_body`; 409 `connection_disabled` for a
    disabled or paused connection). A connection nothing uses yet (idle) still ACCEPTS and stores the
    call: the sender has delivered it, the person sees it among the connection's events, and a first
    test right after creating the connection works. `runtime` is accepted for callers written against
    0.11.0 and is not used.
    Refusals are recorded on the connection (`last_error`) when `store` is given."""
    if connection.source_id != SOURCE_ID:
        raise ValueError(f"connection {connection.connection_id} is not a {SOURCE_REF} connection")
    at = now if now is not None else datetime.now(timezone.utc)
    at_iso = at if isinstance(at, str) else format_timestamp(at if isinstance(at, datetime) else datetime.fromtimestamp(float(at), timezone.utc))

    def refused(http_status: int, reason: str, webhook_id: Optional[str] = None) -> Dict[str, Any]:
        if store is not None:
            store.record_event(connection.connection_id, error={"code": f"webhook_{reason}", "at": at_iso,
                                                                "cause": f"A webhook call was refused ({reason}).",
                                                                "fix": "Check the sender's signing secret and clock."})
        out = {"status": "refused", "http_status": http_status, "reason": reason}
        if webhook_id:
            out["webhook_id"] = webhook_id
        return out

    if connection.status == "disabled" or connection.paused:
        return refused(409, "connection_disabled")
    del runtime  # idle connections accept calls; only paused / disabled refuse
    if len(body) > int(max_body_bytes):
        return refused(413, "too_large")
    if secret is None:
        if store is None:
            raise ValueError("append_webhook_event needs the connection's secret (secret=) or its store (store=)")
        secret = store.secrets(connection.connection_id, principal=None).get("secret")
        if not secret:
            raise ValueError(f"connection {connection.connection_id} has no signing secret")
    verdict = verify_standard_webhook(secret, headers, body, at, int(connection.config.get("tolerance_s") or DEFAULT_TOLERANCE_S))
    if not verdict.ok:
        return refused(401, str(verdict.reason), verdict.webhook_id)
    try:
        data = webhook_event_data(str(verdict.webhook_id), body, _header(headers, "content-type"))
    except ValueError:
        return refused(415, "unsupported_body", verdict.webhook_id)
    data["headers"] = public_headers(headers)
    subject = _subject_from(data.get("body"), connection.config.get("subject_path")) or str(verdict.webhook_id)
    event_time = format_timestamp(datetime.fromtimestamp(int(verdict.timestamp or 0), timezone.utc))
    digest = hashlib.sha256(str(verdict.webhook_id).encode("utf-8")).hexdigest()[:32]
    result = append_connection_event(
        inbox, connection, event_id=f"webhook:{digest}", data=data, time=event_time, subject=subject,
        dedupekey=str(verdict.webhook_id), trust="untrusted", store=store,
    )
    if result is None or result.duplicate:
        refused_dup = refused(409, "replayed_id", verdict.webhook_id)
        refused_dup["status"] = "duplicate"
        return refused_dup
    return {"status": "appended", "http_status": 202, "event_id": result.event_id, "seq": result.seq,
            "webhook_id": verdict.webhook_id}


__all__ = [
    "SOURCE_ID",
    "SOURCE_REF",
    "SOURCE_VERSION",
    "WebhookReceivedTriggerAdapter",
    "WebhookVerification",
    "append_webhook_event",
    "new_webhook_secret",
    "sign_standard_webhook",
    "verify_standard_webhook",
    "webhook_event_data",
]
