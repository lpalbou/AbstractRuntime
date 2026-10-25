"""Sample occurrence inputs: what a triggered flow would receive, without waiting for the trigger.

Pressing Run on a flow that an automation triggers behaves as if the trigger fired, with the
trigger's own configuration (the chosen mailbox, the feed connection, the webhook connection): the
person never types `messages`, `mailbox` or `event` by hand. The host (the gateway) calls

    sample = sample_occurrence_inputs(runtime, source_id, source_version, config, count=100, principal=...)

and starts the flow with `sample["inputs"]`: exactly the flat named inputs plus `event` that the
controller's `build_prepared` gives a real occurrence of that trigger (see docs/automations.md,
"Occurrence inputs and outputs"), built from real data and read only:

- `email.received@1`: the newest `count` messages of the bound mailbox folder that pass the
  trigger's filters, read-only on ONE IMAP connection of the mailbox's own client
  (`batched_newest_messages`: AbstractCore's `EmailClient.session()` + `search` + `get_many`,
  batched fetches); a message whose text bodies exceed `EMAIL_SAMPLE_BODY_CAP` (1 MB) comes with
  `body_skipped` instead of its bodies; attachments are listed, never downloaded; nothing is
  marked read, no cursor or inbox change. The fetch stops after `EMAIL_SAMPLE_BUDGET_S` (20 s) with what it has
  (`truncated: true`). `messages` is stored as an artifact like a real occurrence's, and listed
  in `resolve_vars`;
- `feed.item@1`: the latest `count` items of the connection's feed, one fetch, no cursor change;
- `webhook.received@1`: the last `count` calls stored for the connection; when none was
  received yet, ONE documented sample call (`SAMPLE_WEBHOOK_CALL`, `sample_call: true`);
- `schedule@1` / `schedule@2`: `scheduled_at` = now, `occurrence` = 0;
- `manual@1`: nothing but `event`.

Lists are newest first. The `event` carries the CloudEvents extension `abstracttest: true` (a flow
or a person can tell a test run from a real one), and its id starts with `test:`.

Returns `{"inputs": {...}, "resolve_vars": [...], "truncated": bool, "source": "<id>@<version>",
"count": <items in the sample>}`. `resolved_sample_inputs(sample, artifact_store)` gives the inputs
with their artifact refs resolved (what the occurrence run's own vars hold), for a host that
starts the test run with `Runtime.start` rather than `START_SUBWORKFLOW resolve_vars`.
"""

from __future__ import annotations

import copy
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..triggers.cloudevents import ensure_cloudevent_attributes, to_cloudevent
from ..triggers.protocol import format_timestamp
from ..triggers.registry import get_trigger_adapter

DEFAULT_SAMPLE_COUNT = 100
MAX_SAMPLE_COUNT = 1000
EMAIL_SAMPLE_BUDGET_S = 20.0
# Reading limit of a sampled message's text bodies: over it, the body is skipped with a typed record
# (never cut, ADR-0026). Normal mail is far below; the cap bounds a test's transfer.
EMAIL_SAMPLE_BODY_CAP = 1024 * 1024
EMAIL_SAMPLE_BATCH = 100  # messages read per round when the trigger has filters
# The call a webhook connection that never received one is tested with.
SAMPLE_WEBHOOK_CALL: Dict[str, Any] = {
    "webhook_id": "sample",
    "content_type": "application/json",
    "headers": {"content-type": "application/json"},
    "body": {"sample": True, "message": "A sample webhook call: this connection has not received one yet."},
}


class SampleError(ValueError):
    """The sample cannot be built (`reason_code`: invalid_sample, connection_not_found,
    email_not_configured, sample_unavailable)."""

    def __init__(self, message: str, *, reason_code: str = "invalid_sample") -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _now_iso() -> str:
    return format_timestamp(datetime.now(timezone.utc))


def _envelope(adapter: Any, binding: Dict[str, Any], payload: Dict[str, Any], *, now: str) -> Dict[str, Any]:
    env = adapter.normalize(binding, event_id=f"test:{uuid.uuid4().hex}", fired_at=now, payload=payload)
    caps = (getattr(adapter, "descriptor", None) or {}).get("capabilities") or {}
    env = ensure_cloudevent_attributes(dict(env), trust="untrusted" if caps.get("content_trust") == "untrusted" else "trusted")
    env["abstracttest"] = True
    return env


def _connection(runtime: Any, config: Mapping[str, Any], principal: Optional[str]) -> Any:
    store = getattr(runtime, "trigger_connection_store", None)
    if store is None:
        raise SampleError("this runtime has no trigger connection store", reason_code="sample_unavailable")
    cid = (config or {}).get("connection_id")
    conn = store.get(str(cid or ""), principal=principal) if cid else None
    if conn is None:
        raise SampleError(f"no trigger connection {cid!r} for this user", reason_code="connection_not_found")
    return conn


def _batch_payload(events: List[Dict[str, Any]], *, connection_id: str) -> Dict[str, Any]:
    trusted = bool(events) and all(e.get("abstracttrust") == "trusted" for e in events)
    return {
        "count": len(events),
        "event_ids": [str(e.get("id")) for e in events],
        "content_trust": "trusted" if trusted else "untrusted",
        "connection_id": connection_id,
        "events": events,
    }


def _email_sample(runtime: Any, adapter: Any, config: Dict[str, Any], binding: Dict[str, Any], *, count: int,
                  email_context: Any, now: str, budget_s: float) -> tuple:
    """(payload, inputs, mailbox_binding, truncated) for `email.received@1`."""
    from ..email.binding import email_binding_for_trigger
    from ..email.feeder import email_event_payload
    from ..triggers.email_received import ENVELOPE_FIELDS, message_matches

    mail_binding = email_binding_for_trigger(getattr(runtime, "email_binding", None), binding)
    ctx = email_context
    if ctx is None:
        resolver = getattr(runtime, "email_context_resolver", None)
        if resolver is None or mail_binding is None:
            raise SampleError("no mailbox is connected for this runtime", reason_code="email_not_configured")
        try:
            ctx = resolver(mail_binding, use="action")
        except TypeError:
            ctx = resolver(mail_binding)
        if ctx is None:
            raise SampleError("the mailbox is not connected or not enabled", reason_code="email_not_configured")
    account = str(config.get("account") or "self")
    account_ref = getattr(mail_binding, "account_ref", None) or f"local:local:{'default' if account in ('self', 'default') else account}"
    folder = str(config.get("folder") or "INBOX")
    deadline = time.monotonic() + float(budget_s)
    emails: List[Dict[str, Any]] = []
    truncated = False
    before_uid = None
    client = ctx.client()
    while len(emails) < count:
        details, more, before_uid, hit_deadline = batched_newest_messages(
            client, folder=folder, before_uid=before_uid, deadline=deadline,
            # Filters may reject messages: read a whole batch at a time then (still one connection each).
            count=(count - len(emails)) if not config.get("filter") else max(count - len(emails), EMAIL_SAMPLE_BATCH),
        )
        for detail in details:
            payload = email_event_payload(account_ref, detail)
            if message_matches(config, payload):
                emails.append(payload)
                if len(emails) >= count:
                    break
        if hit_deadline:
            truncated = True
            break
        if not more or len(emails) >= count:
            break
    payload = {
        "count": len(emails),
        "event_ids": [f"test:{e.get('uid')}" for e in emails],
        "content_trust": "untrusted",
        "messages": [{k: e.get(k) for k in ENVELOPE_FIELDS} for e in emails],
    }
    return payload, {"emails": emails}, mail_binding, truncated


def batched_newest_messages(
    client: Any,
    *,
    folder: str,
    count: int,
    before_uid: Optional[int] = None,
    deadline: Optional[float] = None,
    body_cap: int = EMAIL_SAMPLE_BODY_CAP,
) -> tuple:
    """The newest `count` messages of `folder` (UID < `before_uid` when given), newest first, as
    AbstractCore `MessageDetail`s, read on ONE connection (`EmailClient.session()`):
    `search(limit=count)` lists the newest UIDs, then `get_many(uids, max_message_bytes=body_cap)`
    reads them in batched fetches (headers + structure per 100 UIDs, one PEEK fetch per text
    section). A message whose text bodies exceed `body_cap` arrives with its headers, attachment
    list and a typed `body_skipped` record, never a cut body. Attachments are listed, never
    downloaded. Read-only (EXAMINE, PEEK): nothing is marked read. Needs AbstractCore >= 2.27.0.
    Returns `(details, more, next_before_uid, hit_deadline)`."""
    details: List[Any] = []
    with client.session():
        if deadline is not None and time.monotonic() > deadline:
            return details, True, before_uid, True
        page = client.search(folder=folder, limit=max(1, int(count)), before_uid=before_uid)
        uids = [int(m.uid) for m in page.get("messages") or []]
        more = bool(page.get("has_more"))
        next_before = page.get("next_before_uid") if more else None
        if deadline is not None and time.monotonic() > deadline:
            return details, True, before_uid, True
        got = client.get_many(uids, folder=folder, max_message_bytes=int(body_cap)) if uids else {}
        for uid in uids:  # newest first
            detail = got.get(uid)
            if detail is not None and not isinstance(detail, Exception):
                details.append(detail)
    return details, more, next_before, False


def sample_occurrence_inputs(
    runtime: Any,
    source_id: str,
    source_version: int,
    config: Mapping[str, Any],
    *,
    count: int = DEFAULT_SAMPLE_COUNT,
    principal: Optional[str] = None,
    now: Optional[str] = None,
    email_context: Any = None,
    fetch: Optional[Callable[..., Any]] = None,
    email_budget_s: float = EMAIL_SAMPLE_BUDGET_S,
) -> Dict[str, Any]:
    """The inputs a real occurrence of this trigger would receive, from real data (see the module
    doc). `config` is the trigger's binding config (validated here like at creation); `principal`
    owns the connection a connection-bound source names; `email_context` / `fetch` replace the
    runtime's mail resolver / the feed fetch (hosts and tests). Raises SampleError."""
    from .controller import freeze_named_inputs, occurrence_mailbox, occurrence_named_inputs

    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_SAMPLE_COUNT:
        raise SampleError(f"count must be an integer 1..{MAX_SAMPLE_COUNT}")
    at = now or _now_iso()
    try:
        adapter = get_trigger_adapter(source_id, source_version)
    except LookupError as exc:
        raise SampleError(str(exc), reason_code="invalid_sample") from exc
    try:
        cfg = adapter.validate(dict(config or {}), now=at)
    except ValueError as exc:
        raise SampleError(f"invalid trigger config: {exc}") from exc
    binding = {"binding_id": "test", "source_id": source_id, "source_version": int(source_version), "config": cfg}
    inputs: Dict[str, Any] = {}
    truncated = False
    mail_binding = None
    if source_id == "schedule":
        payload: Dict[str, Any] = {"tick": 0, "scheduled_at": at}
    elif source_id == "manual":
        payload = {"command_id": "test"}
    elif source_id == "email.received":
        payload, inputs, mail_binding, truncated = _email_sample(
            runtime, adapter, cfg, binding, count=count, email_context=email_context, now=at, budget_s=email_budget_s
        )
    elif source_id == "feed.item":
        payload = _feed_sample(runtime, cfg, principal=principal, count=count, now=at, fetch=fetch)
    elif source_id == "webhook.received":
        payload = _webhook_sample(runtime, cfg, principal=principal, count=count, now=at)
    else:
        raise SampleError(f"no sample is available for {source_id}@{source_version}", reason_code="sample_unavailable")
    envelope = _envelope(adapter, binding, payload, now=at)
    named = occurrence_named_inputs(envelope, inputs=inputs, index=0,
                                    mailbox=occurrence_mailbox(mail_binding, binding, source_id))
    resolve_vars = freeze_named_inputs(named, artifact_store=getattr(runtime, "artifact_store", None), run_id=None)
    out_inputs = {**named, "event": to_cloudevent(envelope)}
    return {
        "inputs": out_inputs,
        "resolve_vars": resolve_vars,
        "truncated": bool(truncated),
        "source": f"{source_id}@{source_version}",
        "count": int(payload.get("count") or 0) if source_id not in ("schedule", "manual") else 1,
    }


def _feed_sample(runtime: Any, cfg: Dict[str, Any], *, principal: Optional[str], count: int, now: str, fetch: Any) -> Dict[str, Any]:
    from ..triggers.connections import connection_event
    from ..triggers.feed import FeedParseError, parse_feed, urllib_fetch

    conn = _connection(runtime, cfg, principal)
    url = str(conn.config.get("url") or "")
    try:
        result = (fetch or urllib_fetch)(url, etag=None, last_modified=None, timeout=20.0)
    except Exception as exc:  # noqa: BLE001
        raise SampleError(f"the feed could not be fetched: {type(exc).__name__}", reason_code="sample_unavailable") from exc
    if int(getattr(result, "status", 0)) != 200:
        raise SampleError(f"the feed answered HTTP {getattr(result, 'status', '?')}", reason_code="sample_unavailable")
    try:
        parsed = parse_feed(result.body)
    except FeedParseError as exc:
        raise SampleError(str(exc), reason_code="sample_unavailable") from exc
    ordered = [it for _, it in sorted(enumerate(parsed.items), key=lambda p: (p[1].published or "", -p[0]), reverse=True)]
    events = [
        connection_event(conn, event_id=f"test:{i}", data=item.to_data(feed_title=parsed.title, feed_url=url),
                         time=item.published or now, subject=item.id, dedupekey=item.id, trust="untrusted")
        for i, item in enumerate(ordered[:count])
    ]
    return _batch_payload(events, connection_id=conn.connection_id)


def _webhook_sample(runtime: Any, cfg: Dict[str, Any], *, principal: Optional[str], count: int, now: str) -> Dict[str, Any]:
    from ..triggers.connections import connection_event

    conn = _connection(runtime, cfg, principal)
    inbox = getattr(runtime, "event_inbox", None)
    stored = inbox.read(stream=conn.stream) if inbox is not None else []
    records = [r for r in stored if isinstance((r.get("payload") or {}).get("data"), Mapping)
               and "webhook_id" in r["payload"]["data"]][-count:]
    events = [copy.deepcopy(r["payload"]) for r in reversed(records)]
    if records:
        payload = _batch_payload(events, connection_id=conn.connection_id)
        payload["first_seq"], payload["last_seq"] = int(records[0]["seq"]), int(records[-1]["seq"])
        return payload
    if not events:
        events = [connection_event(conn, event_id="test:sample", data=copy.deepcopy(SAMPLE_WEBHOOK_CALL), time=now,
                                   subject="sample", dedupekey="sample", trust="untrusted")]
        events[0]["sample_call"] = True
    return _batch_payload(events, connection_id=conn.connection_id)


def resolved_sample_inputs(sample: Mapping[str, Any], *, artifact_store: Any) -> Dict[str, Any]:
    """`sample["inputs"]` with the `resolve_vars` artifact refs replaced by their content."""
    from .attention import resolve_strict

    out = copy.deepcopy(dict(sample["inputs"]))
    for key in sample.get("resolve_vars") or []:
        if key in out:
            out[key] = resolve_strict(out[key], artifact_store=artifact_store)
    return out


__all__ = [
    "DEFAULT_SAMPLE_COUNT",
    "EMAIL_SAMPLE_BODY_CAP",
    "EMAIL_SAMPLE_BUDGET_S",
    "batched_newest_messages",
    "MAX_SAMPLE_COUNT",
    "SAMPLE_WEBHOOK_CALL",
    "SampleError",
    "resolved_sample_inputs",
    "sample_occurrence_inputs",
]
