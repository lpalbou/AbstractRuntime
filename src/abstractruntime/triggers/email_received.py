"""`email.received@1`: run an automation on new mail (framework backlog 0992 B4).

The source reads the runtime's durable event inbox (`abstractruntime.email.inbox`), which the
host's mail watcher fills (`abstractruntime.email.feeder`). Like every adapter it performs no
I/O: the controller reads the inbox events after the automation's cursor and passes them in
(`events=`); the adapter filters and batches them from its config and persisted state only.

Config (validated strictly; equality / membership and one literal substring, no expressions):

    {"account": "self", "folder": "INBOX", "uses_model": true, "every": "1h", "max_batch": 100,
     "start_at": <RFC3339, default now>, "auto_submitted": "skip",
     "filter": {"from_in": [addr], "from_domain_in": [domain], "to_in": [addr],
                "subject_contains": str, "has_attachment": bool}}

- `every` is the batch interval: at most one occurrence per `every`, carrying every matching
  message received since the previous one. Default "1h" when `uses_model` (an automation that
  runs a model on new mail), "60s" otherwise; never below "60s" (operator decision, 2026-09-29).
- `max_batch` caps one occurrence's messages; the rest wait for the next occurrence (nothing is
  dropped).
- Each message is admitted at most once per automation: the automation's own inbox cursor
  (`source_state.cursor_seq`) and a `MailCursor`-shaped guard (`source_state.mail_cursor`:
  UIDVALIDITY + last UID) persisted in `_runtime.automation`.
- Mail appended before `start_at` (creation) or while the automation was paused (`rearm`) is
  never admitted.
- `auto_submitted` (RFC 3834 loop protection): "skip" (default) never admits a message whose
  `Auto-Submitted` header is present and not "no" (auto-responders, notifications, other
  automations, this framework's own automatic mail); "admit" lets them through the filter.
  Independently of this option, the account's own automatic mail (framework marker from this
  address, or a Message-ID the host recorded as sent) never enters the inbox at all (feeder).
- `from_domain_in` matches the sender's domain exactly (a subdomain only when listed);
  `to_in` matches any To/Cc address; addresses and domains compare lower-cased.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from email.utils import getaddresses
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .protocol import (
    TriggerAdmission,
    TriggerBinding,
    TriggerConfigError,
    TriggerEnvelope,
    TriggerSource,
    TriggerState,
    TriggerWait,
    format_timestamp,
    parse_duration,
    parse_timestamp,
)

SOURCE_ID = "email.received"
SOURCE_VERSION = 1
SOURCE_REF = f"{SOURCE_ID}@{SOURCE_VERSION}"

_CONFIG_KEYS = ("account", "folder", "uses_model", "every", "max_batch", "start_at", "filter", "auto_submitted")
AUTO_SUBMITTED_SKIP = "skip"
AUTO_SUBMITTED_ADMIT = "admit"
AUTO_SUBMITTED_OPTIONS = (AUTO_SUBMITTED_SKIP, AUTO_SUBMITTED_ADMIT)
_FILTER_KEYS = ("from_in", "from_domain_in", "to_in", "subject_contains", "has_attachment")
MIN_EVERY = timedelta(seconds=60)
DEFAULT_EVERY_MODEL = "1h"
DEFAULT_EVERY_NO_MODEL = "60s"
DEFAULT_MAX_BATCH = 100
MAX_BATCH = 1000
MAX_FILTER_ENTRIES = 200
MAX_SUBJECT_CONTAINS = 200

# Metadata kept in the ledger's envelope (bodies stay out of the envelope; they ride the
# occurrence's inputs).
ENVELOPE_FIELDS = (
    "uid", "uidvalidity", "folder", "message_id", "from", "from_address", "to", "cc", "subject", "date",
    "internaldate",
)


def _address(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TriggerConfigError(f"{field} must be an email address string", field=field)
    a = value.strip().lower()
    local, at, domain = a.partition("@")
    if not at or not local or not domain or "@" in domain or any(c.isspace() or c in "<>,;:\"()[]" for c in a):
        raise TriggerConfigError(f"{field} must be a plain address like name@example.test, got {value!r}", field=field)
    return a


def _domain(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TriggerConfigError(f"{field} must be a domain string", field=field)
    d = value.strip().lower()
    if not d or "@" in d or "." not in d or d.startswith(".") or d.endswith(".") or any(c.isspace() or c in "<>,;:\"()[]/*" for c in d):
        raise TriggerConfigError(f"{field} must be a domain like example.test, got {value!r}", field=field)
    return d


def _str_list(value: Any, field: str, item) -> List[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise TriggerConfigError(f"{field} must be a non-empty list", field=field)
    if len(value) > MAX_FILTER_ENTRIES:
        raise TriggerConfigError(f"{field} holds at most {MAX_FILTER_ENTRIES} entries", field=field)
    out: List[str] = []
    for i, v in enumerate(value):
        norm = item(v, f"{field}[{i}]")
        if norm not in out:
            out.append(norm)
    return out


def validate_filter(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TriggerConfigError("filter must be an object", field="config.filter")
    unknown = sorted(k for k in raw if k not in _FILTER_KEYS)
    if unknown:
        raise TriggerConfigError(f"unknown filter field(s): {unknown}", field=f"config.filter.{unknown[0]}")
    out: Dict[str, Any] = {}
    if raw.get("from_in") is not None:
        out["from_in"] = _str_list(raw["from_in"], "config.filter.from_in", _address)
    if raw.get("from_domain_in") is not None:
        out["from_domain_in"] = _str_list(raw["from_domain_in"], "config.filter.from_domain_in", _domain)
    if raw.get("to_in") is not None:
        out["to_in"] = _str_list(raw["to_in"], "config.filter.to_in", _address)
    if raw.get("subject_contains") is not None:
        s = raw["subject_contains"]
        if not isinstance(s, str) or not s.strip() or len(s) > MAX_SUBJECT_CONTAINS or any(c in s for c in "\r\n"):
            raise TriggerConfigError(
                f"subject_contains must be a one-line string of 1..{MAX_SUBJECT_CONTAINS} characters",
                field="config.filter.subject_contains",
            )
        out["subject_contains"] = s.strip()
    if raw.get("has_attachment") is not None:
        if not isinstance(raw["has_attachment"], bool):
            raise TriggerConfigError("has_attachment must be true or false", field="config.filter.has_attachment")
        out["has_attachment"] = raw["has_attachment"]
    return out


def _addresses(header_value: Any) -> List[str]:
    if not isinstance(header_value, str) or not header_value.strip():
        return []
    return [addr.strip().lower() for _name, addr in getaddresses([header_value]) if addr and "@" in addr]


def is_auto_submitted(payload: Mapping[str, Any]) -> bool:
    """RFC 3834: the message says it was sent automatically (`Auto-Submitted` present, not "no").

    Read from AbstractCore's typed summary field `auto_submitted` (None when the header is
    absent). A message carrying the framework marker counts as automatic as well.
    """
    value = payload.get("auto_submitted")
    if isinstance(value, str) and value.strip() and value.strip().lower() != "no":
        return True
    return bool(str(payload.get("framework_marker") or "").strip())


def message_matches(config: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    """Does one inbox email payload pass the binding's folder and typed filter? (pure)"""
    if not isinstance(payload, Mapping) or payload.get("kind") != "email":
        return False
    if str(payload.get("folder") or "") != str(config.get("folder") or "INBOX"):
        return False
    if config.get("auto_submitted", AUTO_SUBMITTED_SKIP) != AUTO_SUBMITTED_ADMIT and is_auto_submitted(payload):
        return False
    f = config.get("filter") or {}
    sender = str(payload.get("from_address") or "").strip().lower()
    if f.get("from_in") and sender not in f["from_in"]:
        return False
    if f.get("from_domain_in"):
        _local, _at, domain = sender.rpartition("@")
        if not _at or domain not in f["from_domain_in"]:
            return False
    if f.get("to_in"):
        recipients = set(_addresses(payload.get("to"))) | set(_addresses(payload.get("cc")))
        if not recipients.intersection(f["to_in"]):
            return False
    if f.get("subject_contains"):
        if f["subject_contains"].casefold() not in str(payload.get("subject") or "").casefold():
            return False
    if "has_attachment" in f:
        if bool(payload.get("attachments")) != bool(f["has_attachment"]):
            return False
    return True


def _parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _source_state(state: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    src = state.get("source_state") if isinstance(state.get("source_state"), Mapping) else {}
    return {
        "cursor_seq": int(src.get("cursor_seq") or 0),
        "since": src.get("since") or config.get("start_at"),
        "mail_cursor": dict(src["mail_cursor"]) if isinstance(src.get("mail_cursor"), Mapping) else None,
        "last_admitted_at": src.get("last_admitted_at"),
    }


def _guarded(mail_cursor: Optional[Mapping[str, Any]], payload: Mapping[str, Any]) -> bool:
    """True when the message is at or before the automation's MailCursor (already consumed)."""
    if not mail_cursor:
        return False
    try:
        return (
            int(payload.get("uidvalidity")) == int(mail_cursor.get("uidvalidity"))
            and str(payload.get("folder") or "") == str(mail_cursor.get("folder") or "")
            and int(payload.get("uid")) <= int(mail_cursor.get("last_uid"))
        )
    except (TypeError, ValueError):
        return False


class EmailReceivedTriggerAdapter:
    descriptor: TriggerSource = {
        "id": SOURCE_ID,
        "version": SOURCE_VERSION,
        "label": "When an email arrives",
        "config_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "account": {"type": "string", "enum": ["self"]},
                "folder": {"type": "string", "minLength": 1},
                "uses_model": {"type": "boolean"},
                "every": {"type": "string", "format": "duration", "pattern": "^[1-9][0-9]*[smhd]$"},
                "max_batch": {"type": "integer", "minimum": 1, "maximum": MAX_BATCH},
                "start_at": {"type": "string", "format": "date-time"},
                "auto_submitted": {"type": "string", "enum": list(AUTO_SUBMITTED_OPTIONS), "default": AUTO_SUBMITTED_SKIP},
                "filter": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "from_in": {"type": "array", "items": {"type": "string", "format": "email"}},
                        "from_domain_in": {"type": "array", "items": {"type": "string"}},
                        "to_in": {"type": "array", "items": {"type": "string", "format": "email"}},
                        "subject_contains": {"type": "string", "maxLength": MAX_SUBJECT_CONTAINS},
                        "has_attachment": {"type": "boolean"},
                    },
                },
            },
        },
        "event_schema": {
            "type": "object",
            "required": ["count", "event_ids", "messages"],
            "properties": {
                "count": {"type": "integer", "minimum": 1},
                "event_ids": {"type": "array", "items": {"type": "string"}},
                "messages": {"type": "array", "items": {"type": "object"}},
                "first_seq": {"type": "integer"},
                "last_seq": {"type": "integer"},
                "content_trust": {"type": "string", "enum": ["untrusted"]},
            },
        },
        "capabilities": {"kind": "event", "inbox": "email", "content_trust": "untrusted"},
    }

    # --- validation ---------------------------------------------------------------------

    def validate(self, config: Mapping[str, Any], *, now: str) -> Dict[str, Any]:
        if not isinstance(config, Mapping):
            raise TriggerConfigError("email.received config must be an object", field="config")
        unknown = sorted(k for k in config if k not in _CONFIG_KEYS)
        if unknown:
            raise TriggerConfigError(f"unknown email.received field(s): {unknown}", field=f"config.{unknown[0]}")
        account = config.get("account", "self")
        if account != "self":
            raise TriggerConfigError(
                "account must be 'self' (a runtime reads its own user's mailbox)",
                field="config.account",
                reason_code="unsupported_feature",
            )
        folder = config.get("folder", "INBOX")
        if not isinstance(folder, str) or not folder.strip() or any(c in folder for c in "\r\n"):
            raise TriggerConfigError("folder must be a one-line folder name", field="config.folder")
        uses_model = config.get("uses_model", True)
        if not isinstance(uses_model, bool):
            raise TriggerConfigError("uses_model must be true or false", field="config.uses_model")
        every = config.get("every")
        if every is None:
            every = DEFAULT_EVERY_MODEL if uses_model else DEFAULT_EVERY_NO_MODEL
        if parse_duration(every, field="config.every") < MIN_EVERY:
            raise TriggerConfigError("every must be at least 60s", field="config.every")
        max_batch = config.get("max_batch", DEFAULT_MAX_BATCH)
        if isinstance(max_batch, bool) or not isinstance(max_batch, int) or not 1 <= max_batch <= MAX_BATCH:
            raise TriggerConfigError(f"max_batch must be an integer 1..{MAX_BATCH}", field="config.max_batch")
        auto_submitted = config.get("auto_submitted", AUTO_SUBMITTED_SKIP)
        if auto_submitted not in AUTO_SUBMITTED_OPTIONS:
            raise TriggerConfigError(
                f"auto_submitted must be one of {list(AUTO_SUBMITTED_OPTIONS)} (skip: never run on automatic mail)",
                field="config.auto_submitted",
            )
        start = (
            parse_timestamp(config["start_at"], field="config.start_at")
            if config.get("start_at") is not None
            else parse_timestamp(now, field="now")
        )
        return {
            "account": "self",
            "folder": folder.strip(),
            "uses_model": uses_model,
            "every": every,
            "max_batch": max_batch,
            "start_at": format_timestamp(start),
            "auto_submitted": auto_submitted,
            "filter": validate_filter(config.get("filter")),
        }

    def initial_state(self, config: Mapping[str, Any]) -> TriggerState:
        state: Dict[str, Any] = {"anchor": config.get("start_at"), "tick": 0, "scheduled_count": 0, "exhausted": False}
        state["source_state"] = {"cursor_seq": 0, "since": config.get("start_at"), "mail_cursor": None, "last_admitted_at": None}
        return state  # type: ignore[return-value]

    # --- selection (pure) ---------------------------------------------------------------

    def candidates(self, binding: TriggerBinding, *, state: Mapping[str, Any], events: Optional[Sequence[Mapping[str, Any]]]) -> List[Mapping[str, Any]]:
        """Inbox records this automation would admit, oldest first."""
        config = binding["config"]
        src = _source_state(state, config)
        since = _parse_iso(src["since"])
        out: List[Mapping[str, Any]] = []
        for rec in sorted(events or (), key=lambda r: int(r.get("seq") or 0)):
            if int(rec.get("seq") or 0) <= src["cursor_seq"]:
                continue
            appended = _parse_iso(rec.get("appended_at"))
            if since is not None and (appended is None or appended <= since):
                continue
            payload = rec.get("payload") or {}
            if _guarded(src["mail_cursor"], payload):
                continue
            if message_matches(config, payload):
                out.append(rec)
        return out

    def skip_ahead(self, binding: TriggerBinding, *, state: Mapping[str, Any], events: Optional[Sequence[Mapping[str, Any]]]) -> int:
        """The cursor this automation can move to without admitting anything: past every leading
        record it would never admit (derived from config + inbox, so replay-safe)."""
        src = _source_state(state, binding["config"])
        cands = self.candidates(binding, state=state, events=events)
        if cands:
            return max(src["cursor_seq"], int(cands[0]["seq"]) - 1)
        seqs = [int(r.get("seq") or 0) for r in events or ()]
        return max([src["cursor_seq"], *seqs])

    def _due_at(self, config: Mapping[str, Any], src: Mapping[str, Any], now_dt: datetime) -> datetime:
        last = _parse_iso(src.get("last_admitted_at"))
        if last is None:
            return now_dt
        return last + parse_duration(config["every"], field="config.every")

    # --- protocol ---------------------------------------------------------------------------

    def prepare(self, binding: TriggerBinding, *, state: TriggerState, now: str, events: Optional[Sequence[Mapping[str, Any]]] = None) -> TriggerWait:
        cands = self.candidates(binding, state=state, events=events)
        if not cands:
            return {"kind": "idle"}  # woken by the watcher when new mail is appended
        src = _source_state(state, binding["config"])
        due = self._due_at(binding["config"], src, parse_timestamp(now, field="now"))
        return {"kind": "until", "until": format_timestamp(due)}

    def admit(self, binding: TriggerBinding, *, state: TriggerState, now: str, events: Optional[Sequence[Mapping[str, Any]]] = None) -> Optional[TriggerAdmission]:
        config = binding["config"]
        cands = self.candidates(binding, state=state, events=events)
        if not cands:
            return None
        now_dt = parse_timestamp(now, field="now")
        src = _source_state(state, config)
        if self._due_at(config, src, now_dt) > now_dt:
            return None
        batch = list(cands[: int(config.get("max_batch") or DEFAULT_MAX_BATCH)])
        if len(cands) > len(batch):
            cursor_seq = int(batch[-1]["seq"])  # the rest (and what lies between) is read next time
        else:
            cursor_seq = max([src["cursor_seq"], *(int(r.get("seq") or 0) for r in events or ())])
        last = batch[-1]["payload"]
        uv, folder = int(last.get("uidvalidity") or 0), str(last.get("folder") or config.get("folder") or "INBOX")
        top_uid = max(int(r["payload"].get("uid") or 0) for r in batch if int(r["payload"].get("uidvalidity") or 0) == uv)
        prev = src["mail_cursor"]
        if prev and int(prev.get("uidvalidity") or 0) == uv and str(prev.get("folder") or "") == folder:
            top_uid = max(top_uid, int(prev.get("last_uid") or 0))
        event_ids = [str(r["event_id"]) for r in batch]
        digest = hashlib.sha256("\n".join(event_ids).encode("utf-8")).hexdigest()[:32]
        fired_at = format_timestamp(now_dt)
        new_state: Dict[str, Any] = {
            "anchor": state.get("anchor"),
            "tick": int(state.get("tick") or 0) + 1,
            "scheduled_count": int(state.get("scheduled_count") or 0) + 1,
            "exhausted": False,
            "source_state": {
                "cursor_seq": cursor_seq,
                "since": src["since"],
                "mail_cursor": {"uidvalidity": uv, "last_uid": top_uid, "folder": folder},
                "last_admitted_at": fired_at,
            },
        }
        admission: Dict[str, Any] = {
            "event_id": f"{SOURCE_REF}:{binding['binding_id']}:{digest}",
            "fired_at": fired_at,
            "payload": {
                "count": len(batch),
                "event_ids": event_ids,
                "first_seq": int(batch[0]["seq"]),
                "last_seq": int(batch[-1]["seq"]),
                "content_trust": "untrusted",
                "messages": [{k: r["payload"].get(k) for k in ENVELOPE_FIELDS} for r in batch],
            },
            "state": new_state,
            # Whole messages for the occurrence's inputs (not part of the envelope).
            "inputs": {"emails": [dict(r["payload"]) for r in batch]},
        }
        return admission  # type: ignore[return-value]

    def rearm(self, binding: TriggerBinding, *, state: TriggerState, now: str) -> TriggerState:
        src = _source_state(state, binding["config"])
        src["since"] = format_timestamp(parse_timestamp(now, field="now"))  # mail up to now is never admitted
        return {  # type: ignore[return-value]
            "anchor": state.get("anchor"),
            "tick": int(state.get("tick") or 0),
            "scheduled_count": int(state.get("scheduled_count") or 0),
            "exhausted": False,
            "source_state": src,
        }

    def normalize(self, binding: TriggerBinding, *, event_id: str, fired_at: str, payload: Mapping[str, Any]) -> TriggerEnvelope:
        return {
            "event_id": str(event_id),
            "source_id": SOURCE_ID,
            "source_version": SOURCE_VERSION,
            "fired_at": str(fired_at),
            "payload": dict(payload),
            "binding_id": str(binding["binding_id"]),
        }


__all__ = [
    "AUTO_SUBMITTED_ADMIT",
    "AUTO_SUBMITTED_OPTIONS",
    "AUTO_SUBMITTED_SKIP",
    "EmailReceivedTriggerAdapter",
    "is_auto_submitted",
    "SOURCE_ID",
    "SOURCE_REF",
    "SOURCE_VERSION",
    "message_matches",
    "validate_filter",
]
