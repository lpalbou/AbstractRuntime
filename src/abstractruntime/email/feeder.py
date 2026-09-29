"""Mailbox -> durable inbox: the one poll step a mail watcher runs (framework backlog 0992 B4/C4).

The host (the gateway's per-user watcher) calls `EmailInboxFeeder.poll(ctx)` on its cadence
with the user's `EmailContext`; the feeder does the rest through AbstractCore's mail library:

- read-only (EXAMINE / BODY.PEEK; AbstractCore refuses any mailbox-changing command);
- the first poll is a baseline: no events, the cursor at the newest message, so only mail that
  arrives afterwards becomes an event;
- each new message, oldest first, is fetched WHOLE (no truncation) and appended as one inbox
  event `email_event_id(account_ref, folder, uidvalidity, uid)`; the per-folder `MailCursor`
  (UIDVALIDITY + last UID) advances message by message ONLY after that append is durable;
- UIDVALIDITY change (the server rebuilt the folder): AbstractCore resynchronises by date and
  the feeder skips messages already in the inbox (same Message-ID), so nothing is lost and
  nothing is delivered twice;
- a message that cannot be fetched on `failure_threshold` (3) polls in a row is recorded as
  unprocessable (uid, code, cause, fix) and passed, so one bad message never blocks the mailbox;
- a connection / sign-in failure is a typed `{code, cause, fix, retryable}` in the report and the
  stream status, the next poll waits a capped backoff (60 s doubling to 15 min); nothing raises
  and nothing is paused. Credentials never reach the inbox, the status or the report.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .inbox import EventInbox

EMAIL_STREAM_PREFIX = "email:"
# Errors about ONE message (its fetch can fail again and again): they count toward the
# unprocessable threshold. Every other code is about the connection or the account.
MESSAGE_LEVEL_CODES = frozenset({"email_message_not_found", "email_protocol_error", "email_server_error"})
MAX_UNPROCESSABLE_KEPT = 50


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def email_stream(account_ref: str, folder: str) -> str:
    """The inbox stream of one account folder: `email:<account_ref>:<folder>`."""
    return f"{EMAIL_STREAM_PREFIX}{account_ref}:{folder}"


def email_event_id(account_ref: str, folder: str, uidvalidity: int, uid: int) -> str:
    """`email:` + sha256(account_ref, folder, uidvalidity, uid): the message's inbox identity."""
    raw = f"{account_ref}\n{folder}\n{int(uidvalidity)}\n{int(uid)}".encode("utf-8")
    return EMAIL_STREAM_PREFIX + hashlib.sha256(raw).hexdigest()


def email_event_payload(account_ref: str, detail: Any) -> Dict[str, Any]:
    """The inbox payload of one fetched message: every header field and the whole bodies.

    All of it was written by the sender (untrusted data); none of it is a credential.
    """
    data = detail.to_dict()
    for volatile in ("flags", "seen"):
        data.pop(volatile, None)
    try:
        data["uid"] = int(data.get("uid"))
    except (TypeError, ValueError):
        pass
    return {"kind": "email", "account_ref": account_ref, **data}


@dataclass
class PollReport:
    ok: bool
    skipped: bool = False
    baseline: bool = False
    reset: bool = False
    appended: List[str] = field(default_factory=list)
    duplicates: int = 0
    unprocessable: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    next_poll_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "skipped": self.skipped,
            "baseline": self.baseline,
            "reset": self.reset,
            "appended": list(self.appended),
            "duplicates": self.duplicates,
            "unprocessable": [dict(u) for u in self.unprocessable],
            "error": dict(self.error) if self.error else None,
            "next_poll_at": self.next_poll_at,
        }


def _typed(err: Any, at: str) -> Dict[str, Any]:
    return {
        "code": str(getattr(err, "code", "email_error")),
        "cause": str(getattr(err, "cause", "") or "The mail server could not be reached."),
        "fix": str(getattr(err, "fix", "") or "Open Settings -> Email and Test the account."),
        "retryable": bool(getattr(err, "retryable", False)),
        "at": at,
    }


class EmailInboxFeeder:
    """Poll one account folder into an `EventInbox` (state kept in the inbox's stream state)."""

    def __init__(
        self,
        inbox: EventInbox,
        *,
        account_ref: str,
        folder: Optional[str] = None,
        max_messages_per_poll: int = 200,
        failure_threshold: int = 3,
        backoff_initial_s: float = 60.0,
        backoff_max_s: float = 900.0,
    ) -> None:
        ref = str(account_ref or "").strip()
        if not ref:
            raise ValueError("account_ref is required")
        self.inbox = inbox
        self.account_ref = ref
        self.folder = str(folder).strip() if folder else None
        self.max_messages_per_poll = max(1, int(max_messages_per_poll))
        self.failure_threshold = max(1, int(failure_threshold))
        self.backoff_initial_s = float(backoff_initial_s)
        self.backoff_max_s = float(backoff_max_s)

    def __repr__(self) -> str:
        return f"EmailInboxFeeder(account_ref={self.account_ref!r}, folder={self.folder!r})"

    def _folder_of(self, ctx: Any) -> str:
        if self.folder:
            return self.folder
        imap = getattr(getattr(ctx, "account", None), "imap", None)
        return str(getattr(imap, "folder", "") or "INBOX")

    def stream(self, folder: Optional[str] = None) -> str:
        return email_stream(self.account_ref, folder or self.folder or "INBOX")

    def status(self, folder: Optional[str] = None) -> Dict[str, Any]:
        """`{state, cursor, last_poll, last_ok, next_poll_at, consecutive_failures, last_error, unprocessable}`."""
        st = self.inbox.stream_state(self.stream(folder))
        return {
            "state": st.get("state") or "never_polled",
            "cursor": st.get("cursor"),
            "last_poll": st.get("last_poll"),
            "last_ok": st.get("last_ok"),
            "next_poll_at": st.get("next_poll_at"),
            "consecutive_failures": int(st.get("consecutive_failures") or 0),
            "last_error": st.get("last_error"),
            "unprocessable": list(st.get("unprocessable") or []),
        }

    def _backoff(self, failures: int) -> timedelta:
        seconds = min(self.backoff_initial_s * (2 ** max(0, failures - 1)), self.backoff_max_s)
        return timedelta(seconds=seconds)

    def poll(self, ctx: Any, *, now: Optional[str] = None, force: bool = False) -> PollReport:
        from abstractcore.comms.email import EmailError, EmailNotConfigured, MailCursor

        now_dt = _parse(now) or datetime.now(timezone.utc)
        at = _iso(now_dt)
        folder = self._folder_of(ctx)
        stream = email_stream(self.account_ref, folder)
        st = self.inbox.stream_state(stream)
        due = _parse(st.get("next_poll_at"))
        if not force and due is not None and now_dt < due:
            return PollReport(ok=True, skipped=True, next_poll_at=st.get("next_poll_at"), error=st.get("last_error"))

        def save() -> None:
            self.inbox.set_stream_state(stream, st)

        def fail(err: Any) -> PollReport:
            failures = int(st.get("consecutive_failures") or 0) + 1
            nxt = _iso(now_dt + self._backoff(failures))
            st.update(state="error", last_poll=at, consecutive_failures=failures, last_error=_typed(err, at), next_poll_at=nxt)
            save()
            return PollReport(ok=False, error=dict(st["last_error"]), next_poll_at=nxt)

        try:
            if ctx is None:
                raise EmailNotConfigured(
                    "No email account is connected.", "Connect an email account in Settings -> Email."
                )
            if not ctx.account.can_read:
                raise EmailNotConfigured(
                    "The email account has no IMAP (receive) settings.",
                    "Connect the account again with its IMAP host to receive mail.",
                )
            client = ctx.client()
            cursor = MailCursor.from_dict(st.get("cursor"))
            if cursor is not None and cursor.folder != folder:
                cursor = None  # the folder setting changed: start a new baseline
            result = client.fetch_new(cursor, folder=folder, limit=self.max_messages_per_poll)
        except EmailError as err:
            return fail(err)

        report = PollReport(ok=True, baseline=bool(result.baseline), reset=bool(result.reset))
        if result.baseline:
            st.update(cursor=result.cursor.to_dict(), state="ok", last_poll=at, last_ok=at, consecutive_failures=0,
                      last_error=None, next_poll_at=None)
            save()
            return report
        if result.reset:
            st["last_reset_at"] = at
            if not result.messages:
                # Nothing to resynchronise: restart from the newest message of the new epoch
                # (a cursor at UID 0 would deliver the whole rebuilt folder as new mail).
                try:
                    rebase = client.fetch_new(None, folder=folder)
                except EmailError as err:
                    return fail(err)
                st.update(cursor=rebase.cursor.to_dict(), state="ok", last_poll=at, last_ok=at,
                          consecutive_failures=0, last_error=None, next_poll_at=None)
                save()
                return report
        seen_until = _parse(cursor.last_internaldate) if (result.reset and cursor is not None) else None

        failures_by_uid: Dict[str, int] = dict(st.get("message_failures") or {})
        unprocessable: List[Dict[str, Any]] = list(st.get("unprocessable") or [])
        stopped_on: Optional[Dict[str, Any]] = None

        def advance(summary: Any) -> None:
            st["cursor"] = MailCursor(
                int(summary.uidvalidity), int(summary.uid), folder, summary.internaldate or (cursor.last_internaldate if cursor else "")
            ).to_dict()
            failures_by_uid.pop(str(summary.uid), None)
            st["message_failures"] = failures_by_uid

        for summary in result.messages:
            event_id = email_event_id(self.account_ref, folder, summary.uidvalidity, summary.uid)
            older = seen_until is not None and (_parse(summary.internaldate) or seen_until) < seen_until
            if self.inbox.get(event_id) is not None or older or (
                result.reset and summary.message_id and self.inbox.has_dedupe_key(stream, summary.message_id)
            ):
                # Already delivered (same id), or, after a UIDVALIDITY reset, a message older
                # than the newest one seen before the reset (delivered then, or before the
                # baseline) or one already in the inbox under its old UID (same Message-ID).
                report.duplicates += 1
                advance(summary)
                save()
                continue
            try:
                detail = client.get(summary.uid, folder=folder)
            except EmailError as err:
                if err.code not in MESSAGE_LEVEL_CODES:
                    st["message_failures"] = failures_by_uid
                    return fail(err)
                n = failures_by_uid.get(str(summary.uid), 0) + 1
                failures_by_uid[str(summary.uid)] = n
                st["message_failures"] = failures_by_uid
                if n >= self.failure_threshold:
                    item = {"uid": int(summary.uid), "uidvalidity": int(summary.uidvalidity), **_typed(err, at)}
                    unprocessable = (unprocessable + [item])[-MAX_UNPROCESSABLE_KEPT:]
                    st["unprocessable"] = unprocessable
                    report.unprocessable.append(item)
                    advance(summary)
                    save()
                    continue
                stopped_on = _typed(err, at)
                save()
                break  # retried at the next poll; later messages wait behind it (order kept)
            self.inbox.append(
                stream=stream,
                event_id=event_id,
                payload=email_event_payload(self.account_ref, detail),
                dedupe_key=summary.message_id or None,
                appended_at=at,
            )
            advance(summary)  # only after the append is durable
            save()
            report.appended.append(event_id)

        if not result.messages:
            st["cursor"] = result.cursor.to_dict()
        st.update(state="ok", last_poll=at, last_ok=at, consecutive_failures=0, next_poll_at=None,
                  last_error=stopped_on)
        save()
        return report


__all__ = [
    "EMAIL_STREAM_PREFIX",
    "EmailInboxFeeder",
    "MESSAGE_LEVEL_CODES",
    "PollReport",
    "email_event_id",
    "email_event_payload",
    "email_stream",
]
