"""Runtime-owned email/comms wrapper surface for AbstractCore-backed hosts.

Hosts should import this module instead of reaching into
`abstractcore.tools.comms_tools` directly.

Scope:
- host-local email account listing
- host-local email inbox listing/reading
- host-local email sends

Non-goals:
- durable Runtime effect execution
- run-scoped Runtime history
- arbitrary browser-supplied SMTP/IMAP credentials
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple


def _load_comms_helpers() -> Tuple[Callable[..., Dict[str, Any]], Callable[..., Dict[str, Any]], Callable[..., Dict[str, Any]], Callable[..., Dict[str, Any]]]:
    try:
        from abstractcore.tools.comms_tools import list_email_accounts, list_emails, read_email, send_email
    except Exception as exc:  # pragma: no cover - exercised through facade behavior tests
        raise RuntimeError(
            "Email helpers are unavailable. Install a Runtime environment with "
            "AbstractCore's comms tools support."
        ) from exc
    return list_email_accounts, list_emails, read_email, send_email


def _dependency_error(exc: Exception) -> Dict[str, Any]:
    return {
        "success": False,
        "code": "dependency_missing",
        "error": str(exc),
    }


def list_email_accounts() -> Dict[str, Any]:
    try:
        helper, _list_emails, _read_email, _send_email = _load_comms_helpers()
    except Exception as exc:
        return _dependency_error(exc)
    return helper()


def list_emails(
    *,
    account: Optional[str] = None,
    mailbox: Optional[str] = None,
    since: Optional[str] = None,
    status: str = "all",
    limit: int = 20,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    try:
        _list_email_accounts, helper, _read_email, _send_email = _load_comms_helpers()
    except Exception as exc:
        return _dependency_error(exc)
    return helper(
        account=account,
        mailbox=mailbox,
        since=since,
        status=status,
        limit=limit,
        timeout_s=timeout_s,
    )


def read_email(
    *,
    uid: str,
    account: Optional[str] = None,
    mailbox: Optional[str] = None,
    timeout_s: float = 30.0,
    max_body_chars: int = 20000,
) -> Dict[str, Any]:
    try:
        _list_email_accounts, _list_emails, helper, _send_email = _load_comms_helpers()
    except Exception as exc:
        return _dependency_error(exc)
    return helper(
        uid=uid,
        account=account,
        mailbox=mailbox,
        timeout_s=timeout_s,
        max_body_chars=max_body_chars,
    )


def send_email(
    to: Any,
    subject: str,
    *,
    account: Optional[str] = None,
    body_text: Optional[str] = None,
    body_html: Optional[str] = None,
    cc: Any = None,
    bcc: Any = None,
    timeout_s: float = 30.0,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    try:
        _list_email_accounts, _list_emails, _read_email, helper = _load_comms_helpers()
    except Exception as exc:
        return _dependency_error(exc)
    return helper(
        to=to,
        subject=subject,
        account=account,
        body_text=body_text,
        body_html=body_html,
        cc=cc,
        bcc=bcc,
        timeout_s=timeout_s,
        headers=headers,
    )


__all__ = [
    "list_email_accounts",
    "list_emails",
    "read_email",
    "send_email",
]
