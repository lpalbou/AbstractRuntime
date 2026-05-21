from __future__ import annotations

from typing import Any, Dict

import pytest

from abstractruntime.integrations import abstractcore
from abstractruntime.integrations.abstractcore import comms_facade
from abstractruntime.integrations.abstractcore.comms_facade import (
    list_email_accounts,
    list_emails,
    read_email,
    send_email,
)


def test_public_comms_facade_exports_are_available() -> None:
    assert abstractcore.list_email_accounts is list_email_accounts
    assert abstractcore.list_emails is list_emails
    assert abstractcore.read_email is read_email
    assert abstractcore.send_email is send_email
    assert "list_email_accounts" in abstractcore.__all__
    assert "list_emails" in abstractcore.__all__
    assert "read_email" in abstractcore.__all__
    assert "send_email" in abstractcore.__all__
    assert "list_email_accounts" in comms_facade.__all__
    assert "list_emails" in comms_facade.__all__
    assert "read_email" in comms_facade.__all__
    assert "send_email" in comms_facade.__all__


def test_list_email_accounts_delegates_to_core_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        comms_facade,
        "_load_comms_helpers",
        lambda: (
            lambda: {"success": True, "accounts": [{"account": "default"}]},
            lambda **_kwargs: {},
            lambda **_kwargs: {},
            lambda **_kwargs: {},
        ),
    )

    payload = list_email_accounts()

    assert payload == {"success": True, "accounts": [{"account": "default"}]}


def test_list_emails_delegates_to_core_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: Dict[str, Any] = {}

    def _list_emails(**kwargs: Any) -> Dict[str, Any]:
        recorded.update(kwargs)
        return {"success": True, "messages": []}

    monkeypatch.setattr(
        comms_facade,
        "_load_comms_helpers",
        lambda: (
            lambda: {},
            _list_emails,
            lambda **_kwargs: {},
            lambda **_kwargs: {},
        ),
    )

    payload = list_emails(account="default", mailbox="INBOX", since="7d", status="unread", limit=5, timeout_s=9)

    assert payload == {"success": True, "messages": []}
    assert recorded == {
        "account": "default",
        "mailbox": "INBOX",
        "since": "7d",
        "status": "unread",
        "limit": 5,
        "timeout_s": 9,
    }


def test_read_email_delegates_to_core_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: Dict[str, Any] = {}

    def _read_email(**kwargs: Any) -> Dict[str, Any]:
        recorded.update(kwargs)
        return {"success": True, "uid": kwargs["uid"]}

    monkeypatch.setattr(
        comms_facade,
        "_load_comms_helpers",
        lambda: (
            lambda: {},
            lambda **_kwargs: {},
            _read_email,
            lambda **_kwargs: {},
        ),
    )

    payload = read_email(uid="123", account="default", mailbox="INBOX", timeout_s=11, max_body_chars=5000)

    assert payload == {"success": True, "uid": "123"}
    assert recorded == {
        "uid": "123",
        "account": "default",
        "mailbox": "INBOX",
        "timeout_s": 11,
        "max_body_chars": 5000,
    }


def test_send_email_delegates_to_core_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: Dict[str, Any] = {}

    def _send_email(**kwargs: Any) -> Dict[str, Any]:
        recorded.update(kwargs)
        return {"success": True, "message_id": "<sent@example.com>"}

    monkeypatch.setattr(
        comms_facade,
        "_load_comms_helpers",
        lambda: (
            lambda: {},
            lambda **_kwargs: {},
            lambda **_kwargs: {},
            _send_email,
        ),
    )

    payload = send_email(
        ["a@example.com"],
        "Hello",
        account="ops",
        body_text="Body",
        cc=["c@example.com"],
        bcc=["b@example.com"],
        timeout_s=13,
        headers={"X-Test": "1"},
    )

    assert payload == {"success": True, "message_id": "<sent@example.com>"}
    assert recorded == {
        "to": ["a@example.com"],
        "subject": "Hello",
        "account": "ops",
        "body_text": "Body",
        "body_html": None,
        "cc": ["c@example.com"],
        "bcc": ["b@example.com"],
        "timeout_s": 13,
        "headers": {"X-Test": "1"},
    }


def test_comms_helpers_return_structured_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        comms_facade,
        "_load_comms_helpers",
        lambda: (_ for _ in ()).throw(RuntimeError("email helper unavailable")),
    )

    assert list_email_accounts() == {
        "success": False,
        "code": "dependency_missing",
        "error": "email helper unavailable",
    }
    assert list_emails() == {
        "success": False,
        "code": "dependency_missing",
        "error": "email helper unavailable",
    }
    assert read_email(uid="123") == {
        "success": False,
        "code": "dependency_missing",
        "error": "email helper unavailable",
    }
    assert send_email(["a@example.com"], "Hello") == {
        "success": False,
        "code": "dependency_missing",
        "error": "email helper unavailable",
    }
