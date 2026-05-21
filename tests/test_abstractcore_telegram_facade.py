from __future__ import annotations

from typing import Any, Dict

import pytest

from abstractruntime.integrations import abstractcore
from abstractruntime.integrations.abstractcore import telegram_facade
from abstractruntime.integrations.abstractcore.telegram_facade import (
    TelegramTdlibNotAvailable,
    bootstrap_telegram_auth_from_env,
    get_global_telegram_client,
    send_telegram_message,
    stop_global_telegram_client,
)


class _FakeUpstreamTdlibNotAvailable(RuntimeError):
    pass


class _FakeTdlibConfig:
    def __init__(
        self,
        *,
        api_id: int,
        api_hash: str,
        phone: str,
        database_directory: str,
        files_directory: str,
        database_encryption_key: str = "",
        use_secret_chats: bool = True,
        login_code: str | None = None,
        two_factor_password: str | None = None,
    ) -> None:
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.database_directory = database_directory
        self.files_directory = files_directory
        self.database_encryption_key = database_encryption_key
        self.use_secret_chats = use_secret_chats
        self.login_code = login_code
        self.two_factor_password = two_factor_password

    @staticmethod
    def from_env() -> "_FakeTdlibConfig":
        return _FakeTdlibConfig(
            api_id=123,
            api_hash="hash-1",
            phone="+33123456789",
            database_directory="/tmp/tdlib-db",
            files_directory="/tmp/tdlib-files",
            database_encryption_key="sekret",
            use_secret_chats=True,
            login_code=None,
            two_factor_password=None,
        )


class _FakeTdlibClient:
    last_created: "_FakeTdlibClient | None" = None
    wait_result = True
    last_error_value: str | None = None

    def __init__(self, *, config: _FakeTdlibConfig) -> None:
        self.config = config
        self.started = False
        self.stopped = False
        self.last_error = self.last_error_value
        type(self).last_created = self

    def start(self) -> None:
        self.started = True

    def wait_until_ready(self, *, timeout_s: float = 30.0) -> bool:
        self.timeout_s = timeout_s
        return bool(type(self).wait_result)

    def stop(self) -> None:
        self.stopped = True


def _fake_tdlib_components(
    *,
    get_client: Any = None,
    stop_client: Any = None,
) -> tuple[Any, Any, Any, Any, Any]:
    return (
        _FakeTdlibConfig,
        _FakeTdlibClient,
        _FakeUpstreamTdlibNotAvailable,
        get_client or (lambda *, start=False: {"start": start}),
        stop_client or (lambda: None),
    )


def test_public_telegram_facade_exports_are_available() -> None:
    assert abstractcore.TelegramTdlibNotAvailable is TelegramTdlibNotAvailable
    assert abstractcore.bootstrap_telegram_auth_from_env is bootstrap_telegram_auth_from_env
    assert abstractcore.get_global_telegram_client is get_global_telegram_client
    assert abstractcore.stop_global_telegram_client is stop_global_telegram_client
    assert abstractcore.send_telegram_message is send_telegram_message
    assert "TelegramTdlibNotAvailable" in abstractcore.__all__
    assert "bootstrap_telegram_auth_from_env" in abstractcore.__all__
    assert "get_global_telegram_client" in abstractcore.__all__
    assert "stop_global_telegram_client" in abstractcore.__all__
    assert "send_telegram_message" in abstractcore.__all__
    assert "TelegramTdlibNotAvailable" in telegram_facade.__all__
    assert "bootstrap_telegram_auth_from_env" in telegram_facade.__all__
    assert "get_global_telegram_client" in telegram_facade.__all__
    assert "stop_global_telegram_client" in telegram_facade.__all__
    assert "send_telegram_message" in telegram_facade.__all__


def test_bootstrap_telegram_auth_from_env_returns_ready_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeTdlibClient.wait_result = True
    _FakeTdlibClient.last_error_value = None
    _FakeTdlibClient.last_created = None
    monkeypatch.setattr(telegram_facade, "_load_tdlib_components", lambda: _fake_tdlib_components())

    payload = bootstrap_telegram_auth_from_env(
        login_code="12345",
        two_factor_password="pw-1",
        timeout_s=9,
    )

    created = _FakeTdlibClient.last_created
    assert created is not None
    assert payload == {
        "success": True,
        "ready": True,
        "database_directory": "/tmp/tdlib-db",
        "files_directory": "/tmp/tdlib-files",
        "use_secret_chats": True,
        "phone": "+33123456789",
    }
    assert created.started is True
    assert created.stopped is True
    assert created.timeout_s == 9.0
    assert created.config.login_code == "12345"
    assert created.config.two_factor_password == "pw-1"


def test_bootstrap_telegram_auth_from_env_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeTdlibClient.wait_result = False
    _FakeTdlibClient.last_error_value = "authorization incomplete"
    _FakeTdlibClient.last_created = None
    monkeypatch.setattr(telegram_facade, "_load_tdlib_components", lambda: _fake_tdlib_components())

    payload = bootstrap_telegram_auth_from_env(timeout_s=5)

    created = _FakeTdlibClient.last_created
    assert created is not None
    assert payload == {
        "success": False,
        "ready": False,
        "database_directory": "/tmp/tdlib-db",
        "files_directory": "/tmp/tdlib-files",
        "use_secret_chats": True,
        "phone": "+33123456789",
        "error": "authorization incomplete",
    }
    assert created.stopped is True


def test_get_global_telegram_client_re_raises_stable_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*, start: bool = False) -> Any:
        raise _FakeUpstreamTdlibNotAvailable("tdjson missing")

    monkeypatch.setattr(
        telegram_facade,
        "_load_tdlib_components",
        lambda: _fake_tdlib_components(get_client=_raise),
    )

    with pytest.raises(TelegramTdlibNotAvailable, match="tdjson missing"):
        get_global_telegram_client(start=True)


def test_stop_global_telegram_client_is_noop_when_tdlib_support_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        telegram_facade,
        "_load_tdlib_components",
        lambda: (_ for _ in ()).throw(TelegramTdlibNotAvailable("missing tdlib")),
    )

    stop_global_telegram_client()


def test_send_telegram_message_delegates_to_core_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: Dict[str, Any] = {}

    def _send(**kwargs: Any) -> Dict[str, Any]:
        recorded.update(kwargs)
        return {"success": True, "message_ids": [7]}

    monkeypatch.setattr(telegram_facade, "_load_telegram_send_helper", lambda: _send)

    payload = send_telegram_message(
        chat_id=123,
        text="Status green",
        parse_mode="Markdown",
        disable_web_page_preview=True,
        timeout_s=11,
        bot_token_env_var="TG_TOKEN",
    )

    assert payload == {"success": True, "message_ids": [7]}
    assert recorded == {
        "chat_id": 123,
        "text": "Status green",
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
        "timeout_s": 11,
        "bot_token_env_var": "TG_TOKEN",
    }


def test_send_telegram_message_returns_structured_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        telegram_facade,
        "_load_telegram_send_helper",
        lambda: (_ for _ in ()).throw(RuntimeError("telegram helper unavailable")),
    )

    payload = send_telegram_message(chat_id=123, text="hello")

    assert payload == {
        "success": False,
        "code": "dependency_missing",
        "error": "telegram helper unavailable",
    }
