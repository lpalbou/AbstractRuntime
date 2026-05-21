"""Runtime-owned Telegram wrapper surface for AbstractCore-backed hosts.

Hosts should import this module instead of reaching into
`abstractcore.tools.telegram_*` directly.

Scope:
- one-shot TDLib auth bootstrap from env
- stable TDLib "not available" error for host integrations
- global TDLib client lifecycle access
- thin Telegram send helper for notifier parity

Non-goals:
- durable Runtime tool execution
- run-scoped Runtime history
- Telegram artifact send helpers
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Type


class TelegramTdlibNotAvailable(RuntimeError):
    """Stable Runtime-owned error for missing TDLib support."""


def _load_tdlib_components() -> Tuple[Type[Any], Type[Any], Type[BaseException], Callable[..., Any], Callable[[], None]]:
    try:
        from abstractcore.tools.telegram_tdlib import (
            TdlibClient,
            TdlibConfig,
            TdlibNotAvailable,
            get_global_tdlib_client,
            stop_global_tdlib_client,
        )
    except Exception as exc:  # pragma: no cover - exercised through facade behavior tests
        raise TelegramTdlibNotAvailable(
            "Telegram TDLib helpers are unavailable. Install a Runtime environment with "
            "AbstractCore's Telegram TDLib support and configure TDLib."
        ) from exc

    return (
        TdlibConfig,
        TdlibClient,
        TdlibNotAvailable,
        get_global_tdlib_client,
        stop_global_tdlib_client,
    )


def _load_telegram_send_helper() -> Callable[..., Dict[str, Any]]:
    try:
        from abstractcore.tools.telegram_tools import send_telegram_message
    except Exception as exc:  # pragma: no cover - exercised through facade behavior tests
        raise RuntimeError(
            "Telegram send helpers are unavailable. Install a Runtime environment with "
            "AbstractCore's Telegram tools support."
        ) from exc
    return send_telegram_message


def bootstrap_telegram_auth_from_env(
    *,
    login_code: Optional[str] = None,
    two_factor_password: Optional[str] = None,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Run one TDLib auth bootstrap using the current env configuration.

    This creates a temporary TDLib client, waits for authorization readiness,
    then stops the client. The resulting TDLib session stays on disk in the
    configured database directory.
    """

    TdlibConfig, TdlibClient, TdlibNotAvailable, _get_global, _stop_global = _load_tdlib_components()

    try:
        base_cfg = TdlibConfig.from_env()
    except Exception as exc:
        raise ValueError(f"Missing/invalid Telegram env config: {exc}") from exc

    cfg = TdlibConfig(
        api_id=base_cfg.api_id,
        api_hash=base_cfg.api_hash,
        phone=base_cfg.phone,
        database_directory=base_cfg.database_directory,
        files_directory=base_cfg.files_directory,
        database_encryption_key=base_cfg.database_encryption_key,
        use_secret_chats=base_cfg.use_secret_chats,
        login_code=login_code or base_cfg.login_code,
        two_factor_password=two_factor_password or base_cfg.two_factor_password,
    )

    try:
        client = TdlibClient(config=cfg)
    except TdlibNotAvailable as exc:
        raise TelegramTdlibNotAvailable(str(exc)) from exc

    client.start()
    try:
        ok = client.wait_until_ready(timeout_s=float(timeout_s))
        if ok:
            return {
                "success": True,
                "ready": True,
                "database_directory": cfg.database_directory,
                "files_directory": cfg.files_directory,
                "use_secret_chats": bool(cfg.use_secret_chats),
                "phone": cfg.phone,
            }
        return {
            "success": False,
            "ready": False,
            "database_directory": cfg.database_directory,
            "files_directory": cfg.files_directory,
            "use_secret_chats": bool(cfg.use_secret_chats),
            "phone": cfg.phone,
            "error": client.last_error or "Timed out waiting for TDLib authorization",
        }
    finally:
        try:
            client.stop()
        except Exception:
            pass


def get_global_telegram_client(*, start: bool = False) -> Any:
    """Return the process-global TDLib client used by host bridges."""

    _TdlibConfig, _TdlibClient, TdlibNotAvailable, get_global_tdlib_client, _stop_global_tdlib_client = _load_tdlib_components()

    try:
        return get_global_tdlib_client(start=start)
    except TdlibNotAvailable as exc:
        raise TelegramTdlibNotAvailable(str(exc)) from exc


def stop_global_telegram_client() -> None:
    """Stop the process-global TDLib client, if one exists."""

    try:
        _TdlibConfig, _TdlibClient, TdlibNotAvailable, _get_global_tdlib_client, stop_global_tdlib_client = _load_tdlib_components()
    except TelegramTdlibNotAvailable:
        return

    try:
        stop_global_tdlib_client()
    except TdlibNotAvailable:
        return


def send_telegram_message(
    *,
    chat_id: int,
    text: str,
    parse_mode: str = "",
    disable_web_page_preview: bool = False,
    timeout_s: float = 20.0,
    bot_token_env_var: str = "ABSTRACT_TELEGRAM_BOT_TOKEN",
) -> Dict[str, Any]:
    """Send one Telegram message via the current AbstractCore helper.

    This is a host-local operator/notifier helper. If the outbound send belongs
    to a workflow/run, use
    `get_abstractcore_run_facade(runtime).send_telegram_message(...)` instead
    so Runtime authors the durable child-run truth.
    """

    try:
        helper = _load_telegram_send_helper()
    except Exception as exc:
        return {
            "success": False,
            "code": "dependency_missing",
            "error": str(exc),
        }

    return helper(
        chat_id=chat_id,
        text=text,
        parse_mode=parse_mode,
        disable_web_page_preview=disable_web_page_preview,
        timeout_s=timeout_s,
        bot_token_env_var=bot_token_env_var,
    )


__all__ = [
    "TelegramTdlibNotAvailable",
    "bootstrap_telegram_auth_from_env",
    "get_global_telegram_client",
    "stop_global_telegram_client",
    "send_telegram_message",
]
