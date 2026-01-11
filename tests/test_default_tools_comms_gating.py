from __future__ import annotations

import pytest


def _tool_names(specs: list[object]) -> set[str]:
    out: set[str] = set()
    for s in specs:
        if not isinstance(s, dict):
            continue
        name = s.get("name")
        if isinstance(name, str) and name.strip():
            out.add(name.strip())
    return out


def test_comms_tools_are_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractruntime.integrations.abstractcore.default_tools import list_default_tool_specs

    for k in (
        "ABSTRACT_ENABLE_COMMS_TOOLS",
        "ABSTRACT_ENABLE_EMAIL_TOOLS",
        "ABSTRACT_ENABLE_WHATSAPP_TOOLS",
        "ABSTRACT_ENABLE_TELEGRAM_TOOLS",
    ):
        monkeypatch.delenv(k, raising=False)

    specs = list_default_tool_specs()
    names = _tool_names(specs)
    assert "send_email" not in names
    assert "send_whatsapp_message" not in names
    assert "send_telegram_message" not in names

    monkeypatch.setenv("ABSTRACT_ENABLE_COMMS_TOOLS", "1")
    specs2 = list_default_tool_specs()
    names2 = _tool_names(specs2)
    assert "send_email" in names2
    assert "list_emails" in names2
    assert "read_email" in names2
    assert "send_whatsapp_message" in names2
    assert "list_whatsapp_messages" in names2
    assert "read_whatsapp_message" in names2
    assert "send_telegram_message" in names2
    assert "send_telegram_artifact" in names2
