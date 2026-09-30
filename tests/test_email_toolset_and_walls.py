"""Email tool availability follows the host's decision, and email file arguments ride the wall
(framework backlog 0992 B2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from abstractruntime.integrations.abstractcore.default_tools import (
    EMAIL_ACCOUNT_GATE,
    build_default_tool_map,
    comms_toolset_kinds,
    list_default_tool_specs,
    list_tool_catalog,
)
from abstractruntime.integrations.abstractcore.tool_effects import TOOL_EFFECT_CLASSES
from abstractruntime.integrations.abstractcore.workspace_scoped_tools import WorkspaceScope, rewrite_tool_arguments

pytestmark = pytest.mark.basic

EMAIL_TOOLS = {
    "list_email_accounts", "send_email", "reply_email", "list_emails", "search_emails", "read_email", "get_email_attachment",
}


@pytest.fixture(autouse=True)
def _no_env_flags(monkeypatch):
    for k in ("ABSTRACT_ENABLE_COMMS_TOOLS", "ABSTRACT_ENABLE_EMAIL_TOOLS", "ABSTRACT_ENABLE_WHATSAPP_TOOLS",
              "ABSTRACT_ENABLE_TELEGRAM_TOOLS"):
        monkeypatch.delenv(k, raising=False)


def _names(specs):
    return {s["name"] for s in specs if isinstance(s, dict)}


def test_the_email_kind_is_the_full_tool_set():
    assert set(comms_toolset_kinds()["email"]) == EMAIL_TOOLS
    assert EMAIL_TOOLS <= set(TOOL_EFFECT_CLASSES)
    assert TOOL_EFFECT_CLASSES["get_email_attachment"] == "write"


def test_host_decides_email_availability_without_env_flags():
    assert not (EMAIL_TOOLS & _names(list_default_tool_specs()))  # default: off
    assert EMAIL_TOOLS <= _names(list_default_tool_specs(email_enabled=True))
    assert EMAIL_TOOLS <= set(build_default_tool_map(email_enabled=True))
    names = _names(list_default_tool_specs(email_enabled=True))
    assert "send_whatsapp_message" not in names and "send_telegram_message" not in names  # only email turns on
    assert not (EMAIL_TOOLS & _names(list_default_tool_specs(email_enabled=False)))


def test_disabled_email_row_names_the_account_fix():
    rows = {r["id"]: r for r in list_tool_catalog(email_enabled=False)}
    assert rows["comms.email"]["enabled"] is False and rows["comms.email"]["gate"] == EMAIL_ACCOUNT_GATE
    assert {getattr(t, "__name__", "") for t in rows["comms.email"]["tools"]} == EMAIL_TOOLS
    default = {r["id"]: r for r in list_tool_catalog()}  # no host decision: off, no env flag
    assert default["comms.email"]["enabled"] is False and default["comms.email"]["gate"] == EMAIL_ACCOUNT_GATE


def _scope(tmp_path: Path) -> WorkspaceScope:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return WorkspaceScope.from_input_data({"workspace_root": str(ws)})


@pytest.mark.parametrize("tool", ["send_email", "reply_email"])
def test_attachments_cannot_carry_files_out_of_the_workspace(tool, tmp_path):
    scope = _scope(tmp_path)
    with pytest.raises(ValueError):
        rewrite_tool_arguments(tool_name=tool, args={"to": "a@example.test", "attachments": ["/etc/hosts"]}, scope=scope)
    with pytest.raises(ValueError):
        rewrite_tool_arguments(tool_name=tool, args={"attachments": "../../outside.txt"}, scope=scope)
    out = rewrite_tool_arguments(tool_name=tool, args={"attachments": ["report.pdf"]}, scope=scope)
    assert out["attachments"] == [str((tmp_path / "ws" / "report.pdf").resolve())]


def test_attachment_downloads_land_in_the_workspace(tmp_path):
    scope = _scope(tmp_path)
    with pytest.raises(ValueError):
        rewrite_tool_arguments(tool_name="get_email_attachment", args={"uid": "1", "index": 0, "output_dir": "/tmp"}, scope=scope)
    out = rewrite_tool_arguments(tool_name="get_email_attachment", args={"uid": "1", "index": 0}, scope=scope)
    assert out["output_dir"] == str((tmp_path / "ws").resolve())
