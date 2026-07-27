"""list_tool_catalog pins (tool-tiers item H / G-1 structural fix; gateway
c4562 seam): exists-but-not-enabled is a VISIBLE state - disabled toolsets
serve real callables + the governing gate, never silence."""
from __future__ import annotations

from abstractruntime.integrations.abstractcore.default_tools import list_tool_catalog


def test_catalog_serves_disabled_toolsets_with_real_specs(monkeypatch) -> None:
    for var in ("ABSTRACT_ENABLE_COMMS_TOOLS", "ABSTRACT_ENABLE_EMAIL_TOOLS",
                "ABSTRACT_ENABLE_TELEGRAM_TOOLS", "ABSTRACT_ENABLE_WHATSAPP_TOOLS",
                "ABSTRACT_ENABLE_AGORA_TOOLS", "ABSTRACT_ENABLE_SHELL_TOOLS",
                "AGORA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cat = {row["id"]: row for row in list_tool_catalog()}

    # laurent's named suspects (dm#221): email/telegram EXIST and are
    # VISIBLE - per-channel rows (gateway c4573 gap: one aggregate row
    # made partially-disabled channels vanish from both lanes).
    for cid, probe in (("comms.email", "send_email"),
                       ("comms.whatsapp", "send_whatsapp_message"),
                       ("comms.telegram", "send_telegram_message")):
        row = cat[cid]
        assert row["enabled"] is False
        assert "ABSTRACT_ENABLE_COMMS_TOOLS" in row["gate"], "the gate is named"
        names = {getattr(getattr(f, "_tool_definition", None), "name", getattr(f, "__name__", "")) for f in row["tools"]}
        assert probe in names, f"real callables ride the disabled row {cid}: {names}"

    agora = cat["agora"]
    assert agora["enabled"] is False and "AGORA_API_KEY" in agora["gate"], \
        "the AND-gate (intent+key) is the stated gate"
    assert len(agora["tools"]) >= 7

    # Always-on sets stay enabled rows.
    assert cat["files"]["enabled"] is True and cat["files"]["gate"] == "always"


def test_catalog_enabled_rows_match_default_toolsets(monkeypatch) -> None:
    """The catalog's enabled rows ARE get_default_toolsets - one source,
    no forked assembly (the gateway fold deletes its own on this pin)."""
    from abstractruntime.integrations.abstractcore.default_tools import get_default_toolsets

    enabled = {row["id"] for row in list_tool_catalog() if row["enabled"]}
    assert enabled == set(get_default_toolsets().keys())


def test_catalog_include_disabled_false_is_enabled_only() -> None:
    rows = list_tool_catalog(include_disabled=False)
    assert all(r["enabled"] for r in rows)


def test_catalog_partial_comms_enablement_never_vanishes(monkeypatch) -> None:
    """gateway c4573 (the never-neither break): email ON via its per-channel
    flag, whatsapp/telegram OFF - the enabled comms row carries email tools
    AND the missing channels still appear as their own disabled rows."""
    for var in ("ABSTRACT_ENABLE_COMMS_TOOLS", "ABSTRACT_ENABLE_TELEGRAM_TOOLS",
                "ABSTRACT_ENABLE_WHATSAPP_TOOLS", "ABSTRACT_ENABLE_AGORA_TOOLS",
                "ABSTRACT_ENABLE_SHELL_TOOLS", "AGORA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ABSTRACT_ENABLE_EMAIL_TOOLS", "1")
    cat = {row["id"]: row for row in list_tool_catalog()}
    assert cat["comms"]["enabled"] is True, "email channel rides the enabled row"
    assert "comms.email" not in cat, "an enabled channel never doubles as disabled"
    assert cat["comms.whatsapp"]["enabled"] is False
    assert cat["comms.telegram"]["enabled"] is False, "telegram VISIBLE despite partial enablement"


def test_camera_row_visible_even_when_uninstalled(monkeypatch) -> None:
    """gateway c4630: camera must appear as a catalog row with its install
    gate even when abstractcamera is not registered in this process -
    'exists but not surfaced' is the class the audit fixes. Env-gated rows
    (comms) show specs when disabled; install-gated camera shows the ROW +
    install path with a specless note when the package is absent."""
    import abstractruntime.integrations.abstractcore.default_tools as dt

    # Simulate camera absent: capability surface serves zero camera tools.
    monkeypatch.setattr(dt, "_camera_capability_tools", lambda: [])
    monkeypatch.setattr(dt, "camera_tools_available", lambda: False)

    cat = {row["id"]: row for row in dt.list_tool_catalog()}
    assert "camera" in cat, "camera is a VISIBLE row even uninstalled"
    cam = cat["camera"]
    assert cam["enabled"] is False
    assert "abstractcamera" in cam["gate"], "the install path is named"
    assert cam["tools"] == [], "no specs to serve until installed (honest)"
    assert "not installed" in (cam.get("note") or ""), "the WHY is stated, never silence"
