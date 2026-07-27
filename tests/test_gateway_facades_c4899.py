"""The three gateway facades from c4899 (import-boundary fix: their catalog
may not import abstractcore directly — every core surface rides runtime's
facade modules).

1. tool_inventory_facade.capability_tool_facts — re-export of core's
   per-capability fact rows (the discovery facts-join's input; without it
   camera rows derive unvetted/destroy).
2. default_tools.capability_plugin_errors — the plugin_errors slice of
   core's registry status, [] on any failure (a catalog read never raises).
3. default_tools.comms_toolset_kinds — the comms kind->names map, exported
   so the gateway's pinned COPY dies; the toolset composition consumes the
   SAME structure, so the map can never disagree with what registers.
"""

from abstractruntime.integrations.abstractcore.default_tools import (
    _COMMS_KIND_TOOLS,
    capability_plugin_errors,
    comms_toolset_kinds,
)
from abstractruntime.integrations.abstractcore.tool_inventory_facade import (
    capability_tool_facts,
)


class TestCapabilityToolFacts:
    def test_unknown_capability_returns_empty_dict(self):
        assert capability_tool_facts("no-such-capability") == {}

    def test_returns_a_copy(self):
        a = capability_tool_facts("no-such-capability")
        a["poison"] = {"mutating": True}
        assert capability_tool_facts("no-such-capability") == {}


class TestCapabilityPluginErrors:
    def test_shape_is_name_error_rows(self):
        rows = capability_plugin_errors()
        assert isinstance(rows, list)
        for row in rows:
            assert set(row.keys()) == {"name", "error"}
            assert row["name"]


class TestBrowserProbeGrantUniverse:
    """Operator dm#24 (core c5005 ask 2): browser_probe joins the grant
    universe as fetch_url's peer — registered in the web toolset when core
    ships it, ask-by-default in the approval fold (mcd: broker/ask at
    write, never silent-auto; the name entry is the belt beside the fact)."""

    def test_registered_in_web_toolset_when_core_ships_it(self):
        import importlib.util

        from abstractruntime.integrations.abstractcore.default_tools import (
            get_default_toolsets,
        )

        names = [
            getattr(t, "__name__", "")
            for t in get_default_toolsets()["web"]["tools"]
        ]
        if importlib.util.find_spec("abstractcore.tools.browser_tools") is not None:
            assert "browser_probe" in names
            assert names.index("browser_probe") > names.index("fetch_url")
        else:  # pragma: no cover - older core
            assert "browser_probe" not in names

    def test_ask_by_default_beside_fetch_url(self):
        from abstractruntime.integrations.abstractcore.tool_executor import (
            ToolApprovalPolicy,
        )

        p = ToolApprovalPolicy()
        assert "browser_probe" in p.require_approval_tools
        assert "fetch_url" in p.require_approval_tools
        assert p.requires_approval([{"name": "browser_probe", "arguments": {}}])


class TestCommsKindMap:
    def test_three_kinds_with_names(self):
        kinds = comms_toolset_kinds()
        assert set(kinds.keys()) == {"email", "whatsapp", "telegram"}
        assert kinds["email"][0] == "list_email_accounts"
        assert "send_telegram_message" in kinds["telegram"]

    def test_map_returns_a_copy(self):
        kinds = comms_toolset_kinds()
        kinds["email"].append("poison")
        assert "poison" not in comms_toolset_kinds()["email"]

    def test_every_declared_name_is_importable(self):
        """The map IS what composition loads (one source): every name must
        resolve to a callable of that exact __name__ in its module."""
        import importlib

        for _kind, (module_path, names) in _COMMS_KIND_TOOLS.items():
            mod = importlib.import_module(module_path)
            for n in names:
                fn = getattr(mod, n)
                assert callable(fn)
                assert getattr(fn, "__name__", "") == n


class TestPermissionModeLadderFacade:
    """[ladderfacade] (gateway c5065, core name-correction c5069): core's C2
    posture ladder + the fused ceiling+mcd helper re-exported through the
    one import boundary — served words, never client copies."""

    def test_ladder_shape_and_ruled_words(self):
        from abstractruntime.integrations.abstractcore.tool_inventory_facade import (
            permission_mode_ladder,
        )

        lad = permission_mode_ladder()
        assert lad["modes"] == ["read-only", "write", "full-auto"]
        assert set(lad["semantics"].keys()) == set(lad["modes"])
        assert lad["max_auto_rank"] == {"read-only": 1, "write": 2, "full-auto": 4}

    def test_fused_helper_carries_the_mcd_belt(self):
        from abstractruntime.integrations.abstractcore.tool_inventory_facade import (
            permission_mode_auto_approves,
        )

        # The exact footgun: an mcd tool at act(2) is UNDER the write
        # ceiling — the fused helper still asks below full-auto.
        assert not permission_mode_auto_approves(
            "write", risk_rank=2, model_controlled_destination=True
        )
        assert permission_mode_auto_approves(
            "write", risk_rank=2, model_controlled_destination=False
        )
        assert permission_mode_auto_approves(
            "full-auto", risk_rank=2, model_controlled_destination=True
        )
        # Fail-closed: unknown mode uses the read-only ceiling.
        assert not permission_mode_auto_approves(
            "banana", risk_rank=2, model_controlled_destination=False
        )
