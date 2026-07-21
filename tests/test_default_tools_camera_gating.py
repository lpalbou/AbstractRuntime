"""Camera toolset gating + approval-defaults fold (backlog abstractcamera-0012).

Drafted by seat: camera; owner review: runtime (commons c3826/c3829). Pins
the two contracts runtime asked for:
- EXPLICIT env gate (ABSTRACT_ENABLE_CAMERA_TOOLS), never key-implied.
- Effective approval defaults, when enabled, contain EXACTLY abstractcamera's
  own derived partition (import, never copy); when disabled, no camera name
  appears in the policy sets. Ask-by-default (ruled c3938, a DEFAULT not a floor): every captures_environment tool
  is in require_approval, never auto_approve.

Skips cleanly when abstractcamera is not installed (the toolset is optional).
"""

from __future__ import annotations

import importlib.util

import pytest

HAS_ABSTRACTCAMERA = importlib.util.find_spec("abstractcamera") is not None


def _tool_names(specs: list[object]) -> set[str]:
    out: set[str] = set()
    for s in specs:
        if isinstance(s, dict) and isinstance(s.get("name"), str) and s["name"].strip():
            out.add(s["name"].strip())
    return out


@pytest.mark.skipif(not HAS_ABSTRACTCAMERA, reason="abstractcamera is not installed")
def test_camera_tools_are_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractruntime.integrations.abstractcore.default_tools import (
        camera_tools_enabled,
        list_default_tool_specs,
    )

    monkeypatch.delenv("ABSTRACT_ENABLE_CAMERA_TOOLS", raising=False)
    # Not key-implied: even with an agora key present, camera stays off.
    monkeypatch.setenv("AGORA_API_KEY", "test-key")
    assert camera_tools_enabled() is False
    assert not (_tool_names(list_default_tool_specs()) & {"camera_open", "camera_capture_photo"})

    monkeypatch.setenv("ABSTRACT_ENABLE_CAMERA_TOOLS", "1")
    assert camera_tools_enabled() is True
    names = _tool_names(list_default_tool_specs())
    assert {"camera_list_devices", "camera_open", "camera_capture_photo", "camera_get_events"} <= names


@pytest.mark.skipif(not HAS_ABSTRACTCAMERA, reason="abstractcamera is not installed")
def test_effective_approval_defaults_fold_camera_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcamera.integrations.abstractcore_tools import camera_tool_approval_defaults
    from abstractruntime.integrations.abstractcore.default_tools import default_approval_policy_sets
    from abstractruntime.integrations.abstractcore.tool_executor import (
        _DEFAULT_REQUIRE_APPROVAL,
        _DEFAULT_SAFE_AUTO_APPROVE,
    )

    partition = camera_tool_approval_defaults()
    cam_auto = set(partition["auto_approve"])
    cam_require = set(partition["require_approval"])

    # Disabled: no camera name anywhere; effective == base.
    monkeypatch.delenv("ABSTRACT_ENABLE_CAMERA_TOOLS", raising=False)
    auto, require = default_approval_policy_sets()
    assert auto == set(_DEFAULT_SAFE_AUTO_APPROVE)
    assert require == set(_DEFAULT_REQUIRE_APPROVAL)
    assert not (auto & (cam_auto | cam_require))
    assert not (require & (cam_auto | cam_require))

    # Enabled: effective == base ∪ camera's own derived partition, exactly.
    monkeypatch.setenv("ABSTRACT_ENABLE_CAMERA_TOOLS", "1")
    auto, require = default_approval_policy_sets()
    assert cam_auto <= auto
    assert cam_require <= require
    assert auto == set(_DEFAULT_SAFE_AUTO_APPROVE) | cam_auto
    assert require == set(_DEFAULT_REQUIRE_APPROVAL) | cam_require


@pytest.mark.skipif(not HAS_ABSTRACTCAMERA, reason="abstractcamera is not installed")
def test_capturing_tools_ask_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcamera.integrations.abstractcore_tools import CAMERA_TOOL_CLASSIFICATION
    from abstractruntime.integrations.abstractcore.default_tools import default_approval_policy_sets

    monkeypatch.setenv("ABSTRACT_ENABLE_CAMERA_TOOLS", "1")
    auto, require = default_approval_policy_sets()
    for name, facts in CAMERA_TOOL_CLASSIFICATION.items():
        if facts["captures_environment"]:
            assert name in require, f"{name} records the environment and must require approval"
            assert name not in auto, f"{name} must never auto-approve"


def test_enabled_without_install_raises_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    # When the flag is set but the import path is broken, camera_approval_sets
    # must fail loudly with install guidance, never silently no-op.
    import abstractruntime.integrations.abstractcore.default_tools as dt

    monkeypatch.setenv("ABSTRACT_ENABLE_CAMERA_TOOLS", "1")

    real_import = __import__

    def _boom(name, *args, **kwargs):
        if name == "abstractcamera.integrations.abstractcore_tools":
            raise ImportError("simulated missing abstractcamera")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _boom)
    with pytest.raises(RuntimeError, match="abstractcamera is not installed"):
        dt.camera_approval_sets()


def test_default_constructed_policy_carries_the_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Camera adversary P1-1 (the unpinned load-bearing wire): a DEFAULT-
    constructed ToolApprovalPolicy (no args) must carry camera's partition
    when the toolset is enabled - reverting the __init__ wiring left every
    test green before this pin. Never skips: when abstractcamera is absent
    the fold is stubbed (P1-2 - all camera pins skipped in CI, so a future
    regression would ship green)."""
    import sys
    import types

    from abstractruntime.integrations.abstractcore.tool_executor import ToolApprovalPolicy

    monkeypatch.setenv("ABSTRACT_ENABLE_CAMERA_TOOLS", "1")
    if not HAS_ABSTRACTCAMERA:
        stub_tools = types.ModuleType("abstractcamera.integrations.abstractcore_tools")

        def camera_tool_approval_defaults():
            return {
                "auto_approve": ["camera_status"],
                "require_approval": ["camera_capture_photo"],
            }

        stub_tools.camera_tool_approval_defaults = camera_tool_approval_defaults
        stub_pkg = types.ModuleType("abstractcamera")
        stub_int = types.ModuleType("abstractcamera.integrations")
        monkeypatch.setitem(sys.modules, "abstractcamera", stub_pkg)
        monkeypatch.setitem(sys.modules, "abstractcamera.integrations", stub_int)
        monkeypatch.setitem(sys.modules, "abstractcamera.integrations.abstractcore_tools", stub_tools)
        expected_auto = {"camera_status"}
        expected_req = {"camera_capture_photo"}
    else:
        from abstractcamera.integrations.abstractcore_tools import (
            camera_tool_approval_defaults,
        )

        defaults = camera_tool_approval_defaults()
        expected_auto = set(defaults.get("auto_approve") or [])
        expected_req = set(defaults.get("require_approval") or [])

    policy = ToolApprovalPolicy()  # DEFAULT construction - the wire under pin
    assert expected_auto <= policy.auto_approve_tools, "fold reached the default policy"
    assert expected_req <= policy.require_approval_tools

    # Disabled twin: no camera name anywhere in a default policy.
    monkeypatch.setenv("ABSTRACT_ENABLE_CAMERA_TOOLS", "0")
    policy2 = ToolApprovalPolicy()
    camera_names = expected_auto | expected_req
    assert not (camera_names & policy2.auto_approve_tools)
    assert not (camera_names & policy2.require_approval_tools)


def test_fold_failure_degrades_loud_and_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Camera adversary P2-1: a broken fold logs and falls back to the BASE
    constants (stricter - default-deny asks for unlisted names); it never
    raises (bundle_host's fallback would be the UNGATED executor)."""
    import abstractruntime.integrations.abstractcore.tool_executor as te

    def _boom():
        raise RuntimeError("simulated fold failure")

    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.default_tools.default_approval_policy_sets",
        _boom,
    )
    policy = te.ToolApprovalPolicy()  # must not raise
    assert policy.auto_approve_tools == set(te._DEFAULT_SAFE_AUTO_APPROVE)
    assert policy.require_approval_tools == set(te._DEFAULT_REQUIRE_APPROVAL)
    # Default-deny holds: an unknown (camera) name asks.
    assert policy.requires_approval([{"name": "camera_capture_photo"}])
