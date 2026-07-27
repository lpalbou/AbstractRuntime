"""Camera toolset registration + approval-defaults fold (backlog abstractcamera-0012;
env gate removed by operator ruling 2026-07-21 dm:camera--laurent#10; import lane
moved to core's capability surface by operator ruling 2026-07-22 dm#16-20).

Drafted by seat: camera; owner review: runtime. The ruled contracts:
- LAYERING: runtime NEVER imports abstractcamera (laurent, verbatim: "THE ONLY
  PACKAGE THAT CAN AND SHOULD IMPORT ABSTRACT CAMERA IS ABSTRACT CORE"). The
  camera plugin contributes its tools + approval partition through
  abstractcore.capabilities (register_capability_tools / capability_tools /
  capability_tool_policy); runtime consumes THAT surface. Production code is
  the layering boundary — one test below imports camera's classification as
  GROUND TRUTH to verify the whole chain end-to-end, which is test scaffolding,
  not a runtime dependency.
- INSTALLED = REGISTERED: abstractcamera present beside runtime puts the
  camera toolset in the default registry, like files/web/system. The
  ABSTRACT_ENABLE_CAMERA_TOOLS env var is DEAD — laurent, verbatim: "i don't
  like those stupid variables, remove it! there is a reason why EACH APP can
  decide which tools run, STOP DUPLICATING gating." Exposure control stays
  where apps already have it (allowed_tools / tool_policy / gateway walls).
- Effective approval defaults contain EXACTLY the partition core serves for
  the camera capability (derive-never-copy: the PLUGIN computes it from its
  own classification, core carries it, runtime folds it); when the package is
  absent, no camera name appears in the policy sets. Ask-by-default (ruled
  c3938, a DEFAULT not a floor): every captures_environment tool is in
  require_approval.

Skips cleanly when abstractcamera is not installed (the toolset is optional
by PRESENCE, which is the only gate left).
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
def test_camera_tools_register_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractruntime.integrations.abstractcore.default_tools import (
        camera_tools_available,
        list_default_tool_specs,
    )

    # No env var anywhere: installed is the only condition.
    monkeypatch.delenv("ABSTRACT_ENABLE_CAMERA_TOOLS", raising=False)
    assert camera_tools_available() is True
    names = _tool_names(list_default_tool_specs())
    assert {"camera_list_devices", "camera_open", "camera_capture_photo",
            "camera_preview_photo", "camera_get_events"} <= names

    # The dead flag must stay dead in BOTH polarities (adversary F4: a
    # falsy-only pin misses a kill-switch resurrection that fires on
    # truthy): setting it either way changes NOTHING.
    monkeypatch.setenv("ABSTRACT_ENABLE_CAMERA_TOOLS", "0")
    assert {"camera_open"} <= _tool_names(list_default_tool_specs()), (
        "the removed env var must have no consumer (operator ruling dm#10)"
    )
    monkeypatch.setenv("ABSTRACT_ENABLE_CAMERA_TOOLS", "1")
    assert {"camera_open"} <= _tool_names(list_default_tool_specs()), (
        "a truthy value must be equally meaningless"
    )


def test_runtime_source_has_zero_abstractcamera_imports() -> None:
    """The layering ruling as a pin (laurent dm#16-20): no import statement
    anywhere in runtime's src/ names abstractcamera — the camera lane rides
    core's capability surface exclusively. Grep-grade on purpose: any future
    convenience import regresses this test before it regresses the ruling."""
    import re
    from pathlib import Path

    import abstractruntime

    src_root = Path(abstractruntime.__file__).resolve().parent
    pattern = re.compile(r"^\s*(import abstractcamera\b|from abstractcamera\b)", re.MULTILINE)
    offenders = [
        str(p)
        for p in src_root.rglob("*.py")
        if pattern.search(p.read_text(encoding="utf-8", errors="replace"))
    ]
    assert offenders == [], f"abstractcamera import statements in runtime: {offenders}"


@pytest.mark.skipif(not HAS_ABSTRACTCAMERA, reason="abstractcamera is not installed")
def test_effective_approval_defaults_fold_core_served_partition() -> None:
    """Runtime's fold == base constants ∪ the partition CORE serves for the
    camera capability. Core's surface is runtime's contract seam now; the
    core-served partition == camera's own derivation is pinned camera-side
    (plugin tests), keeping one authority per hop."""
    from abstractcore.capabilities import capability_tool_policy
    from abstractruntime.integrations.abstractcore.default_tools import default_approval_policy_sets
    from abstractruntime.integrations.abstractcore.tool_executor import (
        _DEFAULT_REQUIRE_APPROVAL,
        _DEFAULT_SAFE_AUTO_APPROVE,
    )

    served = capability_tool_policy("camera")
    cam_auto = set(served.get("auto_approve") or [])
    cam_require = set(served.get("require_approval") or [])
    assert cam_auto or cam_require, "installed camera must serve a non-empty partition"

    # Installed: effective == base ∪ core-served partition, exactly.
    auto, require = default_approval_policy_sets()
    assert cam_auto <= auto
    assert cam_require <= require
    assert auto == set(_DEFAULT_SAFE_AUTO_APPROVE) | cam_auto
    assert require == set(_DEFAULT_REQUIRE_APPROVAL) | cam_require


@pytest.mark.skipif(not HAS_ABSTRACTCAMERA, reason="abstractcamera is not installed")
def test_capturing_tools_ask_by_default() -> None:
    """END-TO-END ground truth: every tool camera CLASSIFIES as
    captures_environment lands in runtime's require set. The abstractcamera
    import here is deliberate TEST scaffolding — it verifies the whole
    plugin→core→runtime chain against the authority's own facts; production
    runtime code imports zero abstractcamera (pinned above)."""
    from abstractcamera.integrations.abstractcore_tools import CAMERA_TOOL_CLASSIFICATION
    from abstractruntime.integrations.abstractcore.default_tools import default_approval_policy_sets

    auto, require = default_approval_policy_sets()
    for name, facts in CAMERA_TOOL_CLASSIFICATION.items():
        if facts["captures_environment"]:
            assert name in require, f"{name} records the environment and must require approval"
            assert name not in auto, f"{name} must never auto-approve"


class _StatusRegistryStub:
    """shared_capability_registry() stand-in serving a fixed status()."""

    def __init__(self, plugins_seen=(), plugin_errors=()):
        self._seen = list(plugins_seen)
        self._errors = list(plugin_errors)

    def status(self):
        return {"plugins_seen": self._seen, "plugin_errors": self._errors}


def test_missing_install_degrades_to_no_camera(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    """Absence of the package is the only gate: core serving no camera tools
    (and no policy) with the plugin truly ABSENT (never seen by core's
    entry-point loader — the ruled c4265 detection, not an import probe)
    yields empty camera sets and a camera-free registry — SILENTLY."""
    import logging

    import abstractruntime.integrations.abstractcore.default_tools as dt

    monkeypatch.setattr("abstractcore.capabilities.capability_tools", lambda cap=None: [])
    monkeypatch.setattr("abstractcore.capabilities.capability_tool_policy", lambda cap: {})
    monkeypatch.setattr(
        "abstractcore.capabilities.shared_capability_registry",
        lambda: _StatusRegistryStub(plugins_seen=[{"name": "abstractvoice"}]),
    )
    monkeypatch.setattr(dt, "_camera_import_warned", False)

    with caplog.at_level(logging.WARNING):
        auto, require = dt.camera_approval_sets()
        assert auto == set() and require == set()
        assert "camera" not in dt.get_default_toolsets()
        # The shared predicate answers consistently with the registry it
        # gates (adversary F1: two gates minted an inconsistency window).
        assert dt.camera_tools_available() is False
    assert not [r for r in caplog.records if "#FALLBACK" in r.getMessage()], (
        "true absence is a normal state and must stay silent"
    )


def test_present_but_broken_install_warns_once_with_the_real_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Adversary F1 + core ruling c4265: a PRESENT-but-broken abstractcamera
    used to vanish silently — indistinguishable from absence. Detection now
    reads core's registry status (plugin_errors names the plugin AND carries
    its real load error; an import probe false-positives on a bare directory
    shadowing sys.path). Warn ONCE, with the error text, and degrade."""
    import logging

    import abstractruntime.integrations.abstractcore.default_tools as dt

    monkeypatch.setattr("abstractcore.capabilities.capability_tools", lambda cap=None: [])
    monkeypatch.setattr(
        "abstractcore.capabilities.shared_capability_registry",
        lambda: _StatusRegistryStub(
            plugins_seen=[{"name": "abstractcamera"}],
            plugin_errors=[{"name": "abstractcamera", "error": "boom at load"}],
        ),
    )
    monkeypatch.setattr(dt, "_camera_import_warned", False)
    with caplog.at_level(logging.WARNING):
        assert dt.camera_tools_available() is False
        assert "camera" not in dt.get_default_toolsets()
    warnings = [r for r in caplog.records if "#FALLBACK" in r.getMessage()]
    assert len(warnings) == 1, "present-but-broken must warn exactly once"
    assert "served no camera tools" in warnings[0].getMessage()
    assert "boom at load" in warnings[0].getMessage(), "the plugin's REAL error must ride the warn"


def test_loaded_but_contributionless_plugin_warns_release_skew(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The third shape: the plugin LOADED clean (plugins_seen) but
    contributed nothing (an installed release predating the capability-tools
    contribution). Named in the warn so the operator upgrades instead of
    debugging a phantom."""
    import logging

    import abstractruntime.integrations.abstractcore.default_tools as dt

    monkeypatch.setattr("abstractcore.capabilities.capability_tools", lambda cap=None: [])
    monkeypatch.setattr(
        "abstractcore.capabilities.shared_capability_registry",
        lambda: _StatusRegistryStub(plugins_seen=[{"name": "abstractcamera"}]),
    )
    monkeypatch.setattr(dt, "_camera_import_warned", False)
    with caplog.at_level(logging.WARNING):
        assert dt.camera_tools_available() is False
    warnings = [r for r in caplog.records if "#FALLBACK" in r.getMessage()]
    assert len(warnings) == 1
    assert "predates" in warnings[0].getMessage()


def test_default_constructed_policy_carries_the_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Camera adversary P1-1 (the unpinned load-bearing wire): a DEFAULT-
    constructed ToolApprovalPolicy (no args) must carry camera's partition
    when abstractcamera is installed - reverting the __init__ wiring left
    every test green before this pin. Never skips: when abstractcamera is
    absent the core surface is stubbed (P1-2 - all camera pins skipped in
    CI, so a future regression would ship green)."""
    from abstractruntime.integrations.abstractcore.tool_executor import ToolApprovalPolicy

    if not HAS_ABSTRACTCAMERA:
        monkeypatch.setattr(
            "abstractcore.capabilities.capability_tool_policy",
            lambda cap: {
                "auto_approve": ["camera_status"],
                "require_approval": ["camera_capture_photo"],
            },
        )
        expected_auto = {"camera_status"}
        expected_req = {"camera_capture_photo"}
    else:
        from abstractcore.capabilities import capability_tool_policy

        served = capability_tool_policy("camera")
        expected_auto = set(served.get("auto_approve") or [])
        expected_req = set(served.get("require_approval") or [])

    policy = ToolApprovalPolicy()  # DEFAULT construction - the wire under pin
    assert expected_auto <= policy.auto_approve_tools, "fold reached the default policy"
    assert expected_req <= policy.require_approval_tools

    # Not-installed twin: no camera name anywhere in a default policy.
    monkeypatch.setattr("abstractcore.capabilities.capability_tool_policy", lambda cap: {})
    policy2 = ToolApprovalPolicy()
    camera_names = expected_auto | expected_req
    assert not (camera_names & policy2.auto_approve_tools)
    assert not (camera_names & policy2.require_approval_tools)


def test_policy_fold_contained_to_served_camera_tools(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Adversary P1-2 (escalation lane): the fold unions the served
    auto_approve into the PROCESS-WIDE default policy, so a buggy or
    hostile 'camera' policy naming foreign tools (write_file, an MCP tool)
    would silently auto-approve them everywhere. The partition must be
    scoped to the names the capability actually serves as tools; foreign
    names drop with one #FALLBACK warn."""
    import logging
    import types

    import abstractruntime.integrations.abstractcore.default_tools as dt
    from abstractruntime.integrations.abstractcore.tool_executor import ToolApprovalPolicy

    served = [
        types.SimpleNamespace(name="camera_status", function=lambda: None),
        types.SimpleNamespace(name="camera_capture_photo", function=lambda: None),
    ]
    monkeypatch.setattr("abstractcore.capabilities.capability_tools", lambda cap=None: list(served))
    monkeypatch.setattr(
        "abstractcore.capabilities.capability_tool_policy",
        lambda cap: {
            "auto_approve": ["camera_status", "write_file", "mcp::fs::delete_tree"],
            "require_approval": ["camera_capture_photo", "some_foreign_tool"],
        },
    )
    monkeypatch.setattr(dt, "_camera_policy_scope_warned", False)

    with caplog.at_level(logging.WARNING):
        auto, require = dt.camera_approval_sets()
    assert auto == {"camera_status"}, "foreign names must never ride into auto_approve"
    assert require == {"camera_capture_photo"}
    warnings = [r for r in caplog.records if "#FALLBACK" in r.getMessage()]
    assert len(warnings) == 1 and "write_file" in warnings[0].getMessage()

    # End to end: the escalation targets stay approval-gated in a default
    # policy (default-deny for unlisted names).
    policy = ToolApprovalPolicy()
    assert policy.requires_approval([{"name": "mcp::fs::delete_tree"}])
    assert policy.requires_approval([{"name": "write_file"}])


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
