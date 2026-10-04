"""The agent's workspace context names the host's shared workspace and each allowed workspace with its mode."""
import os
from pathlib import Path

import pytest

from abstractruntime.integrations.abstractcore.workspace_scoped_tools import WorkspaceScope, describe_workspace_scope
from abstractruntime.utils.workspace_paths import (
    SHARED_WORKSPACE_KEY,
    merge_builtin_workspace_protection,
    shared_workspace_path,
)

pytestmark = pytest.mark.basic


def _dirs(tmp_path: Path) -> dict:
    out = {}
    for name in ("own", "shared", "project", "archive", "secrets"):
        p = tmp_path / name
        p.mkdir()
        out[name] = os.path.realpath(p)
    return out


def test_allowed_posture_lists_shared_then_allowed_with_modes(tmp_path):
    d = _dirs(tmp_path)
    vars0 = {
        "workspace_root": d["own"],
        "workspace_access_mode": "workspace_or_allowed",
        "workspace_allowed_paths": [d["shared"], d["project"], d["archive"]],
        "workspace_ignored_paths": d["secrets"],
        "workspace_read_only_paths": [d["archive"]],
        SHARED_WORKSPACE_KEY: d["shared"],
    }
    lines = describe_workspace_scope(WorkspaceScope.from_input_data(vars0)).splitlines()
    assert lines[1] == f'Default working directory: "{d["own"]}"'
    assert lines[2] == f'Shared workspace: "{d["shared"]}" (read & write)'
    assert lines[3] == "Allowed workspaces:"
    assert f'  "{d["project"]}" (read & write)' in lines
    assert f'  "{d["archive"]}" (read-only)' in lines
    assert not any(ln.startswith("  ") and d["shared"] in ln and "(read" in ln for ln in lines)
    assert not any(d["secrets"] in ln and "(read" in ln for ln in lines)
    assert "Everything else" not in "\n".join(lines)


def test_allow_everything_with_read_only_default_names_everything_else(tmp_path):
    d = _dirs(tmp_path)
    vars0 = {
        "workspace_root": d["own"],
        "workspace_access_mode": "all_except_ignored",
        "workspace_allowed_paths": [d["shared"], d["project"]],
        "workspace_read_only_paths": ["/"],
        "workspace_writable_paths": [d["own"], d["shared"], d["project"]],
        SHARED_WORKSPACE_KEY: d["shared"],
    }
    lines = describe_workspace_scope(WorkspaceScope.from_input_data(vars0)).splitlines()
    assert f'Shared workspace: "{d["shared"]}" (read & write)' in lines
    assert f'  "{d["project"]}" (read & write)' in lines
    assert "Everything else: (read-only)" in lines


def test_without_the_key_nothing_is_invented(tmp_path):
    d = _dirs(tmp_path)
    text = describe_workspace_scope(WorkspaceScope.from_input_data({"workspace_root": d["own"]}))
    assert "Shared workspace" not in text and "Allowed workspaces" not in text


def test_a_child_carries_the_parents_shared_workspace_exactly(tmp_path):
    d = _dirs(tmp_path)
    parent = {SHARED_WORKSPACE_KEY: d["shared"]}
    child = {SHARED_WORKSPACE_KEY: d["secrets"]}
    out = merge_builtin_workspace_protection(parent, child)
    assert out[SHARED_WORKSPACE_KEY] == d["shared"]
    assert shared_workspace_path({"_runtime": {SHARED_WORKSPACE_KEY: d["shared"]}}) == d["shared"]
    assert shared_workspace_path({SHARED_WORKSPACE_KEY: "relative/path"}) is None
