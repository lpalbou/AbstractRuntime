"""The agent's workspace context lists each allowed workspace with its mode (round 11: there is no
shared workspace; the run's private workspace is the default working directory)."""
import os
from pathlib import Path

import pytest

import abstractruntime.utils.workspace_paths as workspace_paths
from abstractruntime.integrations.abstractcore.workspace_scoped_tools import WorkspaceScope, describe_workspace_scope
from abstractruntime.utils.workspace_paths import merge_builtin_workspace_protection

pytestmark = pytest.mark.basic


def _dirs(tmp_path: Path) -> dict:
    out = {}
    for name in ("own", "pictures", "project", "archive", "secrets"):
        p = tmp_path / name
        p.mkdir()
        out[name] = os.path.realpath(p)
    return out


def test_allowed_posture_lists_the_allowed_workspaces_with_modes(tmp_path):
    d = _dirs(tmp_path)
    vars0 = {
        "workspace_root": d["own"],
        "workspace_access_mode": "workspace_or_allowed",
        "workspace_allowed_paths": [d["pictures"], d["project"], d["archive"]],
        "workspace_ignored_paths": d["secrets"],
        "workspace_read_only_paths": [d["archive"]],
        # A stale round-10 key is ignored: nothing is called a shared workspace any more.
        "workspace_shared_path": d["pictures"],
    }
    lines = describe_workspace_scope(WorkspaceScope.from_input_data(vars0)).splitlines()
    assert lines[1] == f'Default working directory: "{d["own"]}"'
    assert lines[2] == "Allowed workspaces:"
    assert lines[3:6] == [
        f'  "{d["pictures"]}" (read & write)',
        f'  "{d["project"]}" (read & write)',
        f'  "{d["archive"]}" (read-only)',
    ]
    text = "\n".join(lines)
    assert "Shared workspace" not in text
    assert not any(d["secrets"] in ln and "(read" in ln for ln in lines)
    assert "Everything else" not in text


def test_allow_everything_with_read_only_default_names_everything_else(tmp_path):
    d = _dirs(tmp_path)
    vars0 = {
        "workspace_root": d["own"],
        "workspace_access_mode": "all_except_ignored",
        "workspace_allowed_paths": [d["project"]],
        "workspace_read_only_paths": ["/"],
        "workspace_writable_paths": [d["own"], d["project"]],
    }
    lines = describe_workspace_scope(WorkspaceScope.from_input_data(vars0)).splitlines()
    assert f'  "{d["project"]}" (read & write)' in lines
    assert "Everything else: (read-only)" in lines
    assert not any("Shared workspace" in ln for ln in lines)


def test_without_allowed_paths_nothing_is_invented(tmp_path):
    d = _dirs(tmp_path)
    text = describe_workspace_scope(WorkspaceScope.from_input_data({"workspace_root": d["own"]}))
    assert "Shared workspace" not in text and "Allowed workspaces" not in text


def test_the_shared_workspace_key_is_gone(tmp_path):
    d = _dirs(tmp_path)
    assert not hasattr(workspace_paths, "SHARED_WORKSPACE_KEY")
    assert not hasattr(workspace_paths, "shared_workspace_path")
    out = merge_builtin_workspace_protection({"workspace_shared_path": d["pictures"]}, {})
    assert "workspace_shared_path" not in out
