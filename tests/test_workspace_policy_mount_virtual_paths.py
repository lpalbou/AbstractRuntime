from __future__ import annotations

from pathlib import Path

import pytest


def test_workspace_or_allowed_resolves_mount_prefixed_relative_paths(tmp_path: Path) -> None:
    from abstractruntime.integrations.abstractcore.workspace_scoped_tools import WorkspaceScope, resolve_user_path

    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)

    allowed = tmp_path / "mnemosyne"
    allowed.mkdir(parents=True, exist_ok=True)
    (allowed / "file.txt").write_text("hi", encoding="utf-8")

    blocked = allowed / "blocked"
    blocked.mkdir(parents=True, exist_ok=True)

    scope = WorkspaceScope.from_input_data(
        {
            "workspace_root": str(ws),
            "workspace_access_mode": "workspace_or_allowed",
            "workspace_allowed_paths": [str(allowed)],
            "workspace_ignored_paths": [str(blocked)],
        }
    )
    assert scope is not None

    resolved = resolve_user_path(scope=scope, user_path="mnemosyne/file.txt")
    assert resolved == (allowed / "file.txt").resolve()

    with pytest.raises(ValueError, match="Path escapes workspace_root"):
        resolve_user_path(scope=scope, user_path="mnemosyne/../ws/oops.txt")

    with pytest.raises(ValueError, match="workspace_ignored_paths"):
        resolve_user_path(scope=scope, user_path="mnemosyne/blocked/x.txt")


def test_redundant_workspace_dir_prefix_is_stripped_for_relative_paths(tmp_path: Path) -> None:
    from abstractruntime.integrations.abstractcore.workspace_scoped_tools import WorkspaceScope, resolve_user_path

    ws = tmp_path / "mnemosyne"
    (ws / "memory").mkdir(parents=True, exist_ok=True)
    (ws / "memory" / "Index.md").write_text("hello", encoding="utf-8")

    scope = WorkspaceScope.from_input_data({"workspace_root": str(ws), "workspace_access_mode": "workspace_only"})
    assert scope is not None

    resolved = resolve_user_path(scope=scope, user_path="mnemosyne/memory/Index.md")
    assert resolved == (ws / "memory" / "Index.md").resolve()

    resolved2 = resolve_user_path(scope=scope, user_path="@mnemosyne/memory/Index.md")
    assert resolved2 == (ws / "memory" / "Index.md").resolve()

