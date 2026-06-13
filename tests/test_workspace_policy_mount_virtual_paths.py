from __future__ import annotations

from pathlib import Path

import pytest

from abstractcore.utils.workspace_paths import build_workspace_mounts


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


def test_mount_alias_collisions_use_shared_digest_aliases(tmp_path: Path) -> None:
    from abstractruntime.integrations.abstractcore.workspace_scoped_tools import WorkspaceScope, resolve_user_path

    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)

    first = tmp_path / "team-a" / "reports"
    second = tmp_path / "team-b" / "reports"
    first.mkdir(parents=True, exist_ok=True)
    second.mkdir(parents=True, exist_ok=True)
    (first / "alpha.txt").write_text("alpha", encoding="utf-8")
    (second / "beta.txt").write_text("beta", encoding="utf-8")

    mounts = build_workspace_mounts(allowed_dirs=[first, second], used_names=set())
    assert len(mounts) == 2
    assert all(name.startswith("reports_") for name in mounts.keys())

    scope = WorkspaceScope.from_input_data(
        {
            "workspace_root": str(ws),
            "workspace_access_mode": "workspace_or_allowed",
            "workspace_allowed_paths": [str(first), str(second)],
        }
    )
    assert scope is not None

    first_alias = next(name for name, path in mounts.items() if path == first.resolve())
    second_alias = next(name for name, path in mounts.items() if path == second.resolve())
    assert resolve_user_path(scope=scope, user_path=f"{first_alias}/alpha.txt") == (first / "alpha.txt").resolve()
    assert resolve_user_path(scope=scope, user_path=f"{second_alias}/beta.txt") == (second / "beta.txt").resolve()


def test_resolve_user_workspace_path_accepts_bare_mount_root_alias(tmp_path: Path) -> None:
    from abstractruntime.integrations.abstractcore.workspace_scoped_tools import WorkspaceScope, resolve_user_workspace_path

    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    mounted = tmp_path / "team-a" / "reports"
    mounted.mkdir(parents=True, exist_ok=True)

    mounts = build_workspace_mounts(allowed_dirs=[mounted], used_names=set())
    mount_alias = next(iter(mounts.keys()))

    scope = WorkspaceScope.from_input_data(
        {
            "workspace_root": str(ws),
            "workspace_access_mode": "workspace_or_allowed",
            "workspace_allowed_paths": [str(mounted)],
        }
    )
    assert scope is not None

    resolved = resolve_user_workspace_path(scope=scope, user_path=mount_alias)
    assert resolved.resolved_path == mounted.resolve()
    assert resolved.virtual_path == mount_alias
    assert resolved.root_path == mounted.resolve()


def test_all_except_ignored_absolute_paths_under_workspace_root_still_canonicalize(tmp_path: Path) -> None:
    from abstractruntime.integrations.abstractcore.workspace_scoped_tools import WorkspaceScope, resolve_user_workspace_path

    ws = tmp_path / "outside"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "secret.txt").write_text("secret\n", encoding="utf-8")

    scope = WorkspaceScope.from_input_data(
        {
            "workspace_root": str(ws),
            "workspace_access_mode": "all_except_ignored",
            "workspace_allowed_paths": [str(ws)],
        }
    )
    assert scope is not None

    resolved = resolve_user_workspace_path(scope=scope, user_path=str(ws / "secret.txt"))
    assert resolved.resolved_path == (ws / "secret.txt").resolve()
    assert resolved.virtual_path == "secret.txt"
    assert resolved.root_path == ws.resolve()
