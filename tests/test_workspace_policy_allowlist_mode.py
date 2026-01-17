from __future__ import annotations

from pathlib import Path

import pytest


def test_workspace_or_allowed_allows_only_under_allowed_paths(tmp_path: Path) -> None:
    from abstractruntime.integrations.abstractcore.workspace_scoped_tools import WorkspaceScope, resolve_user_path

    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    allowed = tmp_path / "allowed"
    allowed.mkdir(parents=True, exist_ok=True)
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

    ok = allowed / "file.txt"
    assert resolve_user_path(scope=scope, user_path=str(ok)) == ok.resolve()

    outside = tmp_path / "outside.txt"
    with pytest.raises(ValueError, match="outside workspace roots"):
        resolve_user_path(scope=scope, user_path=str(outside))

    bad = blocked / "x.txt"
    with pytest.raises(ValueError, match="workspace_ignored_paths"):
        resolve_user_path(scope=scope, user_path=str(bad))

    rel = resolve_user_path(scope=scope, user_path="hello.txt")
    assert rel == (ws / "hello.txt").resolve()

