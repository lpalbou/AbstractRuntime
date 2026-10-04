"""Writable exceptions inside read-only roots (`workspace_writable_paths`, gateway round 9).

"Any folder except denied" with a read-only default = every folder read-only except the run's own
folder, the shared workspace and the read & write folders: the more specific rule wins, a child run
inherits the parent's exceptions exactly and cannot add one.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from abstractruntime.integrations.abstractcore.workspace_scoped_tools import WorkspaceScope, rewrite_tool_arguments
from abstractruntime.utils.workspace_paths import (
    READ_ONLY_PATHS_KEY,
    WRITABLE_PATHS_KEY,
    is_read_only_target,
    merge_builtin_workspace_protection,
    path_is_read_only,
)


def _real(p: Path) -> str:
    return os.path.realpath(str(p))


def test_the_more_specific_rule_wins(tmp_path: Path) -> None:
    ro, rw, inner_ro = tmp_path / "archive", tmp_path / "archive" / "inbox", tmp_path / "archive" / "inbox" / "sealed"
    inner_ro.mkdir(parents=True)
    roots, writable = [_real(ro), _real(inner_ro)], [_real(rw)]
    assert is_read_only_target(Path(_real(ro)) / "a.txt", roots, writable)
    assert not is_read_only_target(Path(_real(rw)) / "a.txt", roots, writable)
    assert is_read_only_target(Path(_real(inner_ro)) / "a.txt", roots, writable)
    assert not is_read_only_target(Path(_real(tmp_path)) / "free.txt", roots, writable)
    # A writable path OUTSIDE the read-only root is no exception to it.
    assert is_read_only_target(Path(_real(ro)) / "a.txt", roots, [_real(tmp_path / "elsewhere")])


def test_a_read_only_default_with_read_write_folders_at_the_tool_scope(tmp_path: Path) -> None:
    root, shared, project, archive = (tmp_path / n for n in ("session", "shared", "project", "archive"))
    for d in (root, shared, project, archive):
        d.mkdir()
    vars0 = {
        "workspace_root": str(root),
        "workspace_access_mode": "all_except_ignored",
        READ_ONLY_PATHS_KEY: ["/"],
        WRITABLE_PATHS_KEY: [str(root), str(shared), str(project)],
    }
    scope = WorkspaceScope.from_input_data(vars0)
    for ok in (root / "a.txt", shared / "b.txt", project / "c.txt"):
        rewrite_tool_arguments(tool_name="write_file", args={"file_path": str(ok), "content": "x"}, scope=scope)
    with pytest.raises(ValueError):
        rewrite_tool_arguments(tool_name="write_file", args={"file_path": str(archive / "d.txt"), "content": "x"}, scope=scope)
    (archive / "e.txt").write_text("read me")
    out = rewrite_tool_arguments(tool_name="read_file", args={"file_path": str(archive / "e.txt")}, scope=scope)
    assert Path(out["file_path"]).resolve() == (archive / "e.txt").resolve()
    assert path_is_read_only(vars0, archive / "d.txt") and not path_is_read_only(vars0, project / "c.txt")


def test_a_child_inherits_the_exceptions_exactly(tmp_path: Path) -> None:
    parent = {READ_ONLY_PATHS_KEY: ["/"], WRITABLE_PATHS_KEY: [str(tmp_path / "project")]}
    child = {WRITABLE_PATHS_KEY: ["/"]}  # a child trying to reopen everything
    out = merge_builtin_workspace_protection(parent, child)
    assert out[WRITABLE_PATHS_KEY] == [_real(tmp_path / "project")]
    assert out[READ_ONLY_PATHS_KEY] == ["/"]
