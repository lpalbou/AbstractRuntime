"""Absolute-path re-anchoring (2026-07-12, adversarially reviewed).

Live incident: with workspace root `<root>` (real tree `<root>/mnemosyne/...`),
the model called read_file("/Users/albou/projects/mnemosyne/...") — a
fabricated absolute prefix for a file genuinely INSIDE the workspace
(/Users/albou/projects/mnemosyne does not exist). The absolute form refused;
the relative retry succeeded. Rule shipped in `resolve_user_path`:

- nonexistent absolute path: recover by suffix (longest first, depth ≥2,
  candidate must EXIST, root before mounts on ties);
- existing absolute path: only same-inode candidates re-anchor (case-alias);
  real outside files still refuse;
- symlink-out and ignored candidates are skipped; refusal keeps ONE unified
  string (no filesystem-existence oracle) with a teaching suffix.

NOTE (Windows): suffix slicing is anchor-free (parts[1:]) and identity rides
os.path.samefile; this suite runs on POSIX CI only.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from abstractruntime.integrations.abstractcore.workspace_scoped_tools import (
    WorkspaceScope,
    resolve_user_path,
)


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    (root / "mnemosyne" / "memory" / "Core").mkdir(parents=True)
    (root / "mnemosyne" / "memory" / "Core" / "Self_Model.md").write_text("self")
    (root / "notes.md").write_text("top-level")
    return root


def _scope(root: Path, **kw) -> WorkspaceScope:
    return WorkspaceScope(root=root, **kw)


# ---------------------------------------------------------------------------
# The incident replay
# ---------------------------------------------------------------------------


def test_incident_fabricated_prefix_reanchors_to_workspace(ws: Path, tmp_path: Path) -> None:
    fabricated = str(tmp_path / "elsewhere" / "mnemosyne" / "memory" / "Core" / "Self_Model.md")
    assert not Path(fabricated).exists()
    resolved = resolve_user_path(scope=_scope(ws), user_path=fabricated)
    assert resolved == (ws / "mnemosyne" / "memory" / "Core" / "Self_Model.md").resolve()


def test_longest_suffix_wins(ws: Path, tmp_path: Path) -> None:
    # Both <root>/memory/Core/x.md and <root>/mnemosyne/memory/Core/x.md exist:
    # the longer suffix (more of the model's stated intent) must win.
    (ws / "memory" / "Core").mkdir(parents=True)
    (ws / "memory" / "Core" / "Self_Model.md").write_text("decoy")
    fabricated = str(tmp_path / "nope" / "mnemosyne" / "memory" / "Core" / "Self_Model.md")
    resolved = resolve_user_path(scope=_scope(ws), user_path=fabricated)
    assert resolved == (ws / "mnemosyne" / "memory" / "Core" / "Self_Model.md").resolve()


# ---------------------------------------------------------------------------
# Refusals (the rule's deny half)
# ---------------------------------------------------------------------------


def test_depth_one_suffix_never_reanchors_nonexistent(ws: Path, tmp_path: Path) -> None:
    # `<elsewhere>/notes.md` (nonexistent) must NOT silently serve <root>/notes.md:
    # a basename-only match is no evidence of shared identity.
    fabricated = str(tmp_path / "elsewhere" / "notes.md")
    with pytest.raises(ValueError, match="escapes workspace_root"):
        resolve_user_path(scope=_scope(ws), user_path=fabricated)


def test_real_outside_file_still_refuses(ws: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "mnemosyne" / "memory" / "Core"
    outside.mkdir(parents=True)
    real = outside / "Self_Model.md"
    real.write_text("REAL outside file — must never be silently substituted")
    with pytest.raises(ValueError, match="escapes workspace_root"):
        resolve_user_path(scope=_scope(ws), user_path=str(real))


def test_unified_error_string_no_existence_oracle(ws: Path, tmp_path: Path) -> None:
    """Nonexistent-unrecoverable and real-outside refusals must be the SAME
    string shape (probing errors must not reveal outside-filesystem existence)."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real = real_dir / "secret.txt"
    real.write_text("x")
    nonexistent = str(tmp_path / "ghost" / "secret.txt")

    def _msg(path: str) -> str:
        try:
            resolve_user_path(scope=_scope(ws), user_path=path)
        except ValueError as e:
            return str(e).replace(path, "<PATH>")
        raise AssertionError("expected refusal")

    assert _msg(str(real)) == _msg(nonexistent)
    # Teaching suffix present + CLI hint substrings stable.
    msg = _msg(nonexistent)
    assert "escapes workspace_root" in msg
    assert "relative to" in msg


def test_write_target_nonexistent_never_reanchors(ws: Path, tmp_path: Path) -> None:
    # A fabricated path whose file does not exist anywhere in the workspace
    # refuses — never "create it under the root because the parent exists".
    fabricated = str(tmp_path / "elsewhere" / "mnemosyne" / "brand_new_file.md")
    with pytest.raises(ValueError, match="escapes workspace_root"):
        resolve_user_path(scope=_scope(ws), user_path=fabricated)


# ---------------------------------------------------------------------------
# Identity branch (existing paths)
# ---------------------------------------------------------------------------


def test_existing_alias_with_matching_suffix_reanchors_by_inode(ws: Path, tmp_path: Path) -> None:
    # The identity branch (portable stand-in for the APFS case-alias case,
    # which CI filesystems can't reproduce): an outside path that EXISTS,
    # whose suffix mirrors the in-root structure, and which IS the workspace
    # file (same inode) re-anchors — identity provable, no substitution.
    alias_dir = tmp_path / "ghost" / "mnemosyne" / "memory" / "Core"
    alias_dir.mkdir(parents=True)
    alias = alias_dir / "Self_Model.md"
    target = ws / "mnemosyne" / "memory" / "Core" / "Self_Model.md"
    os.link(str(target), str(alias))
    resolved = resolve_user_path(scope=_scope(ws), user_path=str(alias))
    assert resolved == target.resolve()
    assert os.path.samefile(str(resolved), str(target))


def test_existing_lookalike_with_matching_suffix_still_refuses(ws: Path, tmp_path: Path) -> None:
    # Same suffix structure but a DIFFERENT file (different inode/content):
    # identity unprovable — never substitute a real outside file.
    look_dir = tmp_path / "ghost" / "mnemosyne" / "memory" / "Core"
    look_dir.mkdir(parents=True)
    lookalike = look_dir / "Self_Model.md"
    lookalike.write_text("different content, different inode")
    with pytest.raises(ValueError, match="escapes workspace_root"):
        resolve_user_path(scope=_scope(ws), user_path=str(lookalike))


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_symlink_out_candidate_is_skipped(ws: Path, tmp_path: Path) -> None:
    # <root>/mnemosyne/leak -> outside dir; a fabricated path whose suffix
    # lands on the symlink must not escape through it.
    outside = tmp_path / "loot"
    outside.mkdir()
    (outside / "data.txt").write_text("outside")
    (ws / "mnemosyne" / "leak").symlink_to(outside)
    fabricated = str(tmp_path / "ghost" / "mnemosyne" / "leak" / "data.txt")
    with pytest.raises(ValueError, match="escapes workspace_root"):
        resolve_user_path(scope=_scope(ws), user_path=fabricated)


def test_ignored_candidate_is_skipped(ws: Path, tmp_path: Path) -> None:
    scope = _scope(ws, ignored_paths=((ws / "mnemosyne").resolve(),))
    fabricated = str(tmp_path / "ghost" / "mnemosyne" / "memory" / "Core" / "Self_Model.md")
    with pytest.raises(ValueError, match="escapes workspace_root"):
        resolve_user_path(scope=scope, user_path=fabricated)


def test_workspace_or_allowed_reanchors_into_mount(ws: Path, tmp_path: Path) -> None:
    mount = tmp_path / "extra"
    (mount / "docs").mkdir(parents=True)
    (mount / "docs" / "guide.md").write_text("g")
    scope = _scope(ws, access_mode="workspace_or_allowed", allowed_paths=(mount.resolve(),))
    fabricated = str(tmp_path / "ghost" / "docs" / "guide.md")
    resolved = resolve_user_path(scope=scope, user_path=fabricated)
    assert resolved == (mount / "docs" / "guide.md").resolve()


def test_root_wins_tie_over_mount(ws: Path, tmp_path: Path) -> None:
    mount = tmp_path / "extra"
    (mount / "mnemosyne" / "memory" / "Core").mkdir(parents=True)
    (mount / "mnemosyne" / "memory" / "Core" / "Self_Model.md").write_text("mount copy")
    scope = _scope(ws, access_mode="workspace_or_allowed", allowed_paths=(mount.resolve(),))
    fabricated = str(tmp_path / "ghost" / "mnemosyne" / "memory" / "Core" / "Self_Model.md")
    resolved = resolve_user_path(scope=scope, user_path=fabricated)
    assert resolved == (ws / "mnemosyne" / "memory" / "Core" / "Self_Model.md").resolve()


# ---------------------------------------------------------------------------
# Untouched behaviors
# ---------------------------------------------------------------------------


def test_contained_absolute_paths_unchanged(ws: Path) -> None:
    inside = str(ws / "notes.md")
    assert resolve_user_path(scope=_scope(ws), user_path=inside) == (ws / "notes.md").resolve()


def test_relative_paths_unchanged(ws: Path) -> None:
    resolved = resolve_user_path(scope=_scope(ws), user_path="mnemosyne/memory/Core/Self_Model.md")
    assert resolved == (ws / "mnemosyne" / "memory" / "Core" / "Self_Model.md").resolve()
