"""laurent's cap ruling (dm#93, c327): acceptance = 20MB; reading physics
stays honest (labeled truncation per read, never refusal, never silent)."""
from __future__ import annotations

from pathlib import Path

from abstractruntime.identity.tools import (
    WORKSPACE_FILE_CAP_BYTES,
    WORKSPACE_READ_SLICE_BYTES,
    WorkspaceRoot,
)


def test_cap_is_the_ruled_20mb() -> None:
    assert WORKSPACE_FILE_CAP_BYTES == 20 * 1024 * 1024
    assert WORKSPACE_READ_SLICE_BYTES == 512 * 1024


def test_big_file_reads_truncated_with_label(tmp_path: Path) -> None:
    ws = WorkspaceRoot(tmp_path)
    big = tmp_path / "workspace" / "big.txt"
    big.write_bytes(b"x" * (WORKSPACE_READ_SLICE_BYTES + 1000))
    out = ws.read_file("big.txt")
    assert "#TRUNCATION" in out, "labeled, never silent"
    assert "stored whole" in out
    assert len(out) < WORKSPACE_READ_SLICE_BYTES + 500, "the slice bounds the prompt"


def test_small_file_reads_whole(tmp_path: Path) -> None:
    ws = WorkspaceRoot(tmp_path)
    (tmp_path / "workspace" / "s.txt").write_text("hello")
    out = ws.read_file("s.txt")
    assert "hello" in out and "#TRUNCATION" not in out
