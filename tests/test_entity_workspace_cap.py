"""Reading physics for the entity workspace (laurent's cap ruling dm#93/c327
+ the 2026-08-01 binary-honesty repair).

Acceptance stays the ruled 20MB ("it's not up to you to decide what size is
accepted"). READING changed shape after the ephemeral incident (operator
2026-08-01): read_file on a 5,104,148-byte attached screenshot decoded half
a megabyte of PNG bytes with errors="replace" into a 494,932-char tool
message that rode the durable visit transcript into every later LLM call,
until the upstream refused the request over its context window and the
session was wedged. Two walls now stand where that happened:

- BINARY HONESTY: non-text content is refused with labeled metadata (name,
  size, detected type) — its bytes never enter a prompt as text. The
  detector is content-based (git's NUL-in-first-8000-bytes heuristic plus
  an invalid-UTF-8 density backstop), so no filename or magic-table false
  positive can refuse a real text file.
- TEXT SLICE SIZED FOR THE CONTEXT IT FEEDS: 24,000 chars (the same bound
  as execute_command output — one number for "one workspace payload
  entering one turn"; ~6k tokens = 12% of the 50k-token recommended
  working context per the same day's re-ruling, 15% of the 40k target it
  was derived against), replacing the old 512KiB byte slice (~131k
  tokens = several times that whole context). Sliced in chars, never
  mid-codepoint.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

from abstractruntime.identity.tools import (
    _EXEC_OUTPUT_CAP,
    WORKSPACE_FILE_CAP_BYTES,
    WORKSPACE_TEXT_READ_CAP_CHARS,
    WorkspaceRoot,
    sniff_binary,
    write_workspace_mounts,
)


def _png_bytes(payload_size: int = 60_000) -> bytes:
    """A structurally real PNG: signature, IHDR, one IDAT of zlib data, IEND.
    The incident file's shape (magic bytes + NUL-laden chunk headers)."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", 16, 16, 8, 6, 0, 0, 0) + b"\x00\x00\x00\x00"
    payload = zlib.compress(bytes(range(256)) * (payload_size // 256), level=0)
    idat = struct.pack(">I", len(payload)) + b"IDAT" + payload + b"\x00\x00\x00\x00"
    iend = struct.pack(">I", 0) + b"IEND" + b"\xaeB`\x82"
    return sig + ihdr + idat + iend


def test_acceptance_cap_is_the_ruled_20mb() -> None:
    assert WORKSPACE_FILE_CAP_BYTES == 20 * 1024 * 1024


def test_text_read_cap_matches_exec_bound_and_context_arithmetic() -> None:
    # One number for the class "one workspace payload entering one turn".
    assert WORKSPACE_TEXT_READ_CAP_CHARS == _EXEC_OUTPUT_CAP == 24_000
    # 12% of the recommended working context at the repo's 4-chars/token
    # heuristic — the arithmetic the constant's comment claims after the
    # 2026-08-01 re-ruling moved the target to 50k (the cap itself was
    # derived against 40k as 15% and deliberately KEPT: it bounds reading
    # physics, not attention, and the re-ruling touched no tool caps).
    # 24_000 chars = 6_000 tokens; 6_000 * 25 / 3 = 50_000: cap × 25 =
    # recommendation × 4 (chars/token) × 3.
    try:
        from abstractmemory import ENTITY_CONTEXT_RECOMMENDED
    except ImportError:  # arithmetic pin still holds against the ruled 50k
        ENTITY_CONTEXT_RECOMMENDED = 50_000
    assert WORKSPACE_TEXT_READ_CAP_CHARS * 25 == ENTITY_CONTEXT_RECOMMENDED * 4 * 3


def test_png_read_refused_with_labeled_metadata(tmp_path: Path) -> None:
    """The incident fixture: PNG magic bytes must never decode into a prompt."""
    ws = WorkspaceRoot(tmp_path)
    data = _png_bytes()
    (tmp_path / "workspace" / "shared").mkdir(parents=True)
    (tmp_path / "workspace" / "shared" / "Screenshot.png").write_bytes(data)
    out = ws.read_file("shared/Screenshot.png")
    assert "#BINARY" in out, "a labeled refusal, never a silent one"
    assert "PNG image" in out, "detected type reaches the entity"
    assert f"{len(data)} bytes" in out, "size reaches the entity"
    assert "shared/Screenshot.png" in out, "name reaches the entity"
    assert "stored intact" in out, "the file itself is not harmed"
    # The whole point: no byte noise. The refusal is metadata-sized and
    # carries not one replacement character of decoded image data.
    assert "�" not in out
    assert len(out) < 1_000
    # Honest capability pointer: the walled surface has no vision tool, so
    # none may be named/invented — the message says so and routes to the
    # human instead.
    assert "no tool that can view images" in out


def test_high_entropy_no_nul_binary_refused(tmp_path: Path) -> None:
    """JPEG-class: no NUL guarantee, caught by invalid-UTF-8 density."""
    ws = WorkspaceRoot(tmp_path)
    jpg = b"\xff\xd8\xff\xe0" + bytes(range(128, 256)) * 200  # 100% invalid continuations
    (tmp_path / "workspace" / "cam.jpg").write_bytes(jpg)
    out = ws.read_file("cam.jpg")
    assert "#BINARY" in out and "JPEG image" in out


def test_utf16_text_refused_with_reencode_hint(tmp_path: Path) -> None:
    ws = WorkspaceRoot(tmp_path)
    (tmp_path / "workspace" / "u16.txt").write_bytes("words in the wrong shape\n".encode("utf-16"))
    out = ws.read_file("u16.txt")
    assert "#BINARY" in out and "UTF-16" in out and "re-save as UTF-8" in out


def test_latin1_prose_is_served_not_refused(tmp_path: Path) -> None:
    """A legacy encoding slip is a text file with scars, not a binary — the
    density bar (30%) keeps accented prose readable instead of walled off."""
    ws = WorkspaceRoot(tmp_path)
    prose = "café crème, très tôt, où ça? Un été à Orléans.\n".encode("latin-1") * 40
    (tmp_path / "workspace" / "notes.txt").write_bytes(prose)
    out = ws.read_file("notes.txt")
    assert "#BINARY" not in out
    assert "caf" in out  # the readable body is served (with U+FFFD scars)


def test_big_text_reads_truncated_with_char_label(tmp_path: Path) -> None:
    ws = WorkspaceRoot(tmp_path)
    (tmp_path / "workspace" / "big.txt").write_text("x" * (WORKSPACE_TEXT_READ_CAP_CHARS + 5_000))
    out = ws.read_file("big.txt")
    assert "#TRUNCATION" in out, "labeled, never silent"
    assert "stored whole" in out
    assert len(out) < WORKSPACE_TEXT_READ_CAP_CHARS + 500, "the slice bounds the prompt"


def test_char_slice_never_splits_a_codepoint(tmp_path: Path) -> None:
    ws = WorkspaceRoot(tmp_path)
    (tmp_path / "workspace" / "emoji.txt").write_text("🌍" * (WORKSPACE_TEXT_READ_CAP_CHARS + 10))
    out = ws.read_file("emoji.txt")
    assert "#TRUNCATION" in out
    assert "�" not in out, "slicing by chars after decode: no torn codepoints"


def test_small_file_reads_whole(tmp_path: Path) -> None:
    ws = WorkspaceRoot(tmp_path)
    (tmp_path / "workspace" / "s.txt").write_text("hello")
    out = ws.read_file("s.txt")
    assert "hello" in out and "#TRUNCATION" not in out and "#BINARY" not in out


def test_mount_reads_hit_the_same_binary_wall(tmp_path: Path) -> None:
    """Mounts route through the same read path — the wall has no side door."""
    home = tmp_path / "home"
    home.mkdir()
    shared = tmp_path / "granted"
    shared.mkdir()
    (shared / "pic.png").write_bytes(_png_bytes())
    write_workspace_mounts(home, [{"name": "granted", "path": str(shared), "mode": "ro"}])
    ws = WorkspaceRoot(home)
    out = ws.read_file("mounts/granted/pic.png")
    assert "#BINARY" in out and "PNG image" in out


def test_sniffer_is_content_based_not_name_based() -> None:
    # A text file that merely STARTS like a magic string stays text: labels
    # apply only after content is judged binary.
    assert sniff_binary(b"RIFF through the ages: a history of file formats.\n" * 10) is None
    assert sniff_binary(b"") is None
    assert sniff_binary(b"plain ascii\n") is None
    # NUL is binary even in an otherwise clean buffer (git's heuristic).
    assert sniff_binary(b"looks fine until\x00it does not") is not None
