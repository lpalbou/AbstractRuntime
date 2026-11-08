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

import pytest

from abstractruntime.identity.tools import (
    _EXEC_OUTPUT_CAP,
    WORKSPACE_FILE_CAP_BYTES,
    WORKSPACE_TEXT_READ_CAP_CHARS,
    ToolElection,
    WorkspaceRoot,
    sniff_binary,
    sniff_image,
    tool_result_media,
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


def test_png_read_is_shown_as_an_image_never_decoded(tmp_path: Path) -> None:
    """The incident fixture, after the 2026-11-08 ruling (images work with
    every other client, so with entities too): PNG bytes still never decode
    into a prompt — the text result is a LABEL and the file itself rides the
    next model call as an image (declared by `tool_result_media`, below)."""
    ws = WorkspaceRoot(tmp_path)
    data = _png_bytes()
    (tmp_path / "workspace" / "shared").mkdir(parents=True)
    (tmp_path / "workspace" / "shared" / "Screenshot.png").write_bytes(data)
    out = ws.read_file("shared/Screenshot.png")
    assert "#IMAGE" in out, "a labeled image result, never decoded bytes"
    assert "#BINARY" not in out, "an image is not refused"
    assert "PNG image" in out, "detected type reaches the entity"
    assert f"{len(data):,} bytes" in out, "size reaches the entity"
    assert "shared/Screenshot.png" in out, "name reaches the entity"
    assert "attached to your next look as an image" in out
    # The whole point still holds: no byte noise. The label is metadata-sized
    # and carries not one replacement character of decoded image data.
    assert "�" not in out
    assert len(out) < 1_000
    # Declared beside the read: the absolute path the host shows the model.
    media = tool_result_media(ToolElection(name="read_file", body="shared/Screenshot.png"), ws)
    assert media == [str((tmp_path / "workspace" / "shared" / "Screenshot.png").resolve())]


def test_jpeg_by_magic_is_an_image_even_without_a_nul(tmp_path: Path) -> None:
    """JPEG-class: no NUL guarantee — the image magic decides, not entropy."""
    ws = WorkspaceRoot(tmp_path)
    jpg = b"\xff\xd8\xff\xe0" + bytes(range(128, 256)) * 200  # 100% invalid continuations
    (tmp_path / "workspace" / "cam.jpg").write_bytes(jpg)
    out = ws.read_file("cam.jpg")
    assert "#IMAGE" in out and "JPEG image" in out
    assert tool_result_media(ToolElection(name="read_file", body="cam.jpg"), ws) == [
        str((tmp_path / "workspace" / "cam.jpg").resolve())
    ]


def test_high_entropy_no_nul_binary_refused(tmp_path: Path) -> None:
    """No NUL, no known magic: caught by invalid-UTF-8 density, refused."""
    ws = WorkspaceRoot(tmp_path)
    blob = bytes(range(128, 256)) * 200  # 100% invalid continuations
    (tmp_path / "workspace" / "blob.bin").write_bytes(blob)
    out = ws.read_file("blob.bin")
    assert "#BINARY" in out and "unrecognized format" in out
    assert tool_result_media(ToolElection(name="read_file", body="blob.bin"), ws) is None


def test_zip_refused_with_honest_capability_wording(tmp_path: Path) -> None:
    """A non-image binary is still refused — and the notice no longer claims
    the entity cannot view images (false since 2026-11-08); it says what is
    true: images are shown when read, other binary content is not readable."""
    ws = WorkspaceRoot(tmp_path)
    (tmp_path / "workspace" / "bundle.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 64 + bytes(range(256)))
    out = ws.read_file("bundle.zip")
    assert "#BINARY" in out and "ZIP-family archive" in out
    assert "stored intact" in out, "the file itself is not harmed"
    assert "no tool that can view images" not in out
    assert "Images (PNG, JPEG, GIF, WEBP) are shown to you when you read them" in out
    assert "other binary content is not readable here" in out
    assert tool_result_media(ToolElection(name="read_file", body="bundle.zip"), ws) is None


def test_tool_result_media_declares_only_what_the_read_served(tmp_path: Path, monkeypatch) -> None:
    """Declare-beside-execute: the media a read_file election shows is the
    image it actually opened — a text file, a missing file, a path outside
    the workspace, an image over the size cap, a tool without sight, and a
    read with no workspace all declare nothing."""
    import abstractruntime.identity.tools as tools_mod

    ws = WorkspaceRoot(tmp_path)
    (tmp_path / "workspace" / "notes.txt").write_text("plain words\n")
    (tmp_path / "workspace" / "pic.png").write_bytes(_png_bytes())
    (tmp_path / "workspace" / "pic.webp").write_bytes(b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 40)
    (tmp_path / "workspace" / "pic.gif").write_bytes(b"GIF89a" + b"\x00" * 40)
    assert tool_result_media(ToolElection(name="read_file", body="notes.txt"), ws) is None
    assert tool_result_media(ToolElection(name="read_file", body="ghost.png"), ws) is None
    assert tool_result_media(ToolElection(name="read_file", body="../../etc/passwd"), ws) is None
    assert tool_result_media(ToolElection(name="read_file", body="pic.png"), None) is None
    assert tool_result_media(ToolElection(name="list_files", body="."), ws) is None
    assert tool_result_media(ToolElection(name="read_file", body="pic.webp"), ws) == [
        str((tmp_path / "workspace" / "pic.webp").resolve())
    ]
    assert tool_result_media(ToolElection(name="read_file", body="pic.gif"), ws) == [
        str((tmp_path / "workspace" / "pic.gif").resolve())
    ]
    # Over the acceptance cap the read refuses — and so does the declaration.
    monkeypatch.setattr(tools_mod, "WORKSPACE_FILE_CAP_BYTES", 100)
    with pytest.raises(ValueError, match="cap"):
        ws.read_file("pic.png")
    assert tool_result_media(ToolElection(name="read_file", body="pic.png"), ws) is None


def test_sniff_image_is_magic_based_and_knows_webp() -> None:
    assert sniff_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8) == ("a PNG image", "image/png")
    assert sniff_image(b"\xff\xd8\xff\xe1" + b"\x00" * 8) == ("a JPEG image", "image/jpeg")
    assert sniff_image(b"GIF87a" + b"\x00" * 8) == ("a GIF image", "image/gif")
    assert sniff_image(b"RIFF\x10\x00\x00\x00WEBPVP8 ") == ("a WEBP image", "image/webp")
    assert sniff_image(b"RIFF\x10\x00\x00\x00WAVEfmt ") is None, "a wav is RIFF but not WEBP"
    assert sniff_image(b"%PDF-1.7\n") is None
    assert sniff_image(b"") is None
    assert sniff_binary(b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"\x00" * 40) == "a WEBP image"
    assert sniff_binary(b"RIFF\x10\x00\x00\x00WAVEfmt " + b"\x00" * 40) == "a RIFF media container (wav/avi/webp)"


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
    """Mounts route through the same read path — the wall has no side door:
    an image on a mount is shown (and declared), a zip there is refused."""
    home = tmp_path / "home"
    home.mkdir()
    shared = tmp_path / "granted"
    shared.mkdir()
    (shared / "pic.png").write_bytes(_png_bytes())
    (shared / "bundle.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    write_workspace_mounts(home, [{"name": "granted", "path": str(shared), "mode": "ro"}])
    ws = WorkspaceRoot(home)
    out = ws.read_file("mounts/granted/pic.png")
    assert "#IMAGE" in out and "PNG image" in out
    assert tool_result_media(ToolElection(name="read_file", body="mounts/granted/pic.png"), ws) == [
        str((shared / "pic.png").resolve())
    ]
    out = ws.read_file("mounts/granted/bundle.zip")
    assert "#BINARY" in out and "ZIP-family archive" in out
    assert tool_result_media(ToolElection(name="read_file", body="mounts/granted/bundle.zip"), ws) is None


def test_sniffer_is_content_based_not_name_based() -> None:
    # A text file that merely STARTS like a magic string stays text: labels
    # apply only after content is judged binary.
    assert sniff_binary(b"RIFF through the ages: a history of file formats.\n" * 10) is None
    assert sniff_binary(b"") is None
    assert sniff_binary(b"plain ascii\n") is None
    # NUL is binary even in an otherwise clean buffer (git's heuristic).
    assert sniff_binary(b"looks fine until\x00it does not") is not None
