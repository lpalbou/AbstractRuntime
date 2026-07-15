"""PDF rendering must not drop or tofu-box the typographic + scientific glyphs
that LLMs routinely emit (non-breaking hyphens, narrow no-break spaces, en/em
dashes, Greek letters, math operators, subscripts).

Regression for the co-scientist report export: the base-14 PDF fonts have no
glyph for characters like U+2011 / U+202F / Δ / ≤, so they rendered as boxes.
The fix registers a Unicode TTF font and normalizes typographic punctuation to
ASCII. These tests round-trip a PDF and assert the text survives with no
replacement/box characters.
"""
from __future__ import annotations

import pytest

pytest.importorskip("reportlab")
pytest.importorskip("pypdf")

from abstractruntime.documents.pdf import (  # noqa: E402
    _normalize_pdf_text,
    extract_pdf_text,
    render_pdf_bytes,
)


def _render_and_extract(text: str, tmp_path) -> str:
    data, meta = render_pdf_bytes(text, title="Glyph Test")
    assert meta["bytes"] > 0
    out = tmp_path / "out.pdf"
    out.write_bytes(data)
    return extract_pdf_text(out).content


def test_typographic_punctuation_is_normalized_not_boxed(tmp_path):
    # The exact offenders from the co-scientist report: non-breaking hyphen,
    # narrow no-break space, en dash, zero-width space, ellipsis, smart quotes.
    text = (
        "Self\u2011evolving AI\u2014systems with 3\u202fcycles, proof\u2011of\u2011concept "
        "\u201cmemory\u201d graphs \u2013 see \u2018notes\u2019\u2026\u200b done."
    )
    extracted = _render_and_extract(text, tmp_path)
    # No replacement / box glyphs anywhere.
    assert "\ufffd" not in extracted
    # The characters that must be normalized away must be gone.
    for gone in ("\u2011", "\u202f", "\u200b"):
        assert gone not in extracted, f"{gone!r} should have been normalized to ASCII"
    # The meaning survives as readable ASCII.
    assert "Self-evolving AI" in extracted
    assert "proof-of-concept" in extracted


def test_scientific_glyphs_survive_via_unicode_font(tmp_path):
    # Greek letters, math operators, subscripts must render as themselves (not
    # be destroyed), which requires the registered Unicode font.
    text = "Energy Δ with τ, δ, ε; bounds ≤ x ≥ y; ratio ≈ 0.5; H₂O; a×b."
    extracted = _render_and_extract(text, tmp_path)
    assert "\ufffd" not in extracted
    for glyph in ("Δ", "τ", "δ", "ε", "≤", "≥", "≈", "×"):
        assert glyph in extracted, f"scientific glyph {glyph!r} did not survive the PDF round-trip"


def test_normalize_maps_punctuation_and_preserves_ascii():
    src = "a\u2011b\u2013c\u202fd\u2026"
    out = _normalize_pdf_text(src)
    assert out == "a-b-c d..."
    # Plain ASCII is untouched.
    assert _normalize_pdf_text("hello world 123") == "hello world 123"


def test_empty_and_none_content_render(tmp_path):
    # Degenerate inputs must not crash the renderer.
    data, meta = render_pdf_bytes("", title=None)
    assert meta["bytes"] > 0
    data2, _ = render_pdf_bytes(None, title="x")
    assert isinstance(data2, bytes) and len(data2) > 0
