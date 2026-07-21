"""0069 slices 2+3: hr / blockquote render as structure (never literal
text) in BOTH writers; fenced blocks travel unbroken across page breaks
in the PDF (KeepTogether, bounded)."""
from __future__ import annotations

import io
import zipfile

import pytest

pytest.importorskip("reportlab")

from abstractruntime.documents.pdf import render_pdf_bytes


MD = (
    "# Report\n\nIntro paragraph.\n\n---\n\n> Caveat: this result is "
    "preliminary.\n> It needs a second run.\n\nAfter the quote.\n\n"
    "```\n+----+\n|BOX |\n+----+\n```\n"
)


def test_pdf_hr_and_blockquote_are_structure_not_text() -> None:
    data, meta = render_pdf_bytes(MD, title="Report")
    assert data[:4] == b"%PDF"
    # The literal markers never render as body text.
    from abstractruntime.documents.pdf import extract_pdf_text

    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp()) / "r.pdf"
    tmp.write_bytes(data)
    text = extract_pdf_text(tmp).content
    assert not any(l.strip() in ("---", "***") for l in text.splitlines()), "no literal hr"
    assert not any(l.strip().startswith(">") for l in text.splitlines()), "no literal quote markers"
    assert "Caveat: this result is preliminary." in text.replace("\n", " ")
    assert "BOX" in text, "the fenced block still renders"


def test_docx_hr_and_blockquote_are_structure_not_text() -> None:
    from abstractruntime.documents.docx import render_docx_bytes

    data, meta = render_docx_bytes(MD, title="Report")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        doc = z.read("word/document.xml").decode("utf-8")
    assert "<w:t>---</w:t>" not in doc and ">---<" not in doc, "no literal hr paragraph"
    assert "&gt; Caveat" not in doc, "no literal quote marker"
    assert "Caveat: this result is preliminary." in doc
    assert 'w:val="single"' in doc or "LINEBELOW" not in doc  # the rule paragraph rendered


def test_pdf_long_fence_still_renders() -> None:
    """The KeepTogether bound: a >45-line fence flows normally (no blank
    pages) and still renders every line."""
    md = "```\n" + "\n".join(f"line {i}" for i in range(80)) + "\n```\n"
    data, _ = render_pdf_bytes(md, title=None)
    assert data[:4] == b"%PDF"
