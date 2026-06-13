"""Permissive PDF helpers for VisualFlow document nodes.

The default backend intentionally uses BSD-licensed libraries:
- pypdf for text and metadata extraction.
- ReportLab for PDF generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import re
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


_ATX_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+[ \t]*)?$")


@dataclass(frozen=True)
class PdfReadResult:
    content: str
    pages: int
    processed_pages: int
    metadata: dict[str, Any]
    warnings: list[str]
    truncated: bool = False


def _require_pypdf():
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ModuleNotFoundError as e:
        raise RuntimeError("PDF reading requires the BSD-licensed 'pypdf' package.") from e
    return PdfReader


def _require_reportlab():
    try:
        from reportlab.lib.pagesizes import letter  # type: ignore[import-not-found]
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore[import-not-found]
        from reportlab.lib.units import inch  # type: ignore[import-not-found]
        from reportlab.platypus import (  # type: ignore[import-not-found]
            ListFlowable,
            ListItem,
            Paragraph,
            Preformatted,
            SimpleDocTemplate,
            Spacer,
        )
    except ModuleNotFoundError as e:
        raise RuntimeError("PDF writing requires the BSD-licensed 'reportlab' package.") from e
    return {
        "letter": letter,
        "ParagraphStyle": ParagraphStyle,
        "getSampleStyleSheet": getSampleStyleSheet,
        "inch": inch,
        "ListFlowable": ListFlowable,
        "ListItem": ListItem,
        "Paragraph": Paragraph,
        "Preformatted": Preformatted,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer,
    }


def _coerce_positive_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        out = int(value)
    except Exception:
        return None
    return out if out > 0 else None


def _metadata_to_json(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    out: dict[str, Any] = {}
    try:
        items = metadata.items()
    except Exception:
        return out
    for key, value in items:
        name = str(key or "").lstrip("/")
        if not name:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
        else:
            out[name] = str(value)
    return out


def extract_pdf_text(
    path: str | Path,
    *,
    page_start: Any = None,
    page_end: Any = None,
    max_chars: Any = None,
) -> PdfReadResult:
    """Extract text from a PDF file with pypdf.

    Page numbers are 1-based when supplied. No truncation is applied unless
    `max_chars` is explicitly provided.
    """

    pdf_path = Path(path)
    PdfReader = _require_pypdf()
    reader = PdfReader(str(pdf_path))

    warnings: list[str] = []
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception as e:
            raise ValueError(f"PDF is encrypted and could not be decrypted with an empty password: {e}") from e

    total_pages = len(reader.pages)
    if total_pages == 0:
        metadata = _metadata_to_json(getattr(reader, "metadata", None))
        metadata.update(
            {
                "file_name": pdf_path.name,
                "file_size": pdf_path.stat().st_size,
                "page_count": 0,
                "processed_pages": 0,
                "content_type": "application/pdf",
            }
        )
        return PdfReadResult(
            content="",
            pages=0,
            processed_pages=0,
            metadata=metadata,
            warnings=["PDF has no pages."],
            truncated=False,
        )

    start = _coerce_positive_int(page_start) or 1
    end = _coerce_positive_int(page_end) or total_pages
    start = max(1, min(start, total_pages if total_pages else 1))
    end = max(start, min(end, total_pages if total_pages else start))

    parts: list[str] = []
    processed = 0
    for page_index in range(start - 1, end):
        processed += 1
        try:
            text = reader.pages[page_index].extract_text() or ""
        except Exception as e:
            warnings.append(f"Page {page_index + 1}: text extraction failed: {e}")
            text = ""
        if text.strip():
            parts.append(f"# Page {page_index + 1}\n\n{text.strip()}")
        else:
            warnings.append(f"Page {page_index + 1}: no extractable text.")

    content = "\n\n".join(parts)
    truncated = False
    max_len = _coerce_positive_int(max_chars)
    if max_len is not None and len(content) > max_len:
        content = content[:max_len]
        truncated = True
        warnings.append(f"#TRUNCATION: PDF text was limited to explicit max_chars={max_len}.")

    metadata = _metadata_to_json(getattr(reader, "metadata", None))
    metadata.update(
        {
            "file_name": pdf_path.name,
            "file_size": pdf_path.stat().st_size,
            "page_count": total_pages,
            "processed_pages": processed,
            "content_type": "application/pdf",
        }
    )

    return PdfReadResult(
        content=content,
        pages=total_pages,
        processed_pages=processed,
        metadata=metadata,
        warnings=warnings,
        truncated=truncated,
    )


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value, indent=2, ensure_ascii=False)
    return str(value)


def _paragraph_markup(text: str) -> str:
    escaped = escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped.replace("\n", "<br/>")


def _parse_atx_heading(line: str) -> tuple[int, str] | None:
    match = _ATX_HEADING_RE.match(line)
    if not match:
        return None
    heading_text = match.group(2).strip()
    if not heading_text:
        return None
    return len(match.group(1)), heading_text


def _append_paragraph(story: list[Any], paragraph_lines: list[str], styles: dict[str, Any], rl: dict[str, Any]) -> None:
    if not paragraph_lines:
        return
    text = " ".join(line.strip() for line in paragraph_lines if line.strip())
    if text:
        story.append(rl["Paragraph"](_paragraph_markup(text), styles["BodyText"]))
        story.append(rl["Spacer"](1, 0.08 * rl["inch"]))
    paragraph_lines.clear()


def _build_pdf_story(markdown_text: str, title: str | None, rl: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    stylesheet = rl["getSampleStyleSheet"]()
    styles = {
        "Title": stylesheet["Title"],
        "Heading1": stylesheet["Heading1"],
        "Heading2": stylesheet["Heading2"],
        "Heading3": stylesheet["Heading3"],
        "Heading4": rl["ParagraphStyle"](
            "AFHeading4",
            parent=stylesheet["Heading3"],
            fontSize=11,
            leading=14,
            spaceBefore=0.12 * rl["inch"],
            spaceAfter=0.06 * rl["inch"],
        ),
        "Heading5": rl["ParagraphStyle"](
            "AFHeading5",
            parent=stylesheet["Heading3"],
            fontSize=10,
            leading=13,
            spaceBefore=0.1 * rl["inch"],
            spaceAfter=0.05 * rl["inch"],
        ),
        "Heading6": rl["ParagraphStyle"](
            "AFHeading6",
            parent=stylesheet["Heading3"],
            fontSize=9,
            leading=12,
            spaceBefore=0.08 * rl["inch"],
            spaceAfter=0.04 * rl["inch"],
        ),
        "BodyText": stylesheet["BodyText"],
        "Bullet": stylesheet["BodyText"],
        "Code": rl["ParagraphStyle"](
            "CodeBlock",
            parent=stylesheet["Code"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            leftIndent=0.12 * rl["inch"],
            rightIndent=0.12 * rl["inch"],
            spaceAfter=0.12 * rl["inch"],
        ),
    }

    story: list[Any] = []
    if title:
        story.append(rl["Paragraph"](_paragraph_markup(title), styles["Title"]))
        story.append(rl["Spacer"](1, 0.18 * rl["inch"]))

    paragraph_lines: list[str] = []
    bullet_items: list[Any] = []
    code_lines: list[str] = []
    in_code = False

    def flush_bullets() -> None:
        nonlocal bullet_items
        if bullet_items:
            story.append(rl["ListFlowable"](bullet_items, bulletType="bullet", leftIndent=0.18 * rl["inch"]))
            story.append(rl["Spacer"](1, 0.08 * rl["inch"]))
            bullet_items = []

    def flush_code() -> None:
        if code_lines:
            story.append(rl["Preformatted"]("\n".join(code_lines), styles["Code"]))
            code_lines.clear()

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                _append_paragraph(story, paragraph_lines, styles, rl)
                flush_bullets()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            _append_paragraph(story, paragraph_lines, styles, rl)
            flush_bullets()
            continue
        heading = _parse_atx_heading(line)
        if heading:
            _append_paragraph(story, paragraph_lines, styles, rl)
            flush_bullets()
            level, heading_text = heading
            story.append(rl["Paragraph"](_paragraph_markup(heading_text), styles[f"Heading{level}"]))
            continue
        bullet = re.match(r"^\s*[-*]\s+(.+)$", line)
        if bullet:
            _append_paragraph(story, paragraph_lines, styles, rl)
            bullet_items.append(rl["ListItem"](rl["Paragraph"](_paragraph_markup(bullet.group(1).strip()), styles["Bullet"])))
            continue
        numbered = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if numbered:
            _append_paragraph(story, paragraph_lines, styles, rl)
            bullet_items.append(rl["ListItem"](rl["Paragraph"](_paragraph_markup(numbered.group(1).strip()), styles["Bullet"])))
            continue
        flush_bullets()
        paragraph_lines.append(line)

    if in_code:
        flush_code()
    _append_paragraph(story, paragraph_lines, styles, rl)
    flush_bullets()

    if not story:
        story.append(rl["Paragraph"]("", styles["BodyText"]))
    return story, styles


def render_pdf_bytes(content: Any, *, title: str | None = None) -> tuple[bytes, dict[str, Any]]:
    """Render text or Markdown-ish content to a PDF byte string with ReportLab."""

    rl = _require_reportlab()
    text = _stringify_content(content)
    buffer = BytesIO()
    doc = rl["SimpleDocTemplate"](
        buffer,
        pagesize=rl["letter"],
        rightMargin=0.72 * rl["inch"],
        leftMargin=0.72 * rl["inch"],
        topMargin=0.72 * rl["inch"],
        bottomMargin=0.72 * rl["inch"],
        title=title or "AbstractRuntime PDF",
    )
    story, _styles = _build_pdf_story(text, title, rl)
    doc.build(story)
    data = buffer.getvalue()
    return data, {
        "bytes": len(data),
        "sha256": sha256(data).hexdigest(),
        "content_type": "application/pdf",
        "renderer": "reportlab",
    }
