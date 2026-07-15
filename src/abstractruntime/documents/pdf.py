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
from xml.sax.saxutils import escape, quoteattr


_ATX_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+[ \t]*)?$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_BARE_URL_RE = re.compile(r"https?://[^\s<>()]+")

# Typographic punctuation the LLM emits that the PDF base fonts render as tofu
# boxes (or that break line-wrapping). Mapped to clean ASCII equivalents so the
# text is correct regardless of which font is available. Non-punctuation
# symbols (Greek letters, math operators, subscripts) are NOT mapped here —
# those depend on a Unicode font (registered below); mapping them to ASCII
# would destroy meaning.
_PDF_PUNCT_MAP = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "\u2014",
    "\u2015": "--", "\u2212": "-",  # figure/en dash, minus -> hyphen; em dash kept for the font
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2032": "'", "\u2033": '"',
    "\u2026": "...",
    "\u2022": "-", "\u2023": "-", "\u2043": "-", "\u2219": "-",
    "\u00a0": " ", "\u2007": " ", "\u2008": " ", "\u2009": " ", "\u200a": " ",
    "\u202f": " ", "\u205f": " ", "\u3000": " ",
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
    "\u2016": "||",
}
_PDF_PUNCT_TABLE = {ord(k): v for k, v in _PDF_PUNCT_MAP.items()}

# Cache the resolved Unicode font family (or "" if none / Helvetica fallback).
_UNICODE_FONT_FAMILY: str | None = None
# Mono font used for inline code / code blocks ("Courier" fallback).
_MONO_FONT_NAME: str = "Courier"


def _normalize_pdf_text(text: str) -> str:
    """Replace typographic punctuation/whitespace with ASCII so it never renders
    as a tofu box. Symbols needing a real Unicode font are left for the font."""
    if not text:
        return text
    return text.translate(_PDF_PUNCT_TABLE)


def _candidate_unicode_fonts() -> list[tuple[str, str, str, str]]:
    """(regular, bold, italic, bolditalic) TTF path tuples to try, in order.
    An env override wins; then DejaVu (matplotlib), then common system fonts."""
    import os

    out: list[tuple[str, str, str, str]] = []
    env = os.environ.get("ABSTRACTRUNTIME_PDF_FONT", "").strip()
    if env and Path(env).is_file():
        out.append((env, env, env, env))
    # DejaVuSans ships with matplotlib and covers Latin+Greek+math+punctuation.
    try:
        import matplotlib  # type: ignore[import-not-found]

        mdir = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        dv = mdir / "DejaVuSans.ttf"
        if dv.is_file():
            out.append(
                (
                    str(dv),
                    str(mdir / "DejaVuSans-Bold.ttf"),
                    str(mdir / "DejaVuSans-Oblique.ttf"),
                    str(mdir / "DejaVuSans-BoldOblique.ttf"),
                )
            )
    except Exception:
        pass
    # Common system locations (Linux DejaVu, macOS Arial Unicode).
    for base in (
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/dejavu",
        "/usr/local/share/fonts",
    ):
        dv = Path(base) / "DejaVuSans.ttf"
        if dv.is_file():
            out.append(
                (
                    str(dv),
                    str(Path(base) / "DejaVuSans-Bold.ttf"),
                    str(Path(base) / "DejaVuSans-Oblique.ttf"),
                    str(Path(base) / "DejaVuSans-BoldOblique.ttf"),
                )
            )
    mac_arial = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if mac_arial.is_file():
        out.append((str(mac_arial), str(mac_arial), str(mac_arial), str(mac_arial)))
    return out


def _mono_font_candidates() -> list[str]:
    out: list[str] = []
    try:
        import matplotlib  # type: ignore[import-not-found]

        m = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSansMono.ttf"
        if m.is_file():
            out.append(str(m))
    except Exception:
        pass
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ):
        if Path(p).is_file():
            out.append(p)
    return out


def _register_unicode_font() -> str:
    """Register a Unicode TTF family (regular/bold/italic + a mono) with
    ReportLab and return the body family name. Returns "" when none is
    available (callers keep Helvetica + punctuation normalization)."""
    global _UNICODE_FONT_FAMILY, _MONO_FONT_NAME
    if _UNICODE_FONT_FAMILY is not None:
        return _UNICODE_FONT_FAMILY
    _UNICODE_FONT_FAMILY = ""
    try:
        from reportlab.pdfbase import pdfmetrics  # type: ignore[import-not-found]
        from reportlab.pdfbase.ttfonts import TTFont  # type: ignore[import-not-found]
    except Exception:
        return _UNICODE_FONT_FAMILY
    for regular, bold, italic, bolditalic in _candidate_unicode_fonts():
        try:
            family = "AFUnicode"
            pdfmetrics.registerFont(TTFont(f"{family}", regular))
            pdfmetrics.registerFont(TTFont(f"{family}-Bold", bold))
            pdfmetrics.registerFont(TTFont(f"{family}-Italic", italic))
            pdfmetrics.registerFont(TTFont(f"{family}-BoldItalic", bolditalic))
            pdfmetrics.registerFontFamily(
                family,
                normal=family,
                bold=f"{family}-Bold",
                italic=f"{family}-Italic",
                boldItalic=f"{family}-BoldItalic",
            )
            _UNICODE_FONT_FAMILY = family
            break
        except Exception:
            continue
    # A Unicode-capable mono for inline code / code blocks (best effort).
    if _UNICODE_FONT_FAMILY:
        for mono in _mono_font_candidates():
            try:
                pdfmetrics.registerFont(TTFont("AFMono", mono))
                _MONO_FONT_NAME = "AFMono"
                break
            except Exception:
                continue
    return _UNICODE_FONT_FAMILY


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
        from reportlab.lib import colors  # type: ignore[import-not-found]
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
            Table,
            TableStyle,
        )
    except ModuleNotFoundError as e:
        raise RuntimeError("PDF writing requires the BSD-licensed 'reportlab' package.") from e
    return {
        "colors": colors,
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
        "Table": Table,
        "TableStyle": TableStyle,
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


def _trim_url_punctuation(url: str) -> tuple[str, str]:
    trailing = ""
    while url and url[-1] in ".,;:!?)]}":
        trailing = url[-1] + trailing
        url = url[:-1]
    return url, trailing


def _inline_text_markup(text: str) -> str:
    escaped = escape(text)
    escaped = re.sub(r"`([^`]+)`", rf"<font name='{_MONO_FONT_NAME}'>\1</font>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def _link_markup(label: str, url: str) -> str:
    label_markup = _inline_text_markup(label)
    return f"<a href={quoteattr(url)} color='blue'><u>{label_markup}</u></a>"


def _markup_bare_urls(text: str) -> str:
    parts: list[str] = []
    cursor = 0
    for match in _BARE_URL_RE.finditer(text):
        parts.append(_inline_text_markup(text[cursor : match.start()]))
        url, trailing = _trim_url_punctuation(match.group(0))
        if url:
            parts.append(_link_markup(url, url))
        if trailing:
            parts.append(_inline_text_markup(trailing))
        cursor = match.end()
    parts.append(_inline_text_markup(text[cursor:]))
    return "".join(parts)


def _paragraph_markup(text: str) -> str:
    lines: list[str] = []
    for line in str(text).split("\n"):
        parts: list[str] = []
        cursor = 0
        for match in _MARKDOWN_LINK_RE.finditer(line):
            parts.append(_markup_bare_urls(line[cursor : match.start()]))
            url, trailing = _trim_url_punctuation(match.group(2))
            if url:
                parts.append(_link_markup(match.group(1), url))
            if trailing:
                parts.append(_inline_text_markup(trailing))
            cursor = match.end()
        parts.append(_markup_bare_urls(line[cursor:]))
        lines.append("".join(parts))
    return "<br/>".join(lines)


def _parse_atx_heading(line: str) -> tuple[int, str] | None:
    match = _ATX_HEADING_RE.match(line)
    if not match:
        return None
    heading_text = match.group(2).strip()
    if not heading_text:
        return None
    return len(match.group(1)), heading_text


def _is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    first = lines[index].strip()
    second = lines[index + 1].strip()
    return "|" in first and bool(_TABLE_SEPARATOR_RE.match(second))


def _split_table_row(line: str) -> list[str]:
    raw = line.strip()
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    return [cell.strip() for cell in raw.split("|")]


def _normalize_table_rows(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return []
    column_count = max(len(row) for row in rows)
    normalized: list[list[str]] = []
    for row in rows:
        padded = [*row, *([""] * max(0, column_count - len(row)))]
        normalized.append(padded[:column_count])
    return normalized


def _table_column_widths(headers: list[str], total_width: float) -> list[float]:
    if not headers:
        return []
    weights: list[float] = []
    for header in headers:
        label = header.strip().lower()
        if any(key in label for key in ("confidence", "status", "score")):
            weights.append(0.72)
        elif any(key in label for key in ("source", "reference", "evidence", "url")):
            weights.append(1.05)
        elif any(key in label for key in ("claim", "finding", "summary", "description")):
            weights.append(1.5)
        else:
            weights.append(1.0)
    weight_total = sum(weights) or float(len(headers))
    return [total_width * weight / weight_total for weight in weights]


def _append_table(
    story: list[Any], rows: list[list[str]], styles: dict[str, Any], rl: dict[str, Any]
) -> None:
    normalized = _normalize_table_rows(rows)
    if not normalized:
        return

    data: list[list[Any]] = []
    for row_index, row in enumerate(normalized):
        style_name = "TableHeader" if row_index == 0 else "TableCell"
        data.append([rl["Paragraph"](_paragraph_markup(cell), styles[style_name]) for cell in row])

    page_width = float(rl["letter"][0])
    usable_width = page_width - (1.44 * rl["inch"])
    table = rl["Table"](
        data,
        colWidths=_table_column_widths(normalized[0], usable_width),
        hAlign="LEFT",
        repeatRows=1,
    )
    table.setStyle(
        rl["TableStyle"](
            [
                ("BACKGROUND", (0, 0), (-1, 0), rl["colors"].HexColor("#E9EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl["colors"].HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.25, rl["colors"].HexColor("#CBD5E1")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.75, rl["colors"].HexColor("#64748B")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl["colors"].white, rl["colors"].HexColor("#F8FAFC")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(rl["Spacer"](1, 0.12 * rl["inch"]))


def _append_paragraph(story: list[Any], paragraph_lines: list[str], styles: dict[str, Any], rl: dict[str, Any]) -> None:
    if not paragraph_lines:
        return
    text = " ".join(line.strip() for line in paragraph_lines if line.strip())
    if text:
        story.append(rl["Paragraph"](_paragraph_markup(text), styles["BodyText"]))
        story.append(rl["Spacer"](1, 0.08 * rl["inch"]))
    paragraph_lines.clear()


def _apply_font_family(styles: dict[str, Any], family: str) -> None:
    """Point every text style at the registered Unicode family (mono styles at
    the Unicode mono) so glyphs like Greek/math/em-dash render instead of tofu.
    ReportLab resolves <b>/<i> via the registered font family."""
    if not family:
        return
    for name, style in styles.items():
        if name in ("Code",):
            style.fontName = _MONO_FONT_NAME
        elif name == "TableHeader":
            style.fontName = f"{family}-Bold"
        else:
            style.fontName = family


def _build_pdf_story(
    markdown_text: str,
    title: str | None,
    rl: dict[str, Any],
    *,
    branding: dict[str, str] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
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
        "TableHeader": rl["ParagraphStyle"](
            "AFTableHeader",
            parent=stylesheet["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
        ),
        "TableCell": rl["ParagraphStyle"](
            "AFTableCell",
            parent=stylesheet["BodyText"],
            fontSize=8,
            leading=10,
        ),
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

    _apply_font_family(styles, _register_unicode_font())

    story: list[Any] = []
    if title:
        story.append(rl["Paragraph"](_paragraph_markup(title), styles["Title"]))
        story.append(rl["Spacer"](1, 0.06 * rl["inch"]))
    if branding:
        # Discreet identity line under the title: workflow@version · date ·
        # framework — url. Small, gray, followed by a thin rule.
        meta_style = rl["ParagraphStyle"](
            "AFReportMeta",
            parent=styles["BodyText"],
            fontName=styles["BodyText"].fontName,
            fontSize=8,
            leading=10,
            textColor=rl["colors"].Color(0.45, 0.45, 0.45),
            alignment=1 if title else 0,  # centered under a title, left otherwise
        )
        bits = []
        if branding.get("workflow"):
            bits.append(branding["workflow"])
        bits.append(branding["date"])
        bits.append(f"{branding['framework']} / {branding['app']}")
        meta_text = " · ".join(bits) + f' — <link href="https://{branding["url"]}">{branding["url"]}</link>'
        story.append(rl["Paragraph"](meta_text, meta_style))
        story.append(rl["Spacer"](1, 0.06 * rl["inch"]))
        rule = rl["Table"]([[""]], colWidths=[6.9 * rl["inch"]], rowHeights=[1])
        rule.setStyle(rl["TableStyle"]([
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, rl["colors"].Color(0.75, 0.75, 0.75)),
        ]))
        story.append(rule)
        story.append(rl["Spacer"](1, 0.16 * rl["inch"]))
    elif title:
        story.append(rl["Spacer"](1, 0.12 * rl["inch"]))

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

    lines = markdown_text.splitlines()
    # Title/H1 dedup: when the document already renders a title page line,
    # a leading markdown H1 with the SAME text would print it twice.
    if title:
        for idx, probe in enumerate(lines):
            if not probe.strip():
                continue
            m = re.match(r"^#\s+(.+?)\s*$", probe.strip())
            if m and m.group(1).strip().lower() == title.strip().lower():
                lines = lines[:idx] + lines[idx + 1 :]
            break
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                _append_paragraph(story, paragraph_lines, styles, rl)
                flush_bullets()
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue
        if _is_table_start(lines, i):
            _append_paragraph(story, paragraph_lines, styles, rl)
            flush_bullets()
            table_rows = [_split_table_row(lines[i])]
            i += 2
            while i < len(lines) and "|" in lines[i].strip():
                table_rows.append(_split_table_row(lines[i]))
                i += 1
            _append_table(story, table_rows, styles, rl)
            continue
        if not line.strip():
            _append_paragraph(story, paragraph_lines, styles, rl)
            flush_bullets()
            i += 1
            continue
        heading = _parse_atx_heading(line)
        if heading:
            _append_paragraph(story, paragraph_lines, styles, rl)
            flush_bullets()
            level, heading_text = heading
            story.append(rl["Paragraph"](_paragraph_markup(heading_text), styles[f"Heading{level}"]))
            i += 1
            continue
        bullet = re.match(r"^\s*[-*]\s+(.+)$", line)
        if bullet:
            _append_paragraph(story, paragraph_lines, styles, rl)
            bullet_items.append(rl["ListItem"](rl["Paragraph"](_paragraph_markup(bullet.group(1).strip()), styles["Bullet"])))
            i += 1
            continue
        numbered = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if numbered:
            _append_paragraph(story, paragraph_lines, styles, rl)
            bullet_items.append(rl["ListItem"](rl["Paragraph"](_paragraph_markup(numbered.group(1).strip()), styles["Bullet"])))
            i += 1
            continue
        flush_bullets()
        paragraph_lines.append(line)
        i += 1

    if in_code:
        flush_code()
    _append_paragraph(story, paragraph_lines, styles, rl)
    flush_bullets()

    if not story:
        story.append(rl["Paragraph"]("", styles["BodyText"]))
    return story, styles


def _normalize_branding(branding: Any) -> dict[str, str] | None:
    """Normalize the branding payload for report exports.

    Branding is opt-in (None keeps the legacy plain render for non-report
    callers). A truthy value yields a dict with the framework identity
    defaults filled; callers may override any field. The report DATE is
    always present (operator requirement: "the date of the report is also
    essential") — defaulting to the generation date.
    """
    if not branding:
        return None
    raw = branding if isinstance(branding, dict) else {}
    from datetime import datetime, timezone

    out = {
        "framework": str(raw.get("framework") or "AbstractFramework"),
        "app": str(raw.get("app") or "AbstractFlow"),
        "url": str(raw.get("url") or "abstractframework.ai"),
        "date": str(raw.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")),
        "workflow": str(raw.get("workflow") or "").strip(),
    }
    return out


def render_pdf_bytes(
    content: Any, *, title: str | None = None, branding: Any = None
) -> tuple[bytes, dict[str, Any]]:
    """Render text or Markdown-ish content to a PDF byte string with ReportLab.

    When `branding` is provided (dict or truthy), the document carries the
    framework identity discreetly: a small meta line under the title, a thin
    rule, a running footer (framework · url | page number) and a tiny
    running header on later pages — plus honest PDF metadata (author/creator).
    """

    rl = _require_reportlab()
    text = _normalize_pdf_text(_stringify_content(content))
    if title:
        title = _normalize_pdf_text(str(title))
    brand = _normalize_branding(branding)
    buffer = BytesIO()
    doc = rl["SimpleDocTemplate"](
        buffer,
        pagesize=rl["letter"],
        rightMargin=0.72 * rl["inch"],
        leftMargin=0.72 * rl["inch"],
        topMargin=0.72 * rl["inch"],
        bottomMargin=0.78 * rl["inch"] if brand else 0.72 * rl["inch"],
        title=title or "AbstractRuntime PDF",
        author=f"{brand['app']} — {brand['framework']}" if brand else None,
        creator=f"{brand['app']} ({brand['framework']}) · {brand['url']}" if brand else None,
        subject=(brand.get("workflow") or None) if brand else None,
    )
    story, _styles = _build_pdf_story(text, title, rl, branding=brand)
    if brand:
        gray = rl["colors"].Color(0.45, 0.45, 0.45)
        line_gray = rl["colors"].Color(0.75, 0.75, 0.75)
        font = _register_unicode_font() or "Helvetica"
        footer_left = f"{brand['framework']} · {brand['url']}"
        if brand.get("workflow"):
            footer_left = f"{footer_left} · {brand['workflow']}"
        header_right = f"{brand['app']} — {brand['date']}"

        def _decorate(canvas, docobj, *, with_header: bool) -> None:
            canvas.saveState()
            page_w = rl["letter"][0]
            left = docobj.leftMargin
            right = page_w - docobj.rightMargin
            # Footer: rule + framework identity left, page number right.
            y = 0.5 * rl["inch"]
            canvas.setStrokeColor(line_gray)
            canvas.setLineWidth(0.5)
            canvas.line(left, y + 10, right, y + 10)
            canvas.setFont(font, 7)
            canvas.setFillColor(gray)
            canvas.drawString(left, y, footer_left)
            canvas.drawRightString(right, y, f"Page {canvas.getPageNumber()}")
            if with_header:
                # Later pages: tiny right-aligned running header.
                canvas.drawRightString(right, rl["letter"][1] - 0.45 * rl["inch"], header_right)
            canvas.restoreState()

        doc.build(
            story,
            onFirstPage=lambda c, d: _decorate(c, d, with_header=False),
            onLaterPages=lambda c, d: _decorate(c, d, with_header=True),
        )
    else:
        doc.build(story)
    data = buffer.getvalue()
    return data, {
        "bytes": len(data),
        "sha256": sha256(data).hexdigest(),
        "content_type": "application/pdf",
        "renderer": "reportlab",
    }
