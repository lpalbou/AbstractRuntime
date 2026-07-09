# ruff: noqa: E501
"""Minimal DOCX writer for Runtime-owned workflow document nodes.

The writer intentionally uses only the Python standard library. It supports the
Markdown structures most research reports need: ATX headings, paragraphs,
fenced code blocks, simple bullet/numbered lists, and pipe tables.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

_ATX_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+[ \t]*)?$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False)
    return str(value)


def _parse_atx_heading(line: str) -> tuple[int, str] | None:
    match = _ATX_HEADING_RE.match(line)
    if not match:
        return None
    heading_text = match.group(2).strip()
    if not heading_text:
        return None
    return len(match.group(1)), heading_text


def _strip_inline_markdown(text: str) -> str:
    # Keep this intentionally conservative; the DOCX must preserve content even
    # when Markdown is imperfect.
    out = re.sub(r"`([^`]+)`", r"\1", str(text))
    out = re.sub(r"\*\*([^*]+)\*\*", r"\1", out)
    out = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", out)
    return out


def _text_run(text: str, *, code: bool = False) -> str:
    attrs = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    run_props = ""
    if code:
        run_props = (
            "<w:rPr>"
            '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/>'
            '<w:sz w:val="18"/>'
            "</w:rPr>"
        )
    return f"<w:r>{run_props}<w:t{attrs}>{escape(text)}</w:t></w:r>"


def _paragraph(
    text: str, *, style: str | None = None, code: bool = False, indent_twips: int = 0
) -> str:
    p_props: list[str] = []
    if style:
        p_props.append(f'<w:pStyle w:val="{style}"/>')
    if indent_twips:
        p_props.append(f'<w:ind w:left="{indent_twips}"/>')
    props = f"<w:pPr>{''.join(p_props)}</w:pPr>" if p_props else ""
    return f"<w:p>{props}{_text_run(_strip_inline_markdown(text), code=code)}</w:p>"


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
    return [_strip_inline_markdown(cell.strip()) for cell in raw.split("|")]


def _table(rows: list[list[str]]) -> str:
    row_xml: list[str] = []
    for row in rows:
        cells = []
        for cell in row:
            cells.append(
                f'<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/></w:tcPr>{_paragraph(cell)}</w:tc>'
            )
        row_xml.append(f"<w:tr>{''.join(cells)}</w:tr>")
    return (
        "<w:tbl>"
        "<w:tblPr>"
        '<w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/>'
        "</w:tblPr>"
        f"{''.join(row_xml)}"
        "</w:tbl>"
    )


def _document_body(markdown_text: str, title: str | None) -> str:
    lines = markdown_text.splitlines()
    parts: list[str] = []
    if title:
        parts.append(_paragraph(title, style="Title"))

    paragraph_lines: list[str] = []
    code_lines: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            parts.append(_paragraph(" ".join(x.strip() for x in paragraph_lines if x.strip())))
            paragraph_lines = []

    def flush_code() -> None:
        nonlocal code_lines
        if code_lines:
            parts.append(_paragraph("\n".join(code_lines), style="CodeBlock", code=True))
            code_lines = []

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.rstrip()

        if line.strip().startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if _is_table_start(lines, i):
            flush_paragraph()
            table_rows = [_split_table_row(lines[i])]
            i += 2
            while i < len(lines) and "|" in lines[i].strip():
                table_rows.append(_split_table_row(lines[i]))
                i += 1
            parts.append(_table(table_rows))
            continue

        if not line.strip():
            flush_paragraph()
            i += 1
            continue

        heading = _parse_atx_heading(line)
        if heading:
            flush_paragraph()
            level, heading_text = heading
            parts.append(_paragraph(heading_text, style=f"Heading{min(level, 6)}"))
            i += 1
            continue

        bullet = re.match(r"^\s*[-*]\s+(.+)$", line)
        numbered = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if bullet or numbered:
            flush_paragraph()
            marker = "-" if bullet else "1."
            body = (bullet or numbered).group(1)
            parts.append(_paragraph(f"{marker} {body}", indent_twips=360))
            i += 1
            continue

        paragraph_lines.append(line)
        i += 1

    flush_paragraph()
    flush_code()
    if not parts:
        parts.append(_paragraph(""))
    return "".join(parts)


def _content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
"""


def _rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>
"""


def _document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="40"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:pPr><w:spacing w:before="200" w:after="100"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:pPr><w:spacing w:before="160" w:after="80"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading4"><w:name w:val="heading 4"/><w:rPr><w:b/><w:sz w:val="22"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading5"><w:name w:val="heading 5"/><w:rPr><w:b/><w:sz w:val="20"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading6"><w:name w:val="heading 6"/><w:rPr><w:b/><w:sz w:val="18"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="CodeBlock"><w:name w:val="Code Block"/><w:pPr><w:spacing w:before="120" w:after="120"/></w:pPr><w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/><w:sz w:val="18"/></w:rPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/></w:tblBorders></w:tblPr></w:style>
</w:styles>
"""


def _core_props_xml(title: str | None) -> str:
    doc_title = escape(title or "AbstractRuntime DOCX")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>{doc_title}</dc:title>
</cp:coreProperties>
"""


def _document_xml(body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1037" w:right="1037" w:bottom="1037" w:left="1037" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def render_docx_bytes(content: Any, *, title: str | None = None) -> tuple[bytes, dict[str, Any]]:
    """Render text or Markdown-ish content to a DOCX byte string."""

    text = _stringify_content(content)
    body = _document_body(text, title)
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types_xml())
        zf.writestr("_rels/.rels", _rels_xml())
        zf.writestr("docProps/core.xml", _core_props_xml(title))
        zf.writestr("word/_rels/document.xml.rels", _document_rels_xml())
        zf.writestr("word/styles.xml", _styles_xml())
        zf.writestr("word/document.xml", _document_xml(body))

    data = buffer.getvalue()
    return data, {
        "bytes": len(data),
        "sha256": sha256(data).hexdigest(),
        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "renderer": "stdlib-docx",
    }
