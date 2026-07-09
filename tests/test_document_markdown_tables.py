from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from pypdf import PdfReader

from abstractruntime.documents import extract_pdf_text, render_docx_bytes, render_pdf_bytes


def test_pdf_renders_pipe_table_without_raw_markdown(tmp_path: Path) -> None:
    markdown = """# Report

Intro paragraph.

| Claim | Evidence | Confidence |
| --- | --- | --- |
| Supported claim | src_001 | High |
| Caveated claim | src_002 | Medium |

After paragraph.
"""

    data, meta = render_pdf_bytes(markdown, title="Report")
    path = tmp_path / "table.pdf"
    path.write_bytes(data)

    text = extract_pdf_text(path).content
    assert data.startswith(b"%PDF")
    assert meta["content_type"] == "application/pdf"
    assert "Supported claim" in text
    assert "src_001" in text
    assert "Confidence" in text
    assert "After paragraph" in text
    assert "| Claim |" not in text
    assert "| --- |" not in text


def test_docx_renders_pipe_table_without_raw_markdown() -> None:
    markdown = """# Report

| Claim | Evidence |
| --- | --- |
| Supported claim | src_001 |
"""

    data, meta = render_docx_bytes(markdown, title="Report")

    assert data.startswith(b"PK")
    assert meta["content_type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    with ZipFile(BytesIO(data)) as package:
        document_xml = package.read("word/document.xml").decode("utf-8")
    assert "<w:tbl>" in document_xml
    assert "Supported claim" in document_xml
    assert "| Claim |" not in document_xml
    assert "| --- |" not in document_xml


def test_pdf_keeps_pipe_table_inside_fenced_code(tmp_path: Path) -> None:
    markdown = """# Code Sample

```markdown
| Not | A rendered table |
| --- | --- |
| Keep | raw code |
```
"""

    data, _meta = render_pdf_bytes(markdown, title="Code Sample")
    path = tmp_path / "code.pdf"
    path.write_bytes(data)

    text = extract_pdf_text(path).content
    assert "| Not | A rendered table |" in text
    assert "| Keep | raw code |" in text


def test_pdf_handles_malformed_table_rows_without_crashing(tmp_path: Path) -> None:
    markdown = """# Uneven Table

| Claim | Evidence | Confidence |
| :--- | ---: | :---: |
| Short row | src_001 |
| Long row | src_002 | High | ignored extra cell |
"""

    data, _meta = render_pdf_bytes(markdown, title="Uneven Table")
    path = tmp_path / "uneven.pdf"
    path.write_bytes(data)

    text = extract_pdf_text(path).content
    assert "Short row" in text
    assert "Long row" in text
    assert "src_002" in text
    assert "| :--- |" not in text


def test_pdf_renders_markdown_and_bare_urls_as_clickable_links(tmp_path: Path) -> None:
    markdown = """# Linked Report

See [Example Report](https://example.com/report) and https://example.org/source.
"""

    data, _meta = render_pdf_bytes(markdown, title="Linked Report")
    path = tmp_path / "links.pdf"
    path.write_bytes(data)

    text = extract_pdf_text(path).content
    reader = PdfReader(str(path))
    uris: list[str] = []
    for page in reader.pages:
        for annotation_ref in page.get("/Annots") or []:
            annotation = annotation_ref.get_object()
            action = annotation.get("/A") or {}
            uri = action.get("/URI") if hasattr(action, "get") else None
            if uri:
                uris.append(str(uri))

    assert "Example Report" in text
    assert "https://example.org/source" in text
    assert "https://example.com/report" in uris
    assert "https://example.org/source" in uris
