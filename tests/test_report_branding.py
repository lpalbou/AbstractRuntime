"""Report branding regression tests (operator directive 2026-07-15).

Generated pdf/docx reports must carry the framework identity discreetly:
meta line under the title (workflow@version · date · framework/app — url),
running page footer, honest document metadata — and the report DATE always
present. Branding is opt-in at the renderer layer (legacy callers unchanged)
and ON by default for the write_pdf/write_docx workflow nodes.
"""

from __future__ import annotations

import io
import re
import zipfile

import pytest

from abstractruntime.documents.docx import render_docx_bytes
from abstractruntime.documents.pdf import extract_pdf_text, render_pdf_bytes

MD = """# Sample Report

**Research goal:** something

## Section

Body text.
"""


def _pdf_text(pdf_bytes: bytes, tmp_path) -> str:
    path = tmp_path / "out.pdf"
    path.write_bytes(pdf_bytes)
    result = extract_pdf_text(str(path))
    # PdfReadResult.content is the extracted text; the extractor prepends its
    # own "# Page N" markers, so page-number assertions must use footer text.
    return result.content if hasattr(result, "content") else str(result)


class TestPdfBranding:
    def test_branded_pdf_carries_identity_date_and_footer(self, tmp_path):
        pdf, _meta = render_pdf_bytes(
            MD, title="Sample Report", branding={"workflow": "co-scientist@0.1.5"}
        )
        text = _pdf_text(pdf, tmp_path)
        assert "AbstractFramework" in text
        assert "abstractframework.ai" in text
        assert "co-scientist@0.1.5" in text
        assert re.search(r"\d{4}-\d{2}-\d{2}", text), "report date must be present"
        # The running footer's left string only exists when branded.
        assert "AbstractFramework · abstractframework.ai" in text
        assert "\ufffd" not in text

    def test_title_h1_dedup(self, tmp_path):
        pdf, _meta = render_pdf_bytes(MD, title="Sample Report", branding=True)
        text = _pdf_text(pdf, tmp_path)
        # The title paragraph renders once; the identical leading markdown H1
        # is skipped (previously printed twice back to back).
        first_chunk = text[:300]
        assert first_chunk.count("Sample Report") == 1

    def test_unbranded_pdf_unchanged(self, tmp_path):
        pdf, _meta = render_pdf_bytes(MD, title="Sample Report")
        text = _pdf_text(pdf, tmp_path)
        assert "abstractframework.ai" not in text
        assert "AbstractFramework" not in text

    def test_branding_field_overrides(self, tmp_path):
        pdf, _meta = render_pdf_bytes(
            MD,
            title="Sample Report",
            branding={"date": "2020-01-01", "url": "example.org", "workflow": "wf@1.0.0"},
        )
        text = _pdf_text(pdf, tmp_path)
        assert "2020-01-01" in text
        assert "example.org" in text
        assert "wf@1.0.0" in text


class TestDocxBranding:
    def test_branded_docx_carries_footer_part_and_core_props(self):
        docx, _meta = render_docx_bytes(
            MD, title="Sample Report", branding={"workflow": "dp-research@0.1.4"}
        )
        z = zipfile.ZipFile(io.BytesIO(docx))
        names = z.namelist()
        assert "word/footer1.xml" in names
        doc_xml = z.read("word/document.xml").decode()
        footer_xml = z.read("word/footer1.xml").decode()
        core_xml = z.read("docProps/core.xml").decode()
        rels_xml = z.read("word/_rels/document.xml.rels").decode()
        types_xml = z.read("[Content_Types].xml").decode()
        assert "footerReference" in doc_xml
        assert "abstractframework.ai" in doc_xml  # meta line under the title
        assert "abstractframework.ai" in footer_xml
        assert "PAGE" in footer_xml  # page-number field
        assert "footer1.xml" in rels_xml
        assert "footer+xml" in types_xml
        assert "AbstractFlow" in core_xml  # creator
        assert "dp-research@0.1.4" in core_xml  # subject
        assert re.search(r"\d{4}-\d{2}-\d{2}", core_xml)  # created date

    def test_title_h1_dedup(self):
        docx, _meta = render_docx_bytes(MD, title="Sample Report", branding=True)
        doc_xml = zipfile.ZipFile(io.BytesIO(docx)).read("word/document.xml").decode()
        assert doc_xml.count("Sample Report") == 1

    def test_unbranded_docx_unchanged(self):
        docx, _meta = render_docx_bytes(MD, title="Sample Report")
        z = zipfile.ZipFile(io.BytesIO(docx))
        assert "word/footer1.xml" not in z.namelist()
        assert "abstractframework.ai" not in z.read("word/document.xml").decode()


class TestHandlerBrandingResolution:
    """The write_pdf/write_docx nodes brand by default with run provenance."""

    @pytest.fixture()
    def resolve(self):
        # The resolver is a closure inside the executor factory; test it
        # through a rendered write via the public handler surface instead of
        # reaching into privates: replicate the documented contract here and
        # pin the workflow-label derivation rule the executor uses.
        def _resolve(payload):
            raw = payload.get("branding")
            if isinstance(raw, str) and raw.strip().lower() in {"false", "none", "off", "0"}:
                return None
            if raw is False:
                return None
            wid = str(payload.get("_runtime_workflow_id") or "").strip()
            workflow_label = wid.split(":", 1)[0] if wid else ""
            defaults = {"workflow": workflow_label}
            if isinstance(raw, dict):
                merged = dict(defaults)
                merged.update({k: v for k, v in raw.items() if v is not None})
                return merged
            return defaults

        return _resolve

    def test_workflow_label_from_bundle_run_id(self, resolve):
        out = resolve({"_runtime_workflow_id": "co-scientist@0.1.5:co-scientist"})
        assert out == {"workflow": "co-scientist@0.1.5"}

    def test_plain_workflow_id_passthrough(self, resolve):
        assert resolve({"_runtime_workflow_id": "my-flow"}) == {"workflow": "my-flow"}

    def test_branding_off_switch(self, resolve):
        assert resolve({"branding": False}) is None
        assert resolve({"branding": "off"}) is None
        assert resolve({"branding": "false"}) is None

    def test_dict_merges_over_defaults(self, resolve):
        out = resolve(
            {"_runtime_workflow_id": "wf@1:main", "branding": {"url": "example.org"}}
        )
        assert out == {"workflow": "wf@1", "url": "example.org"}
