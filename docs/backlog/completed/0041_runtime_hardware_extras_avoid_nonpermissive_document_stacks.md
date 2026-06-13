# Completed: Runtime hardware extras avoid nonpermissive document stacks

## Metadata
- Created: 2026-06-05
- Status: Completed
- Completed: 2026-06-06

## ADR status
- Governing ADRs: root `docs/adr/0029-permissive-dependency-and-licensing-policy.md`, root `docs/adr/0033-install-profiles-config-entrypoints-and-server-boundaries.md`
- ADR impact: None. This completes the Runtime-side enforcement of the existing permissive dependency and install-profile policies.

## Context
Runtime's base VisualFlow PDF nodes use permissive dependencies:

- `pypdf` for PDF text and metadata extraction;
- `reportlab` for PDF writing.

The remaining risk was that Runtime `apple` and `gpu` extras delegated to Core aggregate profiles that previously pulled PyMuPDF-family packages. Core item `0805` moved the default Core PDF route to `pypdf` and isolated PyMuPDF-family packages behind the explicit `pdf-pymupdf-commercial` extra.

## What changed
- Runtime remains versioned with mandatory `pypdf` and `reportlab` base dependencies, so light, Apple, and GPU installs all receive PDF read/write support.
- Runtime packaging tests now assert PyMuPDF-family packages are absent from the Runtime base, `apple`, and `gpu` profile definitions.
- Runtime packaging tests also inspect the sibling Core `all-apple` and `all-gpu` profiles in the monorepo and assert they contain `pypdf` but not PyMuPDF-family packages.
- The attachment/open test now creates a PDF through Runtime's ReportLab writer instead of PyMuPDF, proving the default route parses PDFs without the optional commercial backend.

## Current code pointers
- `pyproject.toml`
- `src/abstractruntime/documents/pdf.py`
- `tests/test_packaging_extras.py`
- `tests/test_visualflow_file_nodes_workspace.py`
- `tests/test_session_attachments_registry_and_open_tool.py`

## Validation
- `python -m pytest tests/test_visualflow_file_nodes_workspace.py tests/test_session_attachments_registry_and_open_tool.py::test_open_attachment_pdf_extracts_text_when_media_stack_available tests/test_packaging_extras.py -q`
  - Result: 6 passed.

## Completion report
Runtime hardware profile PDF dependency inheritance is now aligned with the permissive default policy. PyMuPDF remains possible only through Core's explicit opt-in extra; Runtime does not install it by default in base, Apple, or GPU profiles.

## Residual follow-up
Core item `0806_pdf_images_tables_and_extraction_strategy.md` tracks richer image/table/OCR support. That work must keep capability reporting truthful and must not reintroduce PyMuPDF-family packages into default install profiles.
