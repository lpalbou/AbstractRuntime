# 0069 — Document renderers: image embedding, hr/blockquote, KeepTogether

- **Status**: COMPLETE 2026-07-21 — slice 1 (image embedding, both writers) by the flow seat (c3584); slices 2+3 (hr/blockquote both writers + bounded KeepTogether for fences in PDF) by runtime under the backlog dispatch (c3815); pinned in tests/test_document_renderers_hr_quote.py. Column-width weighting deliberately NOT taken: _table_column_widths already weights by header length and no live report has shown the starvation case — re-open on evidence.
- **Owner**: runtime (documents/ package)
- **Origin**: flow's co-scientist report-quality wave (agora commons c3166,
  2026-07-19; two fable5 adversaries independently flagged; full context
  in abstractflow docs/backlog/completed/0147). Non-urgent by the filer's
  own ranking — entity cognition holds priority; this banks the record.

## Problem

Professional reports generated through `documents/pdf.py` + `docx.py` hit
three rendering ceilings:

1. **No image embedding** (the ceiling on real figures): `pdf.py` has no
   ReportLab `Image` flowable and no `![](path)` handling; `docx.py` has
   no `add_picture`. Co-scientist figures are ASCII-only today, while the
   operator explicitly asked for professional charts/diagrams.
2. **`---` and `>` render as literal text**: no hr/blockquote branch in
   `pdf.py` — a report's caveat block shows literal `> ` characters and
   stray `---` paragraphs on page 1.
3. **No KeepTogether for fenced blocks + no column width weighting**:
   ASCII diagrams split across page breaks mid-box; prose columns starve
   beside near-equal numeric columns.

## Direction (adversary-verified feasibility, per the filer)

- `![alt](workspace-relative.png)` branch in BOTH writers: ReportLab
  `Image` + python-docx `add_picture` are native; matplotlib is already
  probed by pdf.py for fonts, so no new hard dependency.
- Map `---`/`***` to a thin rule/spacer; `>` to an indented quote style.
- Wrap fenced blocks in `KeepTogether`; weight title/prose columns ~1.5
  vs ~0.5 for short numeric headers.

## Acceptance

- A markdown report with an image ref, an hr, a blockquote, a fenced
  ASCII diagram near a page break, and a mixed prose/numeric table
  renders professionally in both PDF and DOCX (no literal `>`/`---`,
  no split diagram, figure embedded).
- Image paths resolve workspace-relative only (no absolute-path reads
  outside the run workspace — the existing path-discipline rules apply).
- Flow consumes it in co-scientist (their commitment on the thread).
