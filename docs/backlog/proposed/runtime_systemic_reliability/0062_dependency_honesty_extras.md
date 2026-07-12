# Proposed: Dependency honesty — "minimal execution substrate" vs the batteries

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: None (whichever way it resolves, it is a packaging/docs
  decision recorded in the item)

## Context
The package docstring says "minimal execution substrate" (__init__.py:6)
and docs/architecture.md says "dependency-light".

## Current code reality (verified 2026-07-12)
pyproject.toml:36-60 hard-requires: pandas, unstructured[docx,odt,pptx,
rtf,xlsx], reportlab, python-pptx, Pillow, openai, plus
abstractcore[remote,tools,vision,voice,audio,music]. The kernel and the
batteries ship as one wheel.

## Problem or opportunity
The claim and the wheel disagree. Anyone embedding the runtime as "just the
durable kernel" pulls the full document/media stack; install size and
surface area contradict the stated posture.

## Proposed direction (two honest options; pick ONE on promotion)
- (a) CHEAPEST: re-scope the claim — docs say the batteries-included truth.
- (b) STRUCTURAL: split extras `abstractruntime[documents]`,
  `abstractruntime[media]` with loud actionable ImportErrors at the effect
  handler boundary (the codebase's existing missing-plugin pattern), default
  install = kernel + abstractcore basics. COORDINATE WITH GATEWAY before
  changing the default install (they depend on the current wheel contents).

## Why it might matter
Honesty of the public claim; embedding use cases; install size for thin
hosts.

## Promotion criteria
The 1.0 packaging pass, or an actual embedding consumer asking for the thin
wheel.

## Validation ideas
(b): fresh-venv install matrix (base / [documents] / [media]) with the
document/media effect tests skipping-or-failing loudly per matrix cell.

## Non-goals
No removal of capabilities; no silent fallbacks when an extra is missing
(the missing-plugin error pattern is the law).

## Guidance for future agents
Option (a) is fifteen minutes and fully honest — do not let option (b)'s
size make the claim stay wrong in the meantime.
