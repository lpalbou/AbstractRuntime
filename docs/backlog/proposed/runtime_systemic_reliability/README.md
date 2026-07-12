# Runtime systemic reliability — proposed band (before 1.0)

## Status
Proposed (promotion target: planned, gated on a 1.0-preparation wave or the
named triggers per item).

## Purpose
The before-1.0 half of the 2026-07-12 two-adversary meta-analysis
(`../../planned/runtime_systemic_reliability/0044_meta_analysis_record.md`).
These are real findings with agreed fix shapes, but no urgency or blocking
risk today — they earn planned status when their trigger fires or when a
1.0 wave opens.

## Items
- `0056_schema_version_and_quarantine_loading.md`: run-file/record schema
  versioning + per-run quarantine on unknown enums (today: one v-next file
  crashes JSON-store directory scans; SQLite silently zombies).
- `0057_trust_boundary_stamp_registry.md`: declarative
  {tool → stamped-arg → source} table replacing the two hand-rolled stamps.
- `0058_terminal_run_archival_helper.md`: explicit, marker-recorded
  archive_terminal_runs(...) (the ~10k-file directory cliff).
- `0059_memory_span_handler_extraction.md`: ~1,700 lines of span/compaction
  logic out of the Runtime class into memory/span_effects.py.
- `0060_visualflow_compiler_decomposition.md`: visual_to_flow (one
  ~4,150-line function) + the two compiler mega-factories, band-by-band.
- `0061_root_api_deprecation_machinery.md`: module __getattr__ deprecation
  shim + public/internal split BEFORE the split waves need it.
- `0062_dependency_honesty_extras.md`: "minimal execution substrate" claim
  vs pandas/unstructured/reportlab hard deps — docs re-scope or extras.
- `0063_b3_envelope_lane_unification.md`: durable bytes == sent bytes for
  the chat-shape final user message (the tool-loop trailing lane exists;
  make chat shape use it; agent coordinates).

## Reading order
Any; 0061 before 0059/0060 if a split wave starts (it is what makes moves
safe to stage).

## Governing ADRs
None — durable rules land in docs/architecture.md per item scope.

## Non-goals
Everything in the planned track's non-goals list (no async kernel, no tick
decomposition, no new eventing, no ledger auto-pruning, no client-class
merge) binds here too.

## Notes for future agents
Each item names its promotion trigger. Do not promote the whole band at
once — these compete with feature work and should ride a deliberate
hardening wave.
