# 0858 — Typed runs: one `run_type` stamped at start, indexed, carried in the replay view

**Status**: planned · **Priority**: P1 (high) · **Created**: 2026-10-28
**Package**: abstractruntime (enum, stamp, index column, replay view field, backfill helper) · **Slices**: abstractgateway 0951 (start paths, `/runs` filters, migration), clients (framework root 1065)
**Master item**: framework root backlog 1063, `abstractframework/docs/backlog/planned/1063_typed_runs_and_sessions.md`; links root 0993, 1059

## Summary

The runtime owns the vocabulary of what a run is. Today it has two partial ones: `core/run_attribution.py`
derives `session_kind` (`chat|automation|occurrence|discussion`) and `role` at index time from automation
metadata, so every non-automation run is `chat`; `core/run_origin.py` stamps `_meta.origin`
(`person|trigger|agent|system`, `actor`, `client`, `ref`) once at start but nothing indexes it and the
replay view does not carry it. Add one `_meta.run_type` (enum in root 1063: `conversation`, `docs`,
`automation_controller`, `occurrence`, `discussion`, `scheduled`, `entity_visit`, `entity_life`, `import`,
`correction`, `bridge`, `flow`, `subrun`) and `origin.surface` (`desktop|iphone|watch|web|terminal|console|api|none`),
written once like `stamp_origin`, indexed as columns, and exposed in `replay_view_v1` as `run.type` and `run.origin`.

## Current code reality (origin/main, abstractruntime 0.15.0)

- `core/run_attribution.py`: `SESSION_KINDS`, `ROLES`, `INDEX_FIELD_NAMES`, `automation_index_fields(vars)`; first-match derivation from `_meta.automation` / `_meta.occurrence` / `_meta.discussion` / `_meta.schedule`; columns in `storage/sqlite.py` and `storage/json_files.py`.
- `core/run_origin.py`: `ORIGIN_KINDS`, `origin_for`, `origin_of`, `stamp_origin` (written once; an existing origin is kept); the automation controller stamps occurrences (`automations/controller.py`).
- `replay/view.py`: no type or origin on the view.
- Entity homes (`identity/chat.py`, life loop) write runs under synthetic ids for the in-process chat path (memory report decision 3), so `entity_visit`/`entity_life` need the gateway hosts and decision 3 to stamp them.

## Chosen route

1. `core/run_type.py`: `RUN_TYPES`, `stamp_run_type(vars, type)` (once; refuses an unknown value), `run_type_of(vars)` with the backfill derivation for runs that predate the stamp (returns `(type, "stamped"|"derived")`).
2. `Runtime.start` refuses a root run with no `run_type` after one release of warnings (descendants default to `subrun`).
3. Index columns `run_type`, `origin_kind`, `origin_surface` (SQLite migration + JSON index); `list_runs(run_types=..., surfaces=...)`.
4. `run_attribution` derives `session_kind`/`role` from `run_type` for one release (compatibility), then is reduced to the automation fields.
5. `replay_view_v1`: `run.type`, `run.origin` (additive); conformance fixture updated.

## Open decisions

See root 1063 (enum names, `surface` placement, retirement of `session_kind`, interaction with root 1024).

## Acceptance criteria

- [ ] Every start path in the runtime stamps `run_type`; tests per path red when the stamp is removed.
- [ ] Index columns filled for new runs; the backfill helper derives the same type the old fields implied on a fixture of legacy runs.
- [ ] View fixture carries `run.type` and `run.origin`.
