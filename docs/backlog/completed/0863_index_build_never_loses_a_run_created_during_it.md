# 0863 — An index build never loses a run another store object creates during it

**Status**: completed 2026-10-28, released in abstractruntime 0.15.2 · **Priority**: P1 · **Created**: 2026-10-28
**Cross-package**: AbstractGateway backlog 0963 (the flaky `test_runs_reads_archive_and_kinds_once_and_sees_its_own_writes`,
404 `session_not_found` on one CI job per run; it blocked the gateway 0.17.2 PR and release runs).

## Why
`JsonFileRunStore._save` appends a new run's `+id` journal line BEFORE it publishes the run file (J53-1: a crash after
publication can never hide a run). An index build (`_ensure_session_index`, `_ensure_children_index`, the active
index; `warm_session_index` at a host's boot) marks the journal offset (`_mark_journal_start`) and then scans the
folder. A run created by ANOTHER store object (another process: a split-mode runner during the API process's boot;
or another object in the same process) whose `+` line landed before the mark and whose file landed after the scan
was in neither: the session index (and the children index and the trusted listing memo) missed it until the full
reconciliation (`RUN_INDEX_FULL_RECONCILE_S`, 600 s). In the gateway, `POST /sessions/{id}/archive` answered 404
`session_not_found` for a session just written (`session_exists` -> `list_run_index(session_id=...)`).

## Change
`_mark_journal_start` sets the first sync's offset to `_journal_replay_start(size)`: the first record boundary within
`JOURNAL_REPLAY_TAIL_BYTES` (16 KiB, ~400 records) before the journal's size at the mark, instead of the size itself.
The first sync after the build re-applies those records; applying is idempotent (a `+` whose file exists is filed
again; one whose file is still absent stays pending, as for any `+`, until the file appears).

Chosen over "a `+` whose file was absent at scan time stays pending": that needs the build to know which `+` lines
precede the mark, i.e. to read the journal before the mark anyway; the tail replay reuses the existing, tested
pending path unchanged. Cost: one read of at most 16 KiB and one stat per record in it, once per index build (a
build happens at boot or after a journal rewrite). Failure mode: a run whose `+` is more than ~400 records before
the mark and still unpublished after the scan (a publication takes milliseconds) is caught by the periodic
reconciliation, which logs what it repairs; an unreadable tail logs a warning and replays nothing (the previous
behaviour).

## Evidence
- `tests/test_json_file_run_store_index_build_race.py`: another store object's new run, with this store's index
  build run between its `+` line and its publication, is listed by `list_run_index(session_id=...)` and by the
  root listing; with a 50 KB journal the replay is bounded to the tail and starts on a record boundary. 2 passed;
  with the previous `json_files.py`, 2 failed. 20/20 plain runs and 20/20 runs beside 12 CPU-busy processes passed.
- The gateway's previous test order (runs written while the boot warm-up builds the index), with the window between
  `+` and publication widened to 0.3 s: 3/3 failed with AbstractRuntime 0.15.1 (the CI's exact 404), 5/5 passed with
  this change.
