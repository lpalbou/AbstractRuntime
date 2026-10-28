# 0864 — An index build's journal replay signals no change for runs its scan already filed

**Status**: completed 2026-10-28, released in abstractruntime 0.15.3 · **Priority**: P1 · **Created**: 2026-10-28
**Cross-package**: follow-up of [0863](0863_index_build_never_loses_a_run_created_during_it.md) (0.15.2); found by
AbstractGateway's suite (`tests/test_mail_collector_change_driven.py::test_400_finished_runs_a_change_to_one_loads_exactly_one`:
`assert 400 == 0` with 0.15.2), gateway backlog 0966.

## Why
0.15.2 makes the first sync after an index build re-apply the journal's last 16 KiB (the records before the build's
mark). Each re-applied record whose file exists went through `_index_fields_into_indexes` -> `_note_active_status`,
which bumps the active-index counter and notes the run in the change ring. So after every build the change signal
reported every recent run as changed: the gateway's change-driven mail-notification pass loaded 400 finished runs
it had already decided. The 0.15.2 review recorded the burst as low; the gateway suite showed its cost.

## Change
`_sync_indexes` marks the records that lie before the build's mark (`_journal_replay_end`, set by
`_mark_journal_start` when it rewinds); `_apply_journal_entries(replayed=...)` skips such a record (no index write,
no change note, its pending entry dropped) only when `_filed_unchanged`: the scan memo holds the run's file at its
current identity token, AND the built children and session indexes hold the run, AND the active index agrees with
the file's status. The memo alone is not proof (a `load()` memoizes a file the build's scan never saw), and a `~`
for a re-opened run must still reach the active index. Anything else is applied and signalled as before.

## Evidence
- `tests/test_json_file_run_store_index_build_race.py::test_the_replayed_tail_signals_no_change_for_runs_the_build_already_filed`:
  50 runs saved by the same store object (the gateway's case) and by another one; the builds of the session,
  children and listing indexes leave `changed_run_ids(token_before)` empty. Red with 0.15.2's `json_files.py`
  (50 ids); green now. The three 0.15.2 race tests still pass (20/20 runs).
- `test_a_run_the_build_missed_is_filed_even_when_another_read_memoized_it` (the publication after the scan and
  a `load()` before the first sync: the run is in the session and children indexes) and
  `test_a_replayed_reopen_reaches_the_active_index` (`~` before the mark, RUNNING file after the scan): red with a
  memo-only skip (2 failed), the second also red with the skip before the active check; green now.
- Adversarial review (Opus, read-only): found both cases above (fixed); the byte-position parser matches the
  previous `splitlines()` on 20,000 fuzzed journals; `_journal_replay_end` is set and consumed under the journal
  lock, once.
- AbstractGateway `tests/test_mail_collector_change_driven.py`: 5 passed with this change (1 failed with 0.15.2).
- Full suite: 4043 passed.
