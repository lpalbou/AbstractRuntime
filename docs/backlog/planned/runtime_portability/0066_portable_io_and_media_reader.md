# Planned: Portable I/O + media-reader hardening (the any-OS papercuts)

## Metadata
- Created: 2026-07-13
- Status: Planned
- Completed: N/A
- Priority: P1 (silent degrade / crash on Windows + network FS; not destructive)
- Area: storage, utils/atomic_files, integrations/abstractcore/llm_client
- Source: 2026-07-13 portability adversary (fable5) findings 3, 7, 8, 9, 10, 13

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
Beyond the destructive process layer (0065), the runtime's file and
subprocess I/O carries a cluster of portability defects that silently
degrade or crash off macOS/Linux. Gathered here as one focused I/O item so
they land together rather than as scattered one-liners.

## Current code reality (line-verified 2026-07-13)
- Local media subprocess reader (llm_client.py:1652, 1787,
  `_run_local_image_subprocess` / `_run_local_video_subprocess`) uses
  `selectors.DefaultSelector`. On Windows that is `SelectSelector`, whose
  `select()` accepts ONLY sockets → the first `selector.select()` raises
  `OSError` (WSAENOTSOCK), uncaught (try/finally, no except). All local
  in-process image/video generation crashes on Windows. Remote providers
  unaffected.
- `os.replace` is used unguarded in five hot sites: `JsonFileRunStore.save`
  (json_files.py:149 via tmp.replace), `atomic_write_text`
  (utils/atomic_files.py:45), the artifact store (storage/artifacts.py), and
  the command store (storage/commands.py). On Windows, `os.replace` onto a
  target held open by any reader (Python opens without FILE_SHARE_DELETE) or
  touched by an AV scanner throws transient `PermissionError`; `save()` has
  no retry, so the tick loop dies on a timing accident. POSIX unaffected.
  (The AV-scanner PermissionError is already a known workspace incident
  class.)
- SQLite WAL (storage/sqlite.py `_apply_pragmas`) over NFS/SMB: the `-shm`
  file needs coherent mmap; cross-machine access is unsupported by SQLite
  (corruption / "database is locked"). Container bind-mounts of homes onto
  network volumes hit this. `close()`'s `wal_checkpoint(TRUNCATE)` makes a
  copy clean only for threads that call close().
- Timezone detection (llm_client.py `_detect_timezone_name`, ~470): no `TZ`
  / `/etc/timezone` / `/etc/localtime` on Windows → returns None, grounding
  envelope silently loses timezone/country.
- RSS metric (visualflow_compiler/visual/execution_metrics.py
  `process_rss_mb`): no `resource` module on Windows → None (honest degrade,
  low stakes).
- Slug/path handling: Windows reserved device names (con/nul/aux/com1…) and
  trailing dot/space in user-supplied entity slugs / `.flow` install names
  create undeletable/aliased paths; MAX_PATH 260 for deep artifact trees
  without the long-path opt-in. (The path-normalization layer itself is
  already portable — verified: `workspace_paths.py` normalizes `\`→`/`,
  `os.path.samefile` uses file IDs on Windows.)

## Problem
Local media generation crashes on Windows; the tick loop can die on a
Windows replace timing accident; network-mounted homes corrupt silently;
grounding and metrics degrade without a label; reserved slugs create
un-deletable homes.

## What we want to do
- Media reader: replace the `selectors` loop with two daemon reader threads
  draining stdout/stderr into the existing consumers (the standard portable
  shape). Drop `selectors` here.
- `robust_replace(tmp, dst)`: one shared helper — bounded retry (~5 × 20 ms
  backoff) on `PermissionError`, then raise. Swap into all five os.replace
  sites (compose with 0048's durable-IO unification — the helper lives in
  the same module).
- WAL escape hatch: `ABSTRACTRUNTIME_SQLITE_JOURNAL_MODE=delete` env knob
  (labeled `#FALLBACK` when engaged) + a docs line: homes live on local
  filesystems; network mounts are out-of-contract. Do NOT attempt network-FS
  auto-detection (unreliable).
- tz: optional `tzlocal` import (CLDR Windows-registry→IANA), labeled
  fallback; stdlib alone cannot get an IANA name on Windows. Dependency
  policy (operator 2026-07-13 21:23): tzlocal is acceptable ONLY as an
  OPTIONAL import (never hard) — it is MIT, pure-Python, tri-OS, which meets
  the light+permissive+cross-OS bar; the labeled degrade (timezone omitted)
  is the zero-dep default when it is absent.
- Slug validator: reject Windows reserved names + trailing dot/space at the
  entity-slug / bundle-name boundary; note the long-path registry opt-in in
  install docs.
- RSS: leave as honest None, or ctypes `GetProcessMemoryInfo` only if metric
  parity is wanted (low priority).

## Why
Local media, durable saves, and grounding should not silently break or crash
on a supported OS; a network-mounted home should fail loud with a remedy,
not corrupt.

## Requirements
- POSIX behavior unchanged.
- `robust_replace` retry bounded and then loud (never an infinite spin;
  never a swallow).
- The WAL knob is opt-in and labeled; default stays WAL on local FS.
- New deps are OPTIONAL imports only (tzlocal), never hard.

## Suggested implementation
Media reader ~S–M; robust_replace ~S (+ compose with 0048); WAL knob ~S; tz
~S; slug validator ~S. Land as one PR; each is independently testable.

## Scope
The six surfaces above + their tests + a docs line on network-mounted homes.

## Non-goals
- No multi-machine home support (the WAL knob is a local-container escape
  hatch only).
- No path-layer rewrite (already portable).

## Dependencies and related tasks
- 0048 (durable-IO unification) owns the shared write module `robust_replace`
  belongs in; 0065 (process/lock) is the sibling portability item and should
  land first (it is P0).

## Expected outcomes
Local media generation runs on Windows; a Windows replace timing accident
retries instead of killing the tick; a network-mounted home fails loud with
a documented remedy; grounding/metrics degrade with a label. POSIX suite
green; Windows CI smoke exercises the media path + a replace-under-open-handle.

## Validation
- Windows CI (the 0065 lane): local image gen returns bytes; a save while a
  reader holds the file open succeeds via retry.
- WAL knob: setting it flips journal mode + emits the #FALLBACK label
  (unit).
- Slug validator: reserved names refused (unit).

## Progress checklist
- [ ] Media reader → daemon threads (drop selectors)
- [ ] robust_replace helper + five call-site swaps
- [ ] WAL journal-mode env knob + docs line
- [ ] Optional tzlocal fallback
- [ ] Reserved-name slug validator
- [ ] Windows CI coverage for media + replace-under-open

## Guidance for the implementing agent
These are independent; ship whichever unblocks a real Windows user first
(media reader and robust_replace are the crash/loop-death ones). Keep every
fix a thin per-OS branch with POSIX unchanged.
