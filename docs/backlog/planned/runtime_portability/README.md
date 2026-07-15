# Runtime portability backlog track

## Status
Planned.

## Purpose
Make abstractruntime — and therefore the summoned-entity substrate the whole
framework is built toward — run correctly on ANY OS, not just macOS/Linux.
The operator's constraint is absolute ("portable and work on any OS"). A
2026-07-13 two-adversary survey (portability + performance/scale, both
fable5) found that the runtime's most load-bearing recent work — entity
loop supervision, the one-writer-per-home lease, the operator kill switch,
local media generation — rests on POSIX-only primitives that are **broken or
outright destructive on Windows**, and silently weakened in slim Linux
containers and on network filesystems.

The headline: on Windows `os.kill(pid, 0)` calls `TerminateProcess` — so a
`read_loop_status` GET *kills the entity's own-time loop*. No Windows
deployment can even reach the scenarios the lease protects, because the
first status poll terminates the thing being protected.

## Items
- `0065_portable_process_and_lock_layer.md` (P0): the destructive-on-Windows
  process layer — non-lethal liveness, a `flock`↔`msvcrt` lock shim, correct
  detach/terminate, a locale-free start-time token. The "entity substrate on
  any OS" story; also pays on POSIX (deletes the `ps` subprocess + its
  LC_ALL workaround; fixes the silent pid-guard loss in distroless/alpine).
- `0066_portable_io_and_media_reader.md` (P1): the remaining portability
  papercuts as one focused I/O item — a robust `os.replace` retry (Windows
  AV/open-handle `PermissionError`), the local media subprocess reader
  (`select()` rejects non-sockets on Windows → local image/video generation
  crashes), a WAL-journal-mode escape hatch for network-mounted homes, a
  reserved-name slug validator, and honest degrades for tz/RSS.

## Reading order
0065 first (it is P0 and blocks any real Windows deployment); 0066 second
(papercuts that matter once 0065 lets Windows run at all). Both compose with
the existing `runtime_systemic_reliability` track — 0065's lock shim and
0066's robust-replace are consumed by the durable-I/O items there (0048,
0067).

## Governing ADRs
None identified after review. If a Windows CI lane becomes a standing
requirement (see below), that is an ADR-grade process decision — flag it
then.

## Scope
Portability of the runtime's OS-touching surfaces: process supervision,
cross-process mutual exclusion, atomic file writes, subprocess I/O, path
handling, SQLite journal mode. Stdlib-only fixes are strongly preferred; a
new hard dependency (e.g. psutil) must be justified, not defaulted.

## Non-goals
- No behavior change on POSIX beyond deleting now-redundant workarounds.
- No multi-machine shared-home support (network filesystems stay
  out-of-contract; the WAL knob is a local-container escape hatch, not a
  distributed-storage feature).
- No new async/eventing abstractions.

## Notes for future agents
- A Windows CI smoke job is the ONLY thing that keeps these fixes true over
  time — the survey documented CPython Windows behavior but could NOT execute
  on Windows. Treat "add a Windows CI lane" as part of 0065's work item, not
  an afterthought.
- The findings are behavioral claims about CPython on Windows; re-verify
  each against the running interpreter before shipping the fix, and prefer a
  thin per-OS shim with the POSIX path unchanged (lowest regression risk to
  the working macOS/Linux deployments).
