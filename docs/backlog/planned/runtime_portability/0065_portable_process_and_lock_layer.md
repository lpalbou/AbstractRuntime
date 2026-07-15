# Planned: Portable process & lock layer (any-OS entity substrate)

## Metadata
- Created: 2026-07-13
- Status: Planned — POSTPONED (operator 2026-07-13 21:23) pending a testing path
- Completed: N/A
- Priority: P0-when-unblocked (destructive on Windows; blocks any Windows deployment)
- Area: identity/life, storage/lease
- Source: 2026-07-13 portability adversary (fable5) findings 1, 2, 4, 5, 6

## Postponement decision (operator, 2026-07-13 21:23)
The operator will NOT authorize blind Windows development — the survey
documented CPython Windows behavior but could not execute it, and building a
fix nobody can run is unacceptable. This item is POSTPONED until a Windows
testing path exists. The de-blinder is a GitHub Actions `windows-latest`
lane (free for public repos, real Windows): with it, 0065 stops being blind
— the P0 regression pin ("a status read does not kill the loop") runs on a
real Windows runner before the fix is trusted. DECISION OWED FROM OPERATOR:
authorize the Windows CI lane (then 0065 proceeds, tested) OR keep 0065
parked. No POSIX regression risk either way — the fix is a thin per-OS shim
with the POSIX path unchanged; the ONLY reason to wait is Windows
verifiability, which the CI lane resolves. Until the decision lands, the
runtime stays macOS/Linux-only for the entity substrate (documented honest
limit, not a silent gap).

## Dependency policy (operator, 2026-07-13 21:23)
Zero new dependency: stdlib-only, `ctypes` for the Windows calls (no
compiled wheel, no psutil). If any future portability work needs a dep it
must be LIGHT, OSS under MIT/Apache/BSD (never a license that can block us),
and work across Windows/Linux/macOS. That rules psutil out here (compiled;
unnecessary — ctypes suffices).

## ADR status
- Governing ADRs: None
- ADR impact: None for the code; a standing Windows CI lane (see Validation)
  is an ADR-grade process decision — raise it with the `adr` skill if the
  room wants CI portability as durable policy.

## Context
The summoned-entity substrate (the framework's keystone) supervises a
detached own-time loop process, enforces one-writer-per-home with a lease,
and gives the operator a kill switch. All three rest on POSIX-only
primitives. The runtime otherwise targets "any OS" and the operator's
constraint is absolute.

## Current code reality (line-verified 2026-07-13)
- `_pid_alive` (identity/life.py:920) is `os.kill(pid, 0)`. On Windows,
  CPython `os.kill` with any signal other than CTRL_C/CTRL_BREAK calls
  `TerminateProcess(handle, sig)` — so a *liveness probe kills the probed
  process*. Consumers: `read_loop_status` (→ `loop_process_status`, the
  gateway `/loop` status route), the visit door's auto-yield negotiation
  (`await_loop_quiescent`), and `spawn_loop_process`'s running check. Net:
  the first status GET on Windows terminates a live loop (exit 0 → reads as
  a clean death). If `OpenProcess` is instead denied, it reads "dead" — both
  directions wrong.
- `DirectoryLease.acquire` (storage/lease.py:123, `import fcntl` at :133)
  and the loop spawn lock (`spawn_loop_process`, identity/life.py:739) have
  an `ImportError fcntl` degrade. On Windows the lease "acquires" with a
  warning stored ONLY in the metadata dict (no caller surfaces it), and the
  spawn lock's degrade is a bare `pass`. Result: advisory-only, i.e. NO real
  mutual exclusion — two writers over one append-only home = permanent
  hash-chain fork / journal seq interleave, the exact damage the module
  exists to prevent.
- `_pid_start_time` (identity/life.py:112) shells out to `ps -p <pid> -o
  lstart=` (pinned LC_ALL=C/TZ=UTC to make the string comparable). No `ps`
  on Windows → token None → the pid-reuse guard silently switches off (the
  B1 corpse hole reopens). Also None on slim Linux containers
  (distroless/alpine without procps) — a fully-POSIX host silently loses the
  guard.
- `spawn_loop_process` passes `start_new_session=True` (identity/life.py:881).
  On Windows this kwarg is bound as `unused_start_new_session` (verified in
  the stdlib `_execute_child`) — silently ignored; the loop child stays
  attached to the host console/job, so "survives host restarts" is false and
  closing the gateway console kills the life mid-day.
- `hard_stop_loop` (identity/life.py:590) sends `os.kill(pid, SIGTERM)` then
  escalates to `signal.SIGKILL` (:655). On Windows SIGTERM = immediate
  TerminateProcess (no grace — acceptable for a freeze but undocumented),
  and `signal.SIGKILL` does not exist → `AttributeError` inside a `try` that
  catches only `(OSError, ProcessLookupError)` → crash mid-freeze if
  escalation is reached.

## Problem
On Windows the entity substrate is not merely degraded, it is destructive
(status read kills the loop) and unsafe (advisory-only lease can corrupt a
home). On slim Linux containers the pid-reuse guard silently disappears.

## What we want to do
One thin portable OS-abstraction module (proposed `runtime/os_process.py`,
stdlib-only) with a POSIX path unchanged and a Windows path via `ctypes`,
consumed by identity/life.py and storage/lease.py:
1. `process_alive(pid) -> bool`: POSIX `os.kill(pid, 0)`; Windows
   `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` + `GetExitCodeProcess !=
   STILL_ACTIVE`. NON-LETHAL on every OS.
2. `process_start_token(pid) -> Optional[str]`: an equality-stable,
   locale-free incarnation token — Linux `/proc/<pid>/stat` field 22
   (starttime jiffies; also deletes the LC_ALL workaround), macOS keeps `ps
   -p ... -o lstart=` (or `sysctl`), Windows `GetProcessTimes` creation
   FILETIME via ctypes. Labeled `None` degrade only on genuinely exotic
   platforms.
3. `terminate_process(pid, *, escalate)`: POSIX SIGTERM→SIGKILL with the
   existing grace; Windows single `TerminateProcess`. `getattr(signal,
   "SIGKILL", signal.SIGTERM)` so escalation never `AttributeError`s.
4. `detached_popen(argv, **kw)`: POSIX `start_new_session=True`; Windows
   `creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`.
5. A `directory_lock` shim used by `DirectoryLease` and the spawn lock:
   POSIX `fcntl.flock(LOCK_EX|LOCK_NB)`; Windows `msvcrt.locking(fd,
   LK_NBLCK, 1)` on a 1-byte range. Byte-range locks are kernel-owned
   (crash-release on handle close), do not travel with file bytes
   (copy-inertness preserved), and a second handle in the same process
   conflicts (same in-process testability). `read_directory_lease`'s probe
   ports identically.

## Why
Without this, no Windows deployment of the framework's keystone feature is
possible — and the failure mode is silent then destructive, the worst class.
It also pays on POSIX immediately: deletes the `ps` subprocess + LC_ALL
workaround, and restores the pid-reuse guard in slim containers.

## Requirements
- POSIX behavior byte-for-byte unchanged (the working macOS/Linux
  deployments must not regress). The Windows path is additive.
- The lock shim must preserve flock's four stated properties: crash-release,
  copy-inertness, same-process conflict, works-at-every-entry-point.
- The start-token must be equality-stable for the SAME incarnation and
  differ across a pid reuse — it is compared, never parsed for wall-clock.
- Network-filesystem caveat documented at the lock site: NFSv4 emulates
  flock; SMB does not propagate between machines — multi-machine shared
  homes stay out-of-contract.

## Suggested implementation
~60–100 lines. One module, per-OS branches guarded by `sys.platform` /
`os.name`, ctypes for the Windows calls (no compiled dep). Land the shim,
then swap the six call sites. Keep the labeled degrade for platforms none of
the three branches cover.

## Scope
The module + the six call-site swaps (identity/life.py ×5, storage/lease.py
×1 incl. its spawn-lock twin) + a Windows CI smoke lane.

## Non-goals
- No psutil hard dependency (compiled wheel for a pure-Python runtime is not
  warranted; an OPTIONAL psutil fast-path is acceptable but not required).
- No change to the lease's semantics, only its OS primitive.
- No multi-machine home support.

## Dependencies and related tasks
- Composes with 0048 (durable-IO unification) and 0066 (robust os.replace) —
  same "portable I/O" theme; the lock shim is shared machinery.
- Re-verify against the running Windows interpreter (the survey could not
  execute on Windows).

## Expected outcomes
A status poll never kills a loop on any OS; the lease is real mutual
exclusion on Windows; the pid-reuse guard works in containers; the freeze
verb never `AttributeError`s; a detached loop survives host-console close on
Windows. Full suite green on POSIX with zero behavior diffs; Windows CI
smoke green.

## Validation
- POSIX A/B: full existing suite green, `test_directory_lease.py` +
  `test_entity_life_loop.py` unchanged behavior; assert the `ps` subprocess
  is gone from the loop path.
- Windows CI smoke lane (NEW, mandatory): spawn a loop, GET status (assert
  the loop is STILL ALIVE afterward — the P0 regression pin), acquire the
  lease twice (second refuses), freeze once (no AttributeError), reader in a
  second process sees the lock.
- Container check: distroless/alpine (no procps) — start token present via
  /proc, pid-reuse guard active.

## Progress checklist
- [ ] os_process module (liveness, token, terminate, detached_popen)
- [ ] directory_lock shim + lease/spawn-lock swap
- [ ] six call-site swaps in life.py/lease.py
- [ ] Windows CI smoke lane
- [ ] POSIX suite green + ps-subprocess removal asserted
- [ ] Fable5 adversary folded

## Guidance for the implementing agent
The POSIX path is the reference; make Windows match its CONTRACT, not its
mechanism. The one non-negotiable regression pin is "status read does not
kill the loop" — write it first, watch it fail on a Windows runner, then
fix. Do not import psutil to make this pass; ctypes is enough.
