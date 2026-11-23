### Added

- **One contract for "the process behind a pid", with a Windows implementation** (`utils.os_process`): `exists`,
  `reap`, `state` (`alive` / `gone` / `unknown`), `start_token`, `command`, `terminate`, `direct_python` and
  `detached`. POSIX keeps `os.kill(pid, 0)`, `os.waitpid` and `ps` as before. Windows asks the kernel through a
  process handle (ctypes, no new dependency): none of the three exists there. The incarnation token written in
  `loop_status` is unchanged on POSIX; on Windows it is `win-filetime:<process creation time>`.

### Fixed

- **An entity's own time runs on Windows** (`identity.life`). The own-time loop could not start there: it crashed
  at its boot on `os.waitpid(pid, os.WNOHANG)` (`os.WNOHANG` does not exist on Windows), and the `starting` status
  it left made every later `read_loop_status` raise, so the host showed "not running" and each Retry met the same
  error. Liveness, the incarnation token, the command line and the Stop now go through `utils.os_process`
  (`_pid_alive`, `_pid_alive_reaped`, `_pid_state`, `_pid_start_time`, `_pid_command` keep their names).
- **The pid recorded for a spawned loop is the loop's own** (`spawn_loop_process`). In a Windows virtual
  environment `sys.executable` is a launcher whose child is the interpreter, so the recorded pid named the launcher
  and the loop's boot claim would refuse the home as held by another process. The loop is started with the
  interpreter itself (`os_process.direct_python`, the way `multiprocessing` starts its children) and keeps the
  environment.
- **A Stop ends the loop on Windows** (`hard_stop_loop`). It raised on `signal.SIGKILL` after own time was revoked
  and before `stopped` was written, and could verify no loop at all (no token, no command line). It now ends the
  verified incarnation through a handle whose creation time is checked against the recorded token; `signal` in its
  result is `TerminateProcess` there.
- **A loop is recognised by its arguments, not by a split at the first space** (`_is_loop_invocation`): where the
  system gives the exact argument list (Windows), each argument is compared, the interpreter's path is not read, a
  home path with a space is one argument, and a home named by another spelling of the same folder (letter case, an
  8.3 short name) is the same home.
- **A loop started from a console is not ended by a Ctrl+C in that console on Windows** (`os_process.detached`:
  its own process group and a console without a window, where POSIX starts a new session).
- **The home's locks hold across processes on Windows** (`utils.file_lock`: `fcntl.flock`, `LockFileEx` on
  Windows; shared and exclusive, waiting without a limit, released when the file is closed). The home state lock,
  both spawn locks, the lifecycle write barrier and the task inbox append took `fcntl.flock` and, where it does
  not exist, fell back to a lock inside one process or to none. So on Windows the host and the own-time loop did
  not exclude each other: two simultaneous starts both launched a loop, and a Stop's generation bump could land
  between a write's fence check and the write. On Windows the lock sits on one byte far past the end of the file,
  so the content of a locked file (the task inbox) stays readable and writable.
