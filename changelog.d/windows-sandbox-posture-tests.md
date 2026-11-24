### Changed

- **The Windows end-to-end sandbox tests cover every workspace shape** (`tests/test_r12_workspace_sandbox_windows.py`).
  Through the runtime's real tool path (gateway-shaped run vars, the TOOL_CALLS handler, AbstractCore's
  `execute_command` and the runtime's local helper), "Allow everything, refuse listed workspaces" with read &
  write and with read-only by default, and "Deny everything, allow listed workspaces", are each checked: refused
  rows, built-in refusals and the other folders of the data folder unreadable and unwritable, read-only rows
  unwritable, an allowed row inside a refused one, the run's own folder and its `TEMP`, and an unlisted folder
  writable only under read & write by default, each with a positive control. The two newer shapes run when
  AbstractCore's Windows sandbox enforces them (`host_sandbox_kind("allowed_only")`, the release after
  abstractcore 2.38.0) and skip with that reason before; on such an older AbstractCore the test that they are
  refused, with their reason, while the run continues still runs.
