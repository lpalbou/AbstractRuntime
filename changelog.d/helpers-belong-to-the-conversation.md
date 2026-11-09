### Changed

- **A local helper belongs to its conversation, not to the run that started it** (operator decision
  2026-11-09). A dev server an agent started in Code with `local_helper_start` died when the answer was
  delivered: each turn is one run, and the run's end closed its helpers (`browser_probe` then reported
  `ERR_CONNECTION_REFUSED`). A helper started by a run of a session is now kept by the session: it
  outlives the run, and the session's later runs reach it by its `helper_id` (`local_helper_status`,
  `local_helper_stop`). It stops when the agent or the operator stops it, when the host closes the
  session's helpers (`close_session_helpers(runtime, session_id)`, called when the session is archived
  or ends) or when the host stops (`get_local_helper_registry().close_all()`); never at run completion.
  A run without a session keeps the old scope (its helpers end with it). The scope names the runtime
  that holds the run, so the same session id in another store never reaches them. The three tools'
  descriptions say the new lifetime. `register_shell_session_teardown` registers the runtime's session
  scope (`register_session_scope`).
