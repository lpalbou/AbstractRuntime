### Fixed

- **A cancelled turn stops at its next effect boundary and forms nothing** (`identity/chat.py`, `identity/tools.py`,
  `identity/authoring.py`, `identity/life.py`; Codex review 2 #2, runtime half). `ChatSession.cancel_check` (set by
  a host: the gateway's Stop fence; the own-time loop sets it to its moved lifecycle generation) is checked before
  every model call, before a tool batch and before EACH tool in it (`execute_tool_elections(should_stop=)`; the
  elections left carry `cancelled`), and before the turn forms anything. The turn raises `TurnCancelled(report,
  where)`: the report keeps only the tools that really ran (and their files), ends with a `#NOTE the turn was
  cancelled ...` line, and nothing is formed (no elections, diary, commit, episode or authoring). The authoring
  drainer takes `should_stop=` too: checked before any re-arm or claim, before each task and again after the
  capability answers - a Stop then writes nothing and hands the tasks back unspent. The own-time loop ends a
  cancelled tick with `stopped_by=kill_switch` (no failure, no backoff).
