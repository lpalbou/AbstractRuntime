### Fixed

- **A parent run ends CANCELLED when its child was stopped** (Fable review F2, P2). The fenced effect table checked
  only a model call's outcome after it returned, and the Runtime's own built-in handlers (`START_SUBWORKFLOW`,
  `ASK_USER`, `ANSWER_USER`, the waits, `MEMORY_QUERY` ...) were not in the fenced table at all: a child the Stop
  cancelled came back as `completed({cancelled: True})` and the parent went on to end COMPLETED. Every effect's
  outcome is now checked after it returns, and `open_entity_runtime` fences the Runtime's whole table, built-ins
  included (two fence checks per effect).
