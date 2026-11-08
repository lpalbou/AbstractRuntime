### Fixed

- **The requested iteration budget is the one the agent obeys** (`visualflow_compiler/visual/executor.py`,
  `core/runtime.py`). A flow whose On Flow Start node has a `max_iterations` pin with an authored default (the
  shipped basic-agent: 20) used to send that default through the data edge into the Agent node, which overwrote the
  run's `_limits.max_iterations` in the react child: a run that asked for 50 through `_limits` (the gateway default,
  the kit field) ran 20. The pin default no longer stands in for the run's budget; an explicit flat `max_iterations`
  input still counts as the request. START_SUBWORKFLOW now carries the parent's `_limits.max_iterations` into a child
  flow (an explicit child value wins), so one value governs the whole run tree.

### Changed

- **Default iteration budget 50, one constant** (`core/vars.py` `DEFAULT_MAX_ITERATIONS`, exported from
  `abstractruntime`): `ensure_limits`, `RuntimeConfig.max_iterations`, the visual Agent node, the agent adapter, the
  limit status/warnings and the entity visit loop read it; it was 20 (operator 2026-11-08: "By default, set it to 50
  everywhere"). Hosts and packages import the name instead of copying the number.
