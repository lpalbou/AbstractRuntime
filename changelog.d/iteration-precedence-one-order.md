### Fixed

- **One iteration-limit order everywhere: the person's request > the host's stored default > the workflow's own value >
  50** (`core/vars.py`, `core/runtime.py`, `visualflow_compiler/compiler.py`, `visualflow_compiler/visual/executor.py`;
  review 2026-11-09 finding 2). An Agent node's own `max_iterations` pin default (the Flow editor seeds one on every
  Agent node) beat a requested `_limits.max_iterations` (request 100 ran 20), and a deliberate On Flow Start pin value
  never applied, because `Runtime.start` always seeds `_limits.max_iterations` (pin 5 ran 50). `Runtime.start` now
  stamps `_limits.max_iterations_declared` (true when the caller named `_limits.max_iterations`, i.e. a request or a
  host default; `declared_max_iterations(limits)` reads it): a declared budget wins at the start node, at every Agent
  node (over its edge, pin default and config) and in child flows (over a subflow pin's flat value); otherwise the
  workflow's own value (start-pin default, Agent-node pin default or config) wins over the seeded
  `DEFAULT_MAX_ITERATIONS`. An explicit child `_limits.max_iterations` (a delegated sub-agent's narrower budget) is
  kept. Note: a shipped workflow that authors its own value (basic-agent's start pin 20) now runs that value when
  nothing is requested and no host default is stored.
