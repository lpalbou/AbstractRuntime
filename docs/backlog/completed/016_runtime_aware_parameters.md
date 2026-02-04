## 016_runtime_aware_parameters (completed)

### Goal
Make core runtime limits durable and introspectable via a canonical `_limits` namespace in `RunState.vars`.

This enables:
- persisted “runtime parameters” (survive restart/resume)
- host/UI introspection (polling `get_limit_status`)
- workflow/agent-driven enforcement (hybrid model)

### What shipped (in this repository)

#### RuntimeConfig
`src/abstractruntime/core/config.py`
- `RuntimeConfig` dataclass
- `to_limits_dict()` initializes `_limits` for new runs

#### `_limits` helpers
`src/abstractruntime/core/vars.py`
- `LIMITS = "_limits"`
- `get_limits(...)` / `ensure_limits(...)`

#### Limit warnings + APIs
`src/abstractruntime/core/models.py`
- `LimitWarning`

`src/abstractruntime/core/runtime.py`
- `Runtime.get_limit_status(run_id)`
- `Runtime.check_limits(run_state)`
- `Runtime.update_limits(run_id, updates)`

#### Exports
`src/abstractruntime/core/__init__.py` exports `RuntimeConfig`, `LimitWarning`, and `_limits` helpers.

### Design notes

- **Hybrid enforcement**: the runtime surfaces status/warnings; workflows/agents decide what to do when limits are reached.
- **Durability**: `_limits` lives in `RunState.vars` so it is persisted by `RunStore`.
- **Token usage is best-effort**: the runtime may update `estimated_tokens_used` from `LLM_CALL` usage metadata when available.

### Tests

Relevant coverage:
- `tests/test_runtime_config_max_output_tokens_fallback.py` (RuntimeConfig → `_limits`)
- many runtime/effect tests construct `RunState.vars["_limits"]` explicitly to exercise durable behavior

