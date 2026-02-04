# Runtime limits (`_limits`)

AbstractRuntime stores runtime-facing limits in a canonical `RunState.vars["_limits"]` dict. This is used for **durable configuration** (persisted in checkpoints) and for **host/agent introspection**.

Implementation pointers:
- `_limits` helpers/namespace constants: `src/abstractruntime/core/vars.py`
- limit config source of truth: `src/abstractruntime/core/config.py` (`RuntimeConfig`)
- limit APIs: `src/abstractruntime/core/runtime.py` (`get_limit_status`, `check_limits`, `update_limits`)

## What `_limits` contains

`Runtime.start(...)` initializes `_limits` from `RuntimeConfig.to_limits_dict()` when missing. (`src/abstractruntime/core/runtime.py`, `src/abstractruntime/core/config.py`)

Shape (keys are stable; values may be `None` when unknown):

```python
run.vars["_limits"] = {
    "max_iterations": 25,
    "current_iteration": 0,
    "max_tokens": 32768,          # context window (fallback when unknown)
    "max_output_tokens": None,    # provider/model dependent
    "max_input_tokens": None,     # optional budget cap for inputs
    "estimated_tokens_used": 0,   # best-effort (see below)
    "max_history_messages": -1,   # -1 = unlimited
    "warn_iterations_pct": 80,
    "warn_tokens_pct": 80,
}
```

Notes (as implemented today):
- `current_iteration` is **not** automatically incremented by the runtime; higher-level loops (agents/workflows) should update it if they want iteration budgeting.
- `estimated_tokens_used` is a **best-effort, last-known** value. The runtime updates it from `LLM_CALL` usage metadata when available (`src/abstractruntime/core/runtime.py`). It is not guaranteed to be tokenizer-accurate and is not accumulated across calls.

## Configuring limits

You can pass a `RuntimeConfig` when constructing a `Runtime`:

```python
from abstractruntime.core import Runtime, RuntimeConfig
from abstractruntime.storage import InMemoryLedgerStore, InMemoryRunStore

rt = Runtime(
    run_store=InMemoryRunStore(),
    ledger_store=InMemoryLedgerStore(),
    config=RuntimeConfig(
        max_iterations=50,
        max_tokens=65536,
        warn_iterations_pct=75,
    ),
)
```

If you use the AbstractCore convenience factories, they also accept `config=` and may populate model capabilities (`src/abstractruntime/integrations/abstractcore/factory.py`).

## Introspection and updates

### `Runtime.get_limit_status(run_id)`

Returns a structured dict for UI/status display. (`src/abstractruntime/core/runtime.py`)

### `Runtime.check_limits(run_state)`

Returns a list of `LimitWarning` objects for limits approaching/exceeded. (`src/abstractruntime/core/models.py`, `src/abstractruntime/core/runtime.py`)

As of v0.4.0, warnings are computed for:
- `iterations` (`current_iteration` vs `max_iterations`)
- `tokens` (`estimated_tokens_used` vs `max_tokens`)

### `Runtime.update_limits(run_id, updates)`

Updates selected keys in `_limits` durably (saved via the configured `RunStore`). Unknown keys are ignored. (`src/abstractruntime/core/runtime.py`)

Example:

```python
rt.update_limits(run_id, {"max_tokens": 131072, "warn_tokens_pct": 85})
```

## See also

- `architecture.md` — where `_limits` fits in the runtime
- `integrations/abstractcore.md` — where token usage metadata typically comes from (`LLM_CALL`)
- `manual_testing.md` — quick manual checks + running tests

