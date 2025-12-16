## Runtime Limits and Warnings

AbstractRuntime provides infrastructure for tracking and managing resource limits during workflow execution. This enables runtime-aware behavior for iterations, tokens, and history management.

---

### Overview

The limits system uses a canonical `_limits` namespace in `RunState.vars` as the single source of truth for all runtime resource limits. The Runtime provides methods to query limit status and check for warnings, while workflow nodes are responsible for enforcement.

**Design principle**: Hybrid enforcement
- **Runtime provides**: Storage, introspection, status, warnings
- **Workflow nodes enforce**: Check limits and transition when exceeded

---

### The `_limits` Namespace

Every run has a `_limits` namespace initialized from `RuntimeConfig`:

```python
run.vars["_limits"] = {
    # Iteration control
    "max_iterations": 25,
    "current_iteration": 0,

    # Token/context window
    "max_tokens": 32768,
    "max_output_tokens": 4096,
    "estimated_tokens_used": 0,

    # History management
    "max_history_messages": -1,  # -1 = unlimited

    # Warning thresholds (percentage)
    "warn_iterations_pct": 80,
    "warn_tokens_pct": 80,
}
```

---

### RuntimeConfig

Configure limits when creating a runtime:

```python
from abstractruntime.core import RuntimeConfig

config = RuntimeConfig(
    max_iterations=50,
    max_tokens=65536,
    max_history_messages=100,
    warn_iterations_pct=75,
    warn_tokens_pct=80,
)

runtime = create_local_runtime(
    provider="ollama",
    model="qwen3:4b",
    config=config,
)
```

---

### Runtime Methods

#### `get_limit_status(run_id) -> Dict`

Returns structured status for all limits:

```python
status = runtime.get_limit_status(run_id)

# Example output:
{
    "iterations": {
        "current": 5,
        "max": 25,
        "pct": 20.0,
        "warning": False,  # True if pct >= warn_iterations_pct
    },
    "tokens": {
        "estimated_used": 8192,
        "max": 32768,
        "pct": 25.0,
        "warning": False,
    },
    "history": {
        "max_messages": -1,  # -1 = unlimited
    },
}
```

**Use cases**:
- UI status displays
- CLI progress indicators
- Logging and monitoring

#### `check_limits(run) -> List[LimitWarning]`

Returns a list of warnings for limits approaching or exceeded:

```python
from abstractruntime.core import LimitWarning

warnings = runtime.check_limits(run)

for w in warnings:
    print(f"{w.limit_type}: {w.status} ({w.current}/{w.maximum} = {w.pct}%)")
    # Output: "iterations: warning (20/25 = 80.0%)"
```

**LimitWarning fields**:
- `limit_type`: `"iterations"`, `"tokens"`, or `"history"`
- `status`: `"warning"` (at threshold) or `"exceeded"` (at/over limit)
- `current`: Current value
- `maximum`: Maximum allowed
- `pct`: Percentage used (computed)

#### `update_limits(run_id, updates) -> None`

Update limits mid-session:

```python
# Increase token limit dynamically
runtime.update_limits(run_id, {"max_tokens": 131072})

# Allowed keys:
# - max_iterations, max_tokens, max_output_tokens
# - max_history_messages
# - warn_iterations_pct, warn_tokens_pct
# - estimated_tokens_used, current_iteration
```

---

### Agent-Level Access

Agents expose limit methods that delegate to the runtime:

```python
agent = ReactAgent(runtime=runtime, max_iterations=25)
run_id = agent.start("Solve this problem")

# Check status
status = agent.get_limit_status()
if status.get("iterations", {}).get("warning"):
    print("Approaching iteration limit!")

# Update limits
agent.update_limits(max_tokens=65536)
```

---

### Current Integration Status

#### What's Implemented

| Feature | Status | Location |
|---------|--------|----------|
| `_limits` namespace | Implemented | `RunState.vars["_limits"]` |
| `RuntimeConfig` | Implemented | `core/config.py` |
| `LimitWarning` | Implemented | `core/models.py` |
| `Runtime.get_limit_status()` | Implemented | `core/runtime.py` |
| `Runtime.check_limits()` | Implemented | `core/runtime.py` |
| `Runtime.update_limits()` | Implemented | `core/runtime.py` |
| `Agent.get_limit_status()` | Implemented | `agents/react.py`, `agents/codeact.py` |
| `Agent.update_limits()` | Implemented | `agents/react.py`, `agents/codeact.py` |
| Iteration enforcement | Implemented | `react_runtime.py`, `codeact_runtime.py` |

#### What's NOT Yet Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| **Warning surfacing to agents** | Not implemented | Agents don't receive warnings automatically |
| **Observability events** | Not implemented | No events emitted when warnings trigger |
| **Token counting** | Not implemented | `estimated_tokens_used` is placeholder |
| **UI/CLI integration** | Not implemented | Must be polled via `get_limit_status()` |
| **Agent guidance injection** | Not implemented | Could inject "2 turns remaining" in prompt |

---

### Planned Enhancements

#### 1. Agent Guidance Injection

When approaching limits, inject guidance into the agent's prompt:

```python
# In reason_node:
warnings = runtime.check_limits(run)
for w in warnings:
    if w.limit_type == "iterations" and w.status == "warning":
        remaining = w.maximum - w.current
        guidance += f"[NOTICE: Only {remaining} turns remaining. Work efficiently.]"
```

#### 2. Observability Events

Emit events when warnings trigger:

```python
# Future: on_step callback receives warnings
on_step("limit_warning", {
    "limit_type": "iterations",
    "status": "warning",
    "current": 20,
    "maximum": 25,
    "remaining": 5,
})
```

#### 3. Token Estimation

Track token usage through LLM responses:

```python
# After LLM call:
response_tokens = response.get("usage", {}).get("total_tokens", 0)
limits["estimated_tokens_used"] += response_tokens
```

---

### Example: Custom Warning Handler

Until observability is wired, you can check warnings manually:

```python
def run_with_warnings(agent, task):
    run_id = agent.start(task)

    while True:
        state = agent.step()

        # Check for warnings after each step
        warnings = agent.runtime.check_limits(state)
        for w in warnings:
            if w.status == "warning":
                print(f"WARNING: {w.limit_type} at {w.pct}%")
            elif w.status == "exceeded":
                print(f"EXCEEDED: {w.limit_type}")

        if state.status in ("completed", "failed"):
            break

    return state
```

---

### Implementation Files

| File | Purpose |
|------|---------|
| `src/abstractruntime/core/config.py` | `RuntimeConfig` dataclass |
| `src/abstractruntime/core/models.py` | `LimitWarning` dataclass |
| `src/abstractruntime/core/vars.py` | `LIMITS` constant, `ensure_limits()` |
| `src/abstractruntime/core/runtime.py` | Runtime limit methods |
| `abstractagent/adapters/react_runtime.py` | Reads `_limits`, enforces iterations |
| `abstractagent/adapters/codeact_runtime.py` | Reads `_limits`, enforces iterations |
| `abstractagent/agents/react.py` | Agent-level limit methods |
| `abstractagent/agents/codeact.py` | Agent-level limit methods |
