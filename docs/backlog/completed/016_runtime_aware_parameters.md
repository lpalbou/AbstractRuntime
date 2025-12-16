## 016_runtime_aware_parameters (completed)

**Status**: Completed
**Completed**: 2025-12-16

---

## Final Report

### What Was Implemented

Refactored AbstractRuntime to be aware of runtime parameters (max_iterations, max_tokens, max_history_messages) with a canonical `_limits` namespace in RunState.vars.

**Approach**: Hybrid enforcement (runtime provides info/warnings, workflow nodes enforce)
**Rollout**: Phased migration (3 phases)

| Component | Description |
|-----------|-------------|
| `RuntimeConfig` | Frozen dataclass for runtime limits and model capabilities |
| `LimitWarning` | Dataclass for limit warnings (warning/exceeded) |
| `LIMITS` constant | `"_limits"` namespace constant in vars.py |
| `ensure_limits()` | Helper to create/get `_limits` namespace with defaults |
| `Runtime.config` | Property to access RuntimeConfig |
| `Runtime.get_limit_status()` | Get structured limit status for a run |
| `Runtime.check_limits()` | Check if limits approaching/exceeded, returns warnings |
| `Runtime.update_limits()` | Update limits mid-session |
| `Agent.get_limit_status()` | Agent-level delegate to runtime |
| `Agent.update_limits()` | Agent-level delegate to runtime |

### Key Features

1. **Canonical `_limits` Namespace**: Single source of truth in RunState.vars
   ```python
   run.vars["_limits"] = {
       "max_iterations": 25,
       "current_iteration": 0,
       "max_tokens": 32768,
       "max_history_messages": -1,
       "estimated_tokens_used": 0,
       "warn_iterations_pct": 80,
       "warn_tokens_pct": 80,
   }
   ```

2. **Hybrid Enforcement Model**:
   - Runtime provides limit info and warnings
   - Workflow nodes enforce (check and transition)
   - Clear separation of concerns

3. **Model Capabilities Integration**:
   - Factory queries LLM client for model capabilities
   - Capabilities flow through RuntimeConfig to `_limits`
   - max_tokens defaults to model's context window

4. **Backward Compatibility**:
   - Falls back to scratchpad if `_limits` missing
   - All new parameters have sensible defaults
   - No breaking changes to existing APIs

### Files Added

| File | Description |
|------|-------------|
| `src/abstractruntime/core/config.py` | RuntimeConfig dataclass with to_limits_dict() and with_capabilities() |

### Files Modified

| File | Changes |
|------|---------|
| `abstractruntime/core/vars.py` | Added LIMITS constant, ensure_limits(), get_limits() |
| `abstractruntime/core/models.py` | Added LimitWarning dataclass |
| `abstractruntime/core/runtime.py` | Added config property and 3 limit methods |
| `abstractruntime/core/__init__.py` | Exported new classes and functions |
| `abstractruntime/integrations/abstractcore/factory.py` | Config parameter, model capabilities |
| `abstractruntime/integrations/abstractcore/__init__.py` | Exported RuntimeConfig |
| `abstractagent/adapters/react_runtime.py` | Uses _limits namespace |
| `abstractagent/adapters/codeact_runtime.py` | Uses _limits namespace |
| `abstractagent/agents/react.py` | Added get_limit_status(), update_limits() |
| `abstractagent/agents/codeact.py` | Added get_limit_status(), update_limits() |
| `abstractagent/logic/react.py` | build_request() accepts vars parameter |
| `abstractagent/logic/codeact.py` | build_request() accepts vars parameter |
| `abstractflow/adapters/agent_adapter.py` | Populates _limits namespace |

### Test Coverage

| Test Suite | Result |
|------------|--------|
| Phase 1: RuntimeConfig tests | PASS |
| Phase 1: vars.py tests | PASS |
| Phase 1: LimitWarning tests | PASS |
| Phase 1: Runtime limit methods | PASS |
| Phase 1: Exports tests | PASS |
| Phase 2: Factory exports | PASS |
| Phase 3: react_runtime | PASS |
| Phase 3: codeact_runtime | PASS |
| Phase 3: Logic classes | PASS |
| Backward compatibility | PASS |
| AbstractRuntime unit tests | 147 passed |
| AbstractAgent unit tests | 5 passed |

### Usage Examples

#### Basic Usage
```python
from abstractruntime.core import RuntimeConfig

# Create runtime with custom limits
config = RuntimeConfig(
    max_iterations=50,
    max_tokens=65536,
    max_history_messages=100,
    warn_iterations_pct=75,
)

runtime = create_local_runtime(
    provider="ollama",
    model="qwen3:4b",
    config=config,
)
```

#### Checking Limit Status
```python
# Get limit status mid-run
status = runtime.get_limit_status(run_id)
print(f"Iterations: {status['iterations']['current']}/{status['iterations']['max']}")
# Output: Iterations: 5/50

if status['iterations']['warning']:
    print("Approaching iteration limit!")
```

#### Updating Limits Dynamically
```python
# Update limits mid-session (e.g., from /max-tokens command)
runtime.update_limits(run_id, {"max_tokens": 131072})
```

#### Agent-Level Access
```python
agent = ReactAgent(runtime=runtime, max_iterations=25)
run_id = agent.start("Solve this problem")

# Check status through agent
status = agent.get_limit_status()

# Update limits through agent
agent.update_limits(max_tokens=65536)
```

### Architecture

```
                    ┌─────────────────────────────────────┐
                    │           RuntimeConfig             │
                    │  (frozen dataclass with defaults)   │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │              Runtime                 │
                    │  - config property                   │
                    │  - start() initializes _limits       │
                    │  - get_limit_status()                │
                    │  - check_limits() → LimitWarning[]   │
                    │  - update_limits()                   │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │         RunState.vars               │
                    │  {                                   │
                    │    "_limits": {                      │
                    │      "max_iterations": 25,           │
                    │      "current_iteration": 0,         │
                    │      "max_tokens": 32768,            │
                    │      ...                             │
                    │    }                                 │
                    │  }                                   │
                    └─────────────────┬───────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  react_runtime  │         │ codeact_runtime │         │  agent_adapter  │
│  (reads _limits,│         │  (reads _limits,│         │  (populates     │
│   enforces)     │         │   enforces)     │         │   _limits)      │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

---

## Original Proposal

### Goal
Make AbstractRuntime aware of runtime parameters (max_iterations, max_tokens, max_history_messages) with:
- Canonical storage in RunState.vars
- Runtime-level introspection and control
- Mid-session updates capability

### Context / Problem
Parameters were scattered with no single source of truth:

| Parameter | Location | Problem |
|-----------|----------|---------|
| `max_iterations` | `scratchpad` | Only in one place, but not runtime-managed |
| `max_tokens` | `Logic._max_tokens` | Lost on resume, not in RunState |
| `max_history_messages` | `Logic._max_history_messages` | Lost on resume, not in RunState |

The Runtime class had **zero awareness** of these limits.

### Non-goals
- No automatic enforcement by runtime (nodes enforce)
- No observability events in this phase (future work)
- No token counting (estimated_tokens_used is placeholder)

---

### Proposed Design

1. **New `_limits` namespace** in RunState.vars as canonical storage
2. **RuntimeConfig dataclass** for configuration and defaults
3. **Runtime enhancements**: config property, limit methods
4. **Hybrid enforcement**: runtime provides info, nodes enforce
5. **Phased migration**: backward compatibility with scratchpad

---

### Acceptance Criteria

- [x] RuntimeConfig with to_limits_dict() method
- [x] Runtime accepts config parameter
- [x] Runtime.start() initializes _limits from config
- [x] Runtime.get_limit_status() returns structured status
- [x] Runtime.check_limits() returns LimitWarning list
- [x] Runtime.update_limits() persists changes
- [x] Adapters read from _limits with scratchpad fallback
- [x] Logic classes accept vars for limit overrides
- [x] Agent classes expose limit methods
- [x] All unit tests pass
- [x] Backward compatibility verified
