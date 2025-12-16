## 017_limit_warnings_and_observability (planned)

**Status**: Planned
**Priority**: Medium
**Depends on**: 016_runtime_aware_parameters (completed)

---

## Goal

Wire the limit warning infrastructure (created in 016) to:
1. Surface warnings to agents for self-aware behavior
2. Emit observability events for UI/CLI integration
3. Implement basic token estimation

---

## Context / Problem

The `_limits` namespace and `check_limits()` method exist, but:
- **Warnings are not surfaced to agents**: Agents don't know "only 2 turns remaining"
- **No observability events**: UI/CLI must poll `get_limit_status()` manually
- **Token counting is placeholder**: `estimated_tokens_used` is always 0

---

## Proposed Design

### 1. Agent Guidance Injection

When approaching limits, inject guidance into the agent's context/prompt:

```python
# In react_runtime.py reason_node:
def reason_node(run: RunState, ctx) -> StepPlan:
    context, scratchpad, runtime_ns, temp, limits = ensure_react_vars(run)

    # Check for approaching limits
    warnings = []
    current = limits.get("current_iteration", 0)
    max_iter = limits.get("max_iterations", 25)
    warn_pct = limits.get("warn_iterations_pct", 80)

    if max_iter > 0 and (current / max_iter * 100) >= warn_pct:
        remaining = max_iter - current
        warnings.append(f"Only {remaining} turns remaining")

    # Inject into guidance
    if warnings:
        inbox = runtime_ns.get("inbox", [])
        inbox.append({
            "role": "system",
            "content": f"[LIMIT WARNING: {'; '.join(warnings)}. Work efficiently!]"
        })
        runtime_ns["inbox"] = inbox
```

### 2. Observability Events

Emit events via the `on_step` callback when warnings trigger:

```python
# New event types
on_step("limit_warning", {
    "limit_type": "iterations",
    "status": "warning",
    "current": 20,
    "maximum": 25,
    "remaining": 5,
    "pct": 80.0,
})

on_step("limit_exceeded", {
    "limit_type": "iterations",
    "current": 25,
    "maximum": 25,
})
```

### 3. Token Estimation

Track tokens from LLM response metadata:

```python
# In LLM effect handler (llm_client.py):
def handle_llm_call(run, effect, default_next_node):
    response = llm_client.generate(...)

    # Extract token usage
    usage = response.get("usage", {})
    tokens_used = usage.get("total_tokens", 0)

    # Update _limits
    if tokens_used > 0:
        limits = run.vars.get("_limits", {})
        limits["estimated_tokens_used"] = (
            limits.get("estimated_tokens_used", 0) + tokens_used
        )

    return EffectOutcome.completed(response)
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `abstractagent/adapters/react_runtime.py` | Inject warnings into inbox/guidance |
| `abstractagent/adapters/codeact_runtime.py` | Inject warnings into inbox/guidance |
| `abstractruntime/integrations/abstractcore/llm_client.py` | Track token usage |
| `abstractruntime/integrations/abstractcore/effect_handlers.py` | Update _limits after LLM calls |

---

## Acceptance Criteria

- [ ] Agents receive guidance when approaching iteration limits (e.g., "3 turns remaining")
- [ ] Agents receive guidance when approaching token limits
- [ ] `on_step` callback receives `limit_warning` and `limit_exceeded` events
- [ ] `estimated_tokens_used` is updated after each LLM call
- [ ] Token warnings trigger when approaching context window
- [ ] Existing tests continue to pass
- [ ] New tests for warning injection and observability

---

## Test Plan

1. **Warning injection test**: Start agent with max_iterations=5, verify guidance appears at iteration 4
2. **Observability test**: Register on_step callback, verify limit_warning events received
3. **Token tracking test**: Mock LLM response with usage data, verify estimated_tokens_used updated
4. **Threshold test**: Verify warnings only trigger at/above configured percentage

---

## Notes

- Warning injection should be opt-in or configurable (some users may not want it)
- Token estimation depends on LLM providers returning usage data
- Consider rate-limiting warning events to avoid spam
