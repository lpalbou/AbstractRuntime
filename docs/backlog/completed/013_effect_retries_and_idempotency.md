## 013_effect_retries_and_idempotency (completed)

**Status**: Completed  
**Completed**: 2025-12-13

---

## Final Report

### What Was Implemented

| Component | Description |
|-----------|-------------|
| `EffectPolicy` | Protocol for retry and idempotency behavior |
| `DefaultEffectPolicy` | Configurable policy with exponential backoff |
| `RetryPolicy` | Pre-configured policy for LLM/tool retries |
| `NoRetryPolicy` | Policy that never retries |
| `compute_idempotency_key()` | Standalone function for key computation |
| `StepRecord.attempt` | Tracks attempt number (1-indexed) |
| `StepRecord.idempotency_key` | For deduplication on restart |

### Key Features

1. **Configurable Retry**: Per-effect-type max attempts
   ```python
   policy = RetryPolicy(llm_max_attempts=3, tool_max_attempts=2)
   ```

2. **Exponential Backoff**: Configurable base and max delay
   ```python
   policy = DefaultEffectPolicy(
       default_backoff_base=1.0,  # 1s, 2s, 4s, 8s...
       default_backoff_max=60.0,  # Cap at 60s
   )
   ```

3. **Idempotency on Restart**: Prior completed results are reused
   - Key = hash(run_id + node_id + effect_type + payload)
   - Ledger is scanned for matching completed records
   - If found, result is reused without re-execution

4. **Ledger Tracking**: Each attempt is recorded with:
   - `attempt`: 1, 2, 3...
   - `idempotency_key`: For deduplication lookup

### Files Added/Modified

**New:**
- `src/abstractruntime/core/policy.py` - Policy implementations
- `tests/test_retry_idempotency.py` - 18 comprehensive tests

**Modified:**
- `src/abstractruntime/core/models.py` - StepRecord fields
- `src/abstractruntime/core/runtime.py` - Retry loop and idempotency check
- `src/abstractruntime/__init__.py` - Exports
- `src/abstractruntime/scheduler/convenience.py` - effect_policy parameter

### Exception Handling

Effect handlers that raise exceptions are caught and treated as failures:
- Exception is converted to `EffectOutcome.failed()`
- Retry logic applies normally
- Prevents unhandled exceptions from crashing the runtime

### Test Coverage (20 tests)

- Retry logic: no retry by default, retry with policy, exhausted attempts
- Backoff: exponential, capped at max, called between retries
- Idempotency: key computation, differs by run/payload, skip re-execution
- Ledger tracking: records attempt number and idempotency key
- Policy configuration: NoRetryPolicy, custom policy, ScheduledRuntime
- Edge cases: waiting effects not retried, success on first attempt
- Exception handling: exceptions in handlers trigger retry
- File persistence: retry fields persisted in file-based ledger

### Usage Example

```python
from abstractruntime import (
    create_scheduled_runtime,
    RetryPolicy,
)

# Create runtime with retry policy
sr = create_scheduled_runtime(
    effect_policy=RetryPolicy(
        llm_max_attempts=3,
        tool_max_attempts=2,
        backoff_base=1.0,
        backoff_max=30.0,
    ),
)

# Effects will now retry on failure
run_id, state = sr.run(my_workflow)
```

### Crash Recovery Scenario

```
1. Workflow starts, reaches LLM_CALL effect
2. LLM call succeeds, returns result
3. Process crashes before checkpoint saved
4. On restart, runtime sees run at same node
5. Computes idempotency_key for the effect
6. Finds prior COMPLETED record in ledger with same key
7. Reuses prior result instead of re-calling LLM
8. Workflow continues without duplicate side effects
```

---

## Original Proposal

### Goal
Add a policy layer for:
- retry/backoff of effects (LLM calls, tools)
- idempotency keys / deduplication to avoid double-applying side effects

### Context / problem
A durable runtime faces an inherent risk:
- an effect executes successfully
- the process crashes before the checkpoint/ledger update is committed
- on restart, the same node can re-execute the effect

For LLM/tool calls this can cause:
- duplicated external actions
- inconsistent outputs
- difficulty debugging and auditing

The minimal kernel can tolerate at-least-once execution, but for agentic systems we eventually need **controlled retries** and **idempotency strategy**.

### Non-goals
- No global exactly-once guarantees across a cluster.
- No Temporal-style determinism/replay engine.

---

### Proposed design

#### A) Policy interface
Add a small policy abstraction (injected into `Runtime` via context or constructor):

```python
class EffectPolicy(Protocol):
    def max_attempts(self, effect: Effect) -> int: ...
    def backoff_seconds(self, *, effect: Effect, attempt: int) -> float: ...
    def idempotency_key(self, *, run: RunState, node_id: str, effect: Effect) -> str: ...
```

#### B) Ledger attempt records
Extend `StepRecord` with:
- `attempt: int`
- `idempotency_key: str`

Then the runtime can:
- look up the most recent record for the same idempotency_key
- avoid re-executing if there is a completed result already recorded

#### C) Storage needs
This requires at least one of:
- `LedgerStore.get_last_by_key(run_id, idempotency_key)`
- or scanning the ledger list (MVP)

#### D) Semantics
- For each effect step, runtime computes idempotency_key.
- If prior completed record exists → reuse prior result instead of re-executing.
- Otherwise execute and append record.

---

### Files to add / modify
- `src/abstractruntime/core/policy.py` (EffectPolicy)
- Extend `StepRecord`
- Extend `LedgerStore` OR add optional query protocol
- Update runtime loop to apply policy
- Add tests:
  - retry increments attempt
  - dedupe reuses prior result

---

### Acceptance criteria
- Retrying does not create unbounded duplicate effects.
- Idempotency strategy can be plugged/overridden by host.
- Works without external services (file + in-memory backends).

### Test plan
- Unit tests on in-memory ledger with forced failures.
- File-based test simulating crash/restart (persist ledger then re-run).
