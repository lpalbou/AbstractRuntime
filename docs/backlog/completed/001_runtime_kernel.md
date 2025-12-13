## 001_runtime_kernel (completed)

### Goal
Implement the minimal AbstractRuntime kernel that can execute a workflow graph until:
- completion
- failure
- or an interrupt (waiting state)

### What shipped
- `src/abstractruntime/core/models.py`
  - `RunState`, `WaitState`, `Effect`, `EffectType`, `StepPlan`, `StepRecord`
- `src/abstractruntime/core/spec.py`
  - `WorkflowSpec`, `NodeHandler` contract
- `src/abstractruntime/core/runtime.py`
  - `Runtime.start()`, `Runtime.tick()`, `Runtime.resume()`

### Key semantics
- `tick()` progresses until WAITING/COMPLETED/FAILED
- WAITING is explicit (`RunState.waiting`); no Python stacks are preserved

### Notes
- Effect handlers receive `default_next_node` to ensure waiting effects resume safely.

