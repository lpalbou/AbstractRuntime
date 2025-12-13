## 003_wait_primitives (completed)

### Goal
Support long pauses with durable primitives:
- `wait_event(key)`
- `wait_until(when)`
- `ask_user(prompt)`

### What shipped
Built-in handlers in `src/abstractruntime/core/runtime.py`:
- `EffectType.WAIT_EVENT`
- `EffectType.WAIT_UNTIL`
- `EffectType.ASK_USER`

### Notes
- A scheduler worker is not part of the kernel; the host is responsible for driving `tick()`.
- `wait_until` auto-unblocks when `tick()` is called after the time threshold.

