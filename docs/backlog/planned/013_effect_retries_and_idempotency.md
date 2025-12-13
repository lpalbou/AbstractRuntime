## 013_effect_retries_and_idempotency (planned)

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
