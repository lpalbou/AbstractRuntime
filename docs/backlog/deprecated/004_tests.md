## Todo 4 — Tests (no live external services)

### Goal
Add unit tests for:
- AbstractCore integration wiring (LLM_CALL + TOOL_CALLS handlers)
- Remote LLM client request construction (no real HTTP)
- Snapshot store roundtrip + search
- Hash-chain ledger verification

Constraints:
- Tests must not require a running Ollama/LMStudio/OpenAI/etc.
- Tests must not hardcode “special cases for tests” into production code.

The correct general-purpose design approach is **dependency injection seams** (request sender, stub clients) — that’s not test-only logic; it’s the right architecture for portability.

---

### 1) Integration tests (LLM_CALL / TOOL_CALLS)

#### Strategy
- Use the real `Runtime` + `WorkflowSpec`.
- Use **in-memory stores**.
- Use a **deterministic stub** LLM client (returns stable JSON).
- Use a local tool executor (or a stub tool executor).

This verifies:
- result-key wiring into `run.vars`
- ledger append behavior
- WAITING semantics for passthrough tools

#### Minimal workflow for test

- Node `n1`: emits `EffectType.LLM_CALL`, result_key=`"llm"`, next=`n2`
- Node `n2`: emits `EffectType.TOOL_CALLS`, result_key=`"tools"`, next=`n3`
- Node `n3`: returns `complete_output={...}`

#### Expected assertions
- `run.vars["llm"]["content"] == "..."`
- tool results are present (executed mode) OR run becomes WAITING (passthrough mode)
- `runtime.get_ledger(run_id)` contains step records with effect payloads

#### Optional: abstractcore dependency
If these tests use the real AbstractCore tool registry, guard with:

```python
import pytest
abstractcore = pytest.importorskip("abstractcore")
```

This keeps `abstractruntime` tests runnable without AbstractCore installed.

---

### 2) Remote client request construction (no live HTTP)

#### Strategy
The remote client must accept a dependency-injected request sender.

Test uses a stub:
- captures `url`, `headers`, `json`
- returns a deterministic response body shaped like AbstractCore server output

Assertions:
- URL is exactly `server_base_url.rstrip('/') + '/v1/chat/completions'`
- JSON includes:
  - `model`
  - `messages`
  - `temperature`, `max_tokens` when provided
  - `base_url` when you intend to route to openai-compatible endpoints (if used)

This is a **general-purpose seam** that also enables:
- running remote calls in environments without `httpx`
- swapping HTTP backends later

---

### 3) Snapshot store tests

#### Roundtrip
- Create a fake RunState dict (or use `RunState.new()` then `asdict`).
- `save(snapshot)` then `load(snapshot_id)`.
- Assert structural equality of:
  - name/description/tags
  - run_id
  - run_state dict

#### Search
- Save several snapshots.
- Search by:
  - `tag="prod"` returns correct subset
  - `query="invoice"` finds substring matches in name/description

Avoid relying on filesystem ordering.

---

### 4) Hash-chain verification tests

#### Happy path
- Create an underlying `InMemoryLedgerStore`.
- Wrap with `HashChainedLedgerStore`.
- Append 3+ records.
- `verify_ledger_chain(records)` returns `ok=true`.

#### Tamper detection
- Mutate one record’s `result` or `effect`.
- Re-run verification → `ok=false`, `first_bad_index` identifies the corrupted record.

#### Edge cases
- Empty ledger: ok=true
- Single record: ok=true
- Missing hashes: ok=false (or ok=false with clear errors)

---

### 5) Recommended test file structure (in `abstractruntime` repo)

- `tests/test_integrations_abstractcore.py`
- `tests/test_snapshots.py`
- `tests/test_provenance_hash_chain.py`

Keep each test file focused.

---

### Deliverable checklist

- [ ] Tests run without any live external services
- [ ] Remote HTTP tests use injected request sender
- [ ] Snapshot store has roundtrip + search tests
- [ ] Provenance verification has happy-path + tamper detection tests
- [ ] Existing pause/resume tests remain unchanged


