# Evidence capture

AbstractRuntime can record **provenance-first evidence** for selected “external boundary” tools (web + process execution). Evidence is stored durably as:
- a small index entry in `RunState.vars["_runtime"]["memory_spans"]`
- an artifact-backed payload (so checkpoints stay JSON-safe and bounded)

Implementation pointers:
- recorder: `src/abstractruntime/evidence/recorder.py`
- capture hook: `Runtime._maybe_record_tool_evidence(...)` (`src/abstractruntime/core/runtime.py`)
- retrieval helpers: `Runtime.list_evidence(...)` / `Runtime.load_evidence(...)` (`src/abstractruntime/core/runtime.py`)

## When evidence is recorded

Evidence capture runs best-effort after a successful `EffectType.TOOL_CALLS` step:
- only for tool names in `DEFAULT_EVIDENCE_TOOL_NAMES` (`web_search`, `fetch_url`, `execute_command`)
- only when an `ArtifactStore` is configured on the runtime (`Runtime(..., artifact_store=...)`)

If evidence capture fails, the runtime records a warning under `vars["_runtime"]["evidence_warnings"]` and continues execution.

## How to inspect evidence

```python
evidence = rt.list_evidence(run_id)
for e in evidence:
    print(e.get("tool_name"), e.get("created_at"), e.get("evidence_id"))

payload = rt.load_evidence(evidence_id="...")  # loads from ArtifactStore
```

## Storage and privacy

- Evidence payloads can include fetched page text or command stdout/stderr; treat artifacts and ledgers as sensitive.
- Secrets should never be passed as tool arguments (arguments are ledger-recorded). Prefer env-var resolution in tool implementations.

## See also

- `provenance.md` — tamper-evident ledger chain
- `architecture.md` — where evidence fits (runtime-owned, artifact-backed)

