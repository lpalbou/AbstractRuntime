# 0866 — The model queue is visible in the ledger and the replay view; gateway-side generations borrow a pool, never build a client (Core ADR 0011)

**Status**: planned · **Priority**: P0 (this wave, the runtime half) · **Created**: 2026-10-29 (architect of the shared-weights P0)
**Package**: abstractruntime (`integrations/abstractcore/llm_client.py`, the config facade, `replay/view.py`, ledger event schema) · **Slices**: Core 0870 (the scheduler and the `queued` progress phase), gateway 0978 (the Models row, `/v1`, the one-copy test)
**Master item**: Core ADR 0011 (`docs/adr/0011-shared-weights-and-concurrent-requests.md`); Core ADR 0010; gateway ADR 0002; runtime ADR 0007 (relay Core's residency truth), 0008 (facades are the complete surface), 0010 (one replay, many views)

## Summary

Core ADR 0011 (2026-10-29) decides that every request path shares ONE weights object per model and that
concurrency on it is a per-model slot scheduler in the process that holds the weights, with a visible
queued state and per-request cancel. The runtime's part, in this wave:

1. **Lend, never build (R2, worker D).** A gateway-side generation (run summary, Ask on a run, the
   console sandbox, entity chat, backlog assist) that today builds a `LocalAbstractCoreLLMClient` of its
   own, loading its own copy under llama.cpp, borrows a request-scoped lease on the principal's pooled
   client (`MultiLocalAbstractCoreLLMClient`) instead. The lease carries the per-call settings (endpoint
   profile resolver, capability defaults, artifact store) without rebuilding the provider; its claim is
   registered under the caller's readable label ("a summary for the runtime of account default/admin",
   framework ADR 0043) and released with the call. The runtime adds NO lock and NO worker to make this
   safe: the scheduler in Core owns concurrency (ADR 0011 §2); the runtime passes `cancel_event` and the
   progress callback through, as it does today.
2. **The `queued` phase in the ledger.** Core's progress events gain `phase: "queued"` with
   `queue_position`, `queue_ahead`, `slots`, `slots_max`, `slot_note` (ADR 0011 §2.4). The runtime
   records them as it records `prefill` / `generate` (the same `llm.progress` ledger event, the same
   0.5 s cadence, never truncated), and the replay view (ADR 0010) shows "Queued behind N requests on
   MODEL" in place of "Thinking…" (the model's readable name, never an id). The typed errors
   `ModelQueueFullError` / `ModelQueueTimeoutError` are recorded on the failed LLM_CALL with
   `error.code = "model_queue_full"` / `"model_queue_timeout"`, request-local: never retried, never
   counted against endpoint health (the native lane's `queue_full` / `queue_timeout` today).
3. **The concurrency facade.** `config_facade` gains `configured_model_concurrency` /
   `set_model_concurrency` (the siblings of `configured_model_context` / `set_model_context`, Core
   `config.model_concurrency`), with `AbstractCoreTooOld` in words on a Core without it, so the gateway's
   Models row can offer "Set concurrency" through the facade (ADR 0008).
4. **Residency relay.** The pool fields on Core's residency rows and claims (`slots`, `slots_max`,
   `slot_bytes`, `active`, `queued`, `slot_note`) pass through the residency relay unchanged (ADR 0007),
   so the gateway shows them.

## Tests (red without the change)

- `tests/test_gateway_side_generation_borrows_pool.py`: with a fake Core provider that counts
  constructions, a summary, an Ask and a sandbox call on a runtime whose pool holds the model build
  ZERO new providers; the claims list shows the lease under the readable label during the call and not
  after; the cancel event passed to the lease reaches the provider.
- `tests/test_queued_progress_ledger.py`: a provider emitting `queued` events → `llm.progress` ledger
  records with the fields; the replay view's status is "Queued behind 2 requests on MODEL"; after the
  `generate` phase the status is the usual one; `model_queue_full` / `model_queue_timeout` on the
  LLM_CALL error, not retried.
- `tests/test_config_facade_model_concurrency.py`: set / read / refuse-in-words; the too-old Core
  sentence.
- Conformance fixture added to the replay view set (ADR 0010: one fixture, every client).

## Out of scope (later)

Per-principal weighted fairness at the queue (ADR 0011 §2.3 leaves it to the 1064 benchmark);
continuous batching (Core 0871).
