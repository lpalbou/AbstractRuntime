# Planned: Durable-write discipline (kill the per-step serialization amplification)

## Metadata
- Created: 2026-07-13
- Status: Planned
- Completed: N/A
- Priority: P1 (the 24/7 disk-I/O + latency wall; portable by construction)
- Area: storage/json_files, storage/sqlite, core/runtime, core/models, storage/offloading
- Source: 2026-07-13 performance adversary (fable5) findings 1 + 3

## ADR status
- Governing ADRs: None
- ADR impact: None. NOTE — this item ENFORCES an invariant the track already
  ruled on: the track's non-goal "no deep-copy store views (hot-path tax)"
  is currently violated INSIDE `save()` via `asdict`. This is that non-goal
  made real, not a new policy.

## Context
The dominant per-step cost is not one operation but one pattern: the same
conversation-sized bytes are serialized and persisted 5–6× per effectful
step, and two copies land in stores that are later re-read in full. At 24/7
resident scale the first wall is disk I/O + tick latency from ledger growth
and save amplification.

## Current code reality (line-verified 2026-07-13; benched on a 120-message
/ ~185 KB payload, ~2 MB RunState)
- `JsonFileRunStore.save` (json_files.py:141) does `json.dump(asdict(run),
  ..., indent=2)` (:148). `asdict(run)` is a FULL DEEP COPY of the vars tree
  (~1.02 ms + allocation churn); `indent=2` costs +72% serialize CPU (7.16
  vs 4.16 ms) and +5% bytes vs compact. Measured ~10.5 ms/save at 2 MB.
  `SqliteRunStore.save` (sqlite.py:297) has the same `asdict(run)` deep copy
  (~6.3 ms/save). The single-writer contract is already documented
  (json_files.py:36-52); `json.dump` does not mutate its input.
- Saves per ReAct cycle ≈ 5 (17 `self._run_store.save(run)` sites in the
  tick loop) → ~50 ms + ~10 MB written per cycle for one resident; at the
  node-trace cap (100 entries/node) state converges to ~15–25 MB → ~75
  ms/save and tens of GB/day written for one resident.
- Ledger triplication per LLM effect: the STARTED record embeds the full
  effect payload (core/models.py `StepRecord.start`, appended
  ~core/runtime.py:2550); the COMPLETED record is the SAME record re-appended
  with the payload still attached (~2566); the completed result carries the
  conversation a THIRD time via `metadata._provider_request.payload.messages`
  (llm_client.py:6060-6095 attaches it when the provider didn't; HTTP
  transports attach it always, ~10546/11773/11813). Tool waits add a fourth
  via the synthetic completion record (~2207-2218). Measured ~540 KB of
  ledger per LLM effect; 60 effects = 32.4 MB. Because per-turn payloads grow
  with the conversation, ledger bytes grow O(turns²).
- `OffloadingLedgerStore` EXISTS (storage/offloading.py:377) but is NOT wired
  in the runtime's own factories — `create_local_runtime` uses raw
  `JsonlLedgerStore` (integrations/abstractcore/factory.py:38-41) and
  `open_entity_runtime` uses raw `SqliteLedgerStore`
  (identity/entity_runtime.py:137-140).
- `JsonlLedgerStore.append` (json_files.py:453) and SqliteLedgerStore
  (sqlite.py:583/643) also `asdict(record)` per append.

## Problem
The most frequent durable operations in the system (save per transition,
append per effect) each pay a full deep copy + fat serialization, and the
ledger's growth constant is O(turns²) in bytes — the multiplier under the
0047 count-scan cliff.

## What we want to do
Cheapest first, all zero-semantic-change under the single-writer contract:
1. Drop `indent=2` in `JsonFileRunStore.save` (~40% of serialize CPU back).
2. Replace `asdict(run)` / `asdict(record)` with a hand-built top-level dict
   that passes `run.vars` / record fields BY REFERENCE (json.dump doesn't
   mutate; the single-writer contract guarantees no concurrent mutation).
   Apply in both run stores AND both ledger appends.
3. Slim the COMPLETED ledger record to a digest + `same_as_step_id`
   reference to the STARTED record's bytes (STARTED already holds them).
4. Bound or strip `_provider_request.payload.messages` from the durable
   result — it is byte-identical to the effect payload one record above.
5. Wire the existing `OffloadingLedgerStore` into `create_local_runtime` and
   `open_entity_runtime`.

## Why
Roughly halves the cost of the single most frequent durable op (measured
10.5 → ~5 ms/save at 2 MB), and removes the O(turns²) ledger growth constant
that makes the 0047 cliff quadratic. Portable by construction (pure
serialization discipline).

## Requirements
- Byte-content of durable state unchanged EXCEPT the deliberate slimming
  (COMPLETED payload ref, provider_request trim) — pin that replay + history
  reconstruction still resolve the STARTED bytes for a slimmed COMPLETED
  record.
- No aliasing hazard: pass-by-reference is safe ONLY under the documented
  single-writer contract; re-assert that contract at each changed site.
- Offloading wiring must not change the ledger's public read shape
  (consumers see rehydrated records).

## Suggested implementation
Steps 1–2 are two small diffs (S). Steps 3–4 are M (touch the record shape +
history_bundle / replay resolvers — verify every reader of a COMPLETED
record's payload). Step 5 is S (factory wiring) but must be tested against
the offload/rehydrate round-trip.

## Scope
Both run stores, both ledger appends, the COMPLETED-record shape, the
provider_request trim, and the two factory wirings + tests.

## Non-goals
- No coalescing of pure-transition saves (that needs 0045's crash-ordering
  invariant first — separate item).
- No ledger pruning/compaction (append-only; archival is 0058).
- No change to the idempotency key (0047 owns the lookup; this owns the
  bytes).

## Dependencies and related tasks
- 0047 (indexed idempotency): MUST ship together — the index kills
  count-scaling, this kills byte-scaling; either alone leaves half the cliff
  (a bounded tail window still parses fat records). Cross-referenced in 0047.
- 0048 (durable-IO unification): the write module this shares.
- 0053 (bounded run-vars): node-trace byte-cap shrinks the state ~10×,
  compounding step 1–2's win.

## Expected outcomes
Per-save time roughly halved at 2 MB; ledger growth linear (not O(turns²));
offloading live in the default + entity runtimes; full suite green with
replay/history reconstruction verified over slimmed records.

## Validation
- A/B bench: save time + bytes-written per cycle before/after, JSON + SQLite.
- Ledger-growth bench: bytes per LLM effect before/after slimming (target:
  no per-turn conversation duplication).
- Replay correctness: a run with slimmed COMPLETED records replays
  byte-identically; history_bundle resolves the STARTED bytes.
- Offload round-trip: a wired factory's ledger reads rehydrated records.

## Progress checklist
- [x] Drop indent=2 (JSON save) — shipped 2026-07-13 (0067-S)
- [x] asdict → by-reference dict (2 run stores + 2 ledger appends) — shipped 2026-07-13 (0067-S)
- [x] COMPLETED record → digest + same_as_step_id ref — shipped 2026-07-14 as
  `storage/ledger_slim.py`: per-field `$slim` markers (>4KB gate, sha256
  verified) on ALL terminal appends (completed/waiting/failed), resolved by
  history_bundle readers + rehydrated on crash-replay reuse.
- [x] Strip/bound provider_request.messages in durable result — shipped as
  VERIFIED DEDUP, not a strip: `_provider_request.payload.messages` and
  `_runtime_observability.llm_generate_kwargs.*` marker ONLY when
  byte-identical to the STARTED payload (or its documented local
  system+messages+prompt reconstruction). Decorated wire bytes never match
  and stay verbatim — the B3 lane (`/llm --verbatim`) is preserved by
  construction, deliberately diverging from the item's "strip" option.
- [x] Wire OffloadingLedgerStore in factory + entity runtime — shipped with
  the read-side rehydration the Requirements demanded (tag-checked, own
  refs only, recursive).
- [x] Benches recorded (120-msg/185KB shape: 735,723 → 186,554 bytes per
  LLM effect, 74.6% saved, terminal record 551,971 → 2,802); replay +
  history verified (`tests/test_ledger_record_slimming.py`; full suite
  green).
- [x] fable5 adversary FOLDED (2026-07-14, four fixes): P0 factory
  composition (`OffloadingLedgerStore` satisfied the observable Protocol
  by method presence → `subscribe_ledger` raised on every durable
  deployment; one explicit `_compose_durable_ledger` site now, behavior-
  pinned not type-pinned); P1 STARTED-time digest anchor (in-place payload
  mutation between appends kept verbatim, never an unresolvable marker);
  P1 `list()` reverted to refs-on-read (rehydrating reads measured 113x
  time/593x bytes + false chain-tamper alarms; rehydration only on the
  crash-replay probe); P2 targeted replay rehydration (echoed
  marker-shaped DATA survives byte-identically). Suite 1264 green.
  Surviving invariants the adversary could not break: crash-replay
  byte-identity through the full composed path, retries, synthetic tool
  completion, entity wiring, G1 privacy, vars/node_traces isolation.

## Guidance for the implementing agent
Steps 1–2 are safe and independently shippable — do them first for the quick
win. Steps 3–4 change the durable record shape: enumerate every reader of a
COMPLETED record's payload (replay, history_bundle, ledger consumers) and
pin resolution BEFORE slimming. The single-writer contract is the licence
for pass-by-reference; if any store ever gains a concurrent writer, this
reverts.
