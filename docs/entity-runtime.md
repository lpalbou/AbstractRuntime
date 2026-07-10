# Per-entity runtime (homes, leases, visit durability)

How AbstractRuntime hosts a **summoned entity**: one self-contained home
directory per entity, one `Runtime` per home, one writer at a time, and
visit turns that survive process restarts. This page documents the runtime
half of the entity-topology consensus plan ("1 gateway + N runtimes");
the door half (auth, stamps, serving) is AbstractGateway's.

## The home is the unit

An entity home is one directory holding the whole life:

```
entities/castor/
  manifest.json            # entity_id (birth marker — never a lookup key)
  spark.yaml               # the attested seed
  substrate.yaml           # operator's ONE mind choice (provider+model)
  memory.sqlite3           # graph + journal (abstractmemory)
  home.sqlite3             # diary book + command inbox
  runtime_castor.sqlite3   # run store + ledger (THIS page)
  artifacts/               # verbatims and run artifacts
  workspace/               # the entity's own files
  .home_lease              # writer lease (see below)
```

**Copying the directory moves the whole life** — including pending runs,
durable waits, and commitments. Nothing at rest references the door's
address (relocation-stable keys).

## One writer per home: the lease (`identity/lease.py`)

Four writer windows exist for a home: a visit host, the own-time loop's
day, the dream window, and maintenance passes. Exactly one may hold the
home at a time:

```python
from abstractruntime.identity.lease import acquire_home_lease, HomeLeaseHeld, read_home_lease

with acquire_home_lease(home_dir, holder="visit-host", session_id="visit-1"):
    ...  # the home is yours for this window

read_home_lease(home_dir)   # {"holder": ..., "pid": ..., "held": True/False}
```

- Mechanics: `flock(LOCK_EX|LOCK_NB)` on `<home>/.home_lease`. The flock is
  the truth; the file's JSON metadata (holder kind, pid, acquired_at,
  session/run id) is diagnostics for refusal messages and "who holds
  Castor?".
- Refusal raises `HomeLeaseHeld` **naming the incumbent** — loud, never a
  silent wait. The own-time loop treats a held home as a *yield* back to
  its gate; the home-direct chat CLI refuses to double-summon.
- A crashed holder releases with its process (kernel drops the flock with
  the fd). A **copied** home carries stale lease bytes but no lock —
  `read_home_lease` answers `held` by a non-destructive flock probe, never
  by trusting metadata.
- Release truncates the file to a released record; it never unlinks
  (unlink races a concurrent acquirer onto a dead inode).
- The one-lease relay rule: cross-home delivery acquires ONE home's lease
  at a time, never two.

## One Runtime per entity (`identity/entity_runtime.py`)

```python
from abstractruntime.identity.entity_runtime import open_entity_runtime

ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: my_llm_handler})
run_id = ert.runtime.start(workflow=visit_workflow, vars={}, session_id="visit-1")
ert.runtime.tick(workflow=visit_workflow, run_id=run_id)
ert.close()   # checkpoints WAL so the directory is copy-clean
```

- The run store + ledger live in `runtime_<slug>.sqlite3` **inside the
  home** (slug = the directory name, the registry key).
- Effect handlers are the home's own seam + diary handlers (strict entity
  posture): `MEMORY_*` writes land in the home's graph, `DIARY_*` in the
  book, artifacts in the home's store. Handlers are RAW here — the gateway
  wraps door-served instances with stamp verification.
- `extra_handlers` lets the host add `LLM_CALL`/`TOOL_CALLS` at
  composition. Attempting to shadow a home handler **raises**: identity
  effects route through the home, never a host override.
- Every host `LLM_CALL` handler is automatically wrapped with the
  **act-only dereference** (below) — the privacy boundary cannot be
  forgotten.

## Act-only content: references at rest, words only in flight (`identity/act_only.py`)

Diary words must never rest outside the book. When an entity reads its own
diary mid-visit, the **durable transcript carries a typed reference**, not
the words:

```json
{"$act_only": {"tool": "diary_read", "entry_id": "diary_ab12", "gist": "one bounded line"}}
```

- The ref is the tool message's entire `content` (exact JSON, one top-level
  key). Detection is parse-based, never regex — and only on
  `role == "tool"` messages, which hosts append: a visitor pasting
  ref-looking JSON into their message stays inert text.
- At send time the wrapped `LLM_CALL` handler resolves refs **through the
  run's own `DIARY_READ` handler** into a wire copy (in-place content
  substitution; message identity untouched). The original payload — what
  the ledger and run store hold — keeps the ref.
- An unresolvable ref fails the effect **loud and non-retryable**; the
  provider never receives a degraded payload.
- Consequence for audits: like media `{"$artifact": ...}` refs, the wire
  payload is not byte-reconstructible from the ledger alone —
  reconstruction re-resolves refs through doors that enforce authority.

## Durable visit waits: event + deadline

A visit parks on the visitor's next message; the park may carry an idle
deadline (no reaper daemons):

```python
Effect(type=EffectType.WAIT_EVENT, payload={
    "wait_key": "visitor_input",
    "until": "2026-07-11T09:00:00+00:00",          # optional deadline (UTC-normalized)
    "details": {"kind": "visitor_message"},         # self-describing for clients
}, result_key="_temp.resume")
```

- An event resume before the deadline wins (payload lands in `result_key`).
- Past the deadline, `tick()` (or the scheduler's due-scan —
  deadline-carrying EVENT waits join `list_due_wait_until` on every store
  backend) resolves the wait with `{"timed_out": true}` so the workflow
  routes to its close path.
- The ledger wait record carries `wait_key`, `until`, and `details`
  together, so clients can render a chat composer for visit waits and the
  deadline without new transport.

## Tests

`tests/test_home_lease.py`, `tests/test_entity_runtime.py`,
`tests/test_act_only_dereference.py`, `tests/test_wait_event_deadline.py` —
including cross-process lease exclusion, waits-traveling-on-home-copy, the
ledger-keeps-the-ref end-to-end pin, and the three-backend due-scan.
