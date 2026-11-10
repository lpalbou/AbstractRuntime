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
  cognition.yaml           # the cognition profile: a model per capability, recall width, rebuild policy
  identity.json            # look and voice (digital-identity.md)
  .identity.lock           # one identity write at a time
  state_history.jsonl      # biography: state, phase_changed and identity_changed rows
  memory.sqlite3           # graph + journal (abstractmemory)
  home.sqlite3             # diary book + command inbox
  runtime_castor.sqlite3   # run store + ledger (THIS page)
  artifacts/               # verbatims and run artifacts
  workspace/               # the entity's own files
  .writer_lease            # writer lease (see below)
```

**Copying the directory moves the whole life** — including pending runs,
durable waits, and commitments. Nothing at rest references the door's
address (relocation-stable keys).

## One writer per directory: the lease (`storage/lease.py`)

The lease is a **generic one-writer-per-directory mechanism** (entity homes
are its first consumer; project workplaces are the designed second — the
2026-07-10 vocabulary sign-off re-homed it from `identity/` to `storage/`).
It arbitrates *processes*, never principals: visitors' write-rights are
denied structurally at the deposit gate regardless of the lease.

Four writer windows exist for a home: a visit host, the own-time loop's
day, the dream window, and maintenance passes. Exactly one may hold the
directory at a time:

```python
from abstractruntime.storage.lease import acquire_directory_lease, DirectoryLeaseHeld, read_directory_lease

with acquire_directory_lease(home_dir, holder="visit-host", session_id="visit-1"):
    ...  # the directory is yours for this window

read_directory_lease(home_dir)   # {"holder": ..., "pid": ..., "held": True/False}
```

- Mechanics: `flock(LOCK_EX|LOCK_NB)` on `<dir>/.writer_lease`. The flock
  is the truth; the file's JSON metadata (holder kind, pid, acquired_at,
  session/run id) is diagnostics for refusal messages and "who holds
  Castor?".
- Refusal raises `DirectoryLeaseHeld` **naming the incumbent** — loud,
  never a silent wait. The own-time loop treats a held home as a *yield*
  back to its gate; the home-direct chat CLI refuses to double-summon.
- A crashed holder releases with its process (kernel drops the flock with
  the fd). A **copied** directory carries stale lease bytes but no lock —
  `read_directory_lease` answers `held` by a non-destructive flock probe,
  never by trusting metadata.
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

## Live visit corrections

Visit hosts may call `Runtime.queue_visit_guidance(run_id, text=..., speaker=...,
display_text=...)` for the current stamped participant while a visit actor is working.
The canonical host validates ownership before calling this narrower door; generic raw
`Runtime.steer` remains refused for driver-owned visits. The durable sidecar records
visit and turn identity, attributed words and optional original display text.

Before the next actor reasoning boundary, the visit performs fresh channel-labeled memory
recall, merges and deduplicates its handles with those already visible, and appends the
participant's verbatim correction and the refreshed recall. It keeps the existing transcript
prefix. Empty queues add no model call or recall. A correction from an older turn is retained
with attribution and is not silently applied to a later turn. Acceptance and delivery receipts
describe transport; only the actor's later words can acknowledge a change of plan.

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

## Cognition profile (`identity/cognition_profile.py`)

The mind (`substrate.yaml`) is what the entity talks with. The cognition
profile, `<home>/cognition.yaml`, says which model each other part of its
thinking uses, how many memories a turn brings to mind, how readily it
revisits what it already understood, what the look-back at the end of a
tick of its own time sees, and when an idle entity goes to sleep:

```yaml
schema_version: 1
capabilities:
  summaries:     {inherit: mind}    # digests, titles, keywords, cards
  understanding: {provider: "endpoint:compat-theforge-nvidia", model: "qwen/qwen3.8-27b", thinking: low}
  extraction:    {inherit: mind}    # the memory extraction lane
recall:
  shelf_size: 50                    # memories retrieved per turn, in visits and own time (1..400)
rebuild:
  sensitivity: 0.5                  # 0..1, how easily a new memory or a surprise re-opens what is understood
  cap_per_turn: 3                   # 0..10 memory tasks a turn may run at most; 0 = only while resting
lookback:
  context: day                      # what the end-of-tick look-back sees: day or tick
sleep:
  after_hours: 24                   # an idle entity sleeps this long after its last completed sleep (1..720)
  after_events: 500                 # ... or once this many memories formed since it (>= 1)
  after_fragmented_share: 0.15      # ... or once more than this share of its memories is fragmented (0 < x <= 1)
```

| Field | Absent means | Range |
|---|---|---|
| `capabilities.<cap>` | `{inherit: mind}` | `{inherit: mind}` or `{provider, model, thinking?, max_input_tokens?}`; `thinking` from `none, minimal, low, medium, high, xhigh, auto, on, off`; `max_input_tokens` 4096..140000 (140000 when absent) |
| `recall.shelf_size` | the host's default (the gateway's is 50) | whole number 1..400 |
| `rebuild.sensitivity` | 0.5 | number 0..1 |
| `rebuild.cap_per_turn` | 3 | whole number 0..10 |
| `sleep.after_hours` | 24 | whole number 1..720 |
| `sleep.after_events` | 500 | whole number >= 1 |
| `sleep.after_fragmented_share` | 0.15 | number above 0, at most 1: the share of the home's live memories that are fragmented (placed in a [topic](entity-topics.md) only provisionally, without a vector, or with a title or summary pending) |
| `lookback.context` | `day` | `day`: the whole day so far (a prefix cache keeps the new tokens per look-back about constant, roughly 770); `tick`: this tick's exchange plus a recap of the day's 12 latest earlier moments |

```python
from abstractruntime.identity.cognition_profile import (
    read_cognition_profile, resolve_capability, recall_shelf_size, rebuild_policy, lookback_context,
    sleep_thresholds,
)

resolve_capability(home_dir, "summaries")
# {"provider": ..., "model": ..., "thinking": ..., "max_input_tokens": 140000, "source": "mind"}
resolve_capability(home_dir, "summaries", mind={"provider": "lmstudio", "model": "qwen3-4b"})
recall_shelf_size(home_dir, 50)      # the profile's width, else the default you pass
rebuild_policy(home_dir)             # {"sensitivity": 0.5, "cap_per_turn": 3}
lookback_context(home_dir)           # "day" (default) or "tick"
sleep_thresholds(home_dir)           # {"after_hours": 24, "after_events": 500, "after_fragmented_share": 0.15}
read_cognition_profile(home_dir)     # every field, each with its source: "profile" or "default"
```

- A capability that inherits resolves through `resolve_home_substrate`, so it
  is exactly the mind; `mind=` passes a host's already-resolved mind as the
  explicit rung (the gateway passes the mind its visits use). `source` is
  `profile` or `mind`. When neither the profile nor the mind names a model,
  `resolve_capability` raises `SubstrateUnset`: no model is ever chosen by
  code.
- Reads never raise. A file that is not YAML, not a mapping or of another
  `schema_version` reads as all defaults; a single malformed field reads as
  its default while the others keep their values. Each case logs a warning
  that names the field, and `read_cognition_profile(...)["warnings"]` lists
  them.
- `write_cognition_profile(home_dir, profile)` validates first and raises
  `CognitionProfileInvalid` (`.field`, and a sentence such as "Memories per
  turn must be a whole number from 1 to 400.") before the file moves, then
  writes atomically (temporary file in the home, fsync, rename). It stores
  only what was chosen: a capability that inherits and a field that follows
  its default are left out. The gateway is its one caller
  (`PUT /api/gateway/entities/{name}/cognition-profile`), and it records a
  `cognition_profile_changed` marker in the entity's history first.
- The `sleep` block holds the thresholds after which an idle entity goes to sleep; the gateway's idle sweep reads
  them through `sleep_thresholds`. A sleep runs the entity's maintenance, the [topics](entity-topics.md) stage
  included.
- The own-time loop reads `recall.shelf_size` again at each day-open
  (`recall_shelf_size(home, <the loop's --shelf-size>)`), so a change takes
  effect at the next own-time day without restarting the loop; the loop's
  log names the change.

## Tests

`tests/test_directory_lease.py`, `tests/test_entity_runtime.py`,
`tests/test_act_only_dereference.py`, `tests/test_wait_event_deadline.py`,
`tests/test_cognition_profile.py` —
including cross-process lease exclusion, waits-traveling-on-home-copy, the
ledger-keeps-the-ref end-to-end pin, and the three-backend due-scan.
