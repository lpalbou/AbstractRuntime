# Track record and reputation

What an entity (or a regular agent's session) has done, and how much its past outcomes say we can trust it
for each kind of task. Both are **computed from records the runtime and the entity home already keep**:
nothing new is stored, no run records anything extra, and the same records always give the same answer.
Module: `abstractruntime.identity.track_record` (ADR 0052 §5-§6, framework backlog 1119).

> Today "success" means **the work completed** (or the entity said it was done). Nobody has judged whether
> it was good. Every answer says so in its `basis` field: `"completion-rate only"`. Feedback from people
> and other agents has a place in the model (below) but no way in yet: that is backlog 1120.

## Quick start

```python
from abstractruntime.identity import compute_reputation, entity_track_record, session_track_record
from abstractruntime.identity.entity_runtime import open_entity_runtime

er = open_entity_runtime(home_dir)          # or the stores your host already holds
record = entity_track_record(home_dir, er.run_store, er.ledger_store, book_head=er.home.diary.head())
print(record.runs, record.successes, record.failures, record.tokens.total)

reputation = compute_reputation(record)
print(reputation.overall.describe())        # "trust 0.83 over 12 outcomes" / "no experience yet"
for kind, cell in reputation.by_task_kind.items():
    print(kind, cell.describe())

# A regular agent: the same rules over one session of the host's store
session_record = session_track_record("sess_9f2a", run_store, ledger_store)
```

`since="2026-11-01T00:00:00+00:00"` keeps only what started at or after that time.

## What counts

A track record counts **turn roots** — the runs a session shows as its turns — that came from outside:

| Counted | Never counted |
|---|---|
| a run a person started (`_meta.origin.kind == "person"`) | sub-runs (their tokens count with the run above them) |
| a run an automation's trigger started (`"trigger"`) | automation controllers |
| every automation occurrence, **once**, however many attempts it took | the session memory run (`__session_memory__`) and any reserved `__…__` workflow |
| | runs another run or the system started (origin `agent` or `system`), and runs with no recorded origin |

This is what stops an entity from raising its own trust by starting easy runs: only work someone or
something outside asked for counts.

> An entity visit run carries an origin only when the gateway records one at visit open. Until it does,
> visits do not appear in an entity's record — that is reported, never guessed.

For an entity, two files of its home add outcomes that are not runs:

- `task_inbox.jsonl` — a task whose latest status is `done`;
- `work_order.done.md` — each verdict the entity gave when it finished a work order (`done:` or
  `blocked:`), as the work lane archives it.

## Outcomes and their signals

Each counted thing gives at most one outcome, and the evidence names where it was read:

| Signal | Read from | Outcome |
|---|---|---|
| `run_completed` / `run_failed` | the run's status | success / failure (cancelled, running and waiting are not outcomes) |
| `occurrence_completed` / `occurrence_failed` | the automation's `automation.completed` decision for that occurrence | success / failure (cancelled is not an outcome; no decision yet = in progress) |
| `task_done` | the task inbox | success |
| `work_order_done` / `work_order_blocked` | the work-order archive | success / failure |
| `feedback_user` / `feedback_agent` | **nothing yet** (backlog 1120) | a person's or an agent's verdict on the work, with who gave it (`by`) |

`assessed` counts feedback outcomes; it is 0 until feedback exists.

## Task kinds

Trust is per **task kind**: `<phase or role>/<workflow_id>`.

- the phase of the run's entity stamp (`visit`, `work`, `personal`; a stamp minted before phases existed
  is a visit), else
- the run's attribution role: `occurrence`, `discussion`, or `chat` when it has none,
- then the workflow the run executed.

Examples: `visit/entity-visit-react`, `occurrence/daily-digest`, `chat/basic-agent`. Tasks and work orders
are `work/task_inbox` and `work/work_order`. The kind is read from ids only — never from what anyone wrote.

## The track record

| Field | Meaning |
|---|---|
| `subject` | `entity:<slug>` or `session:<session_id>` |
| `identity_ref` | the subject's identity now ([digital identity](digital-identity.md)): `entity:<slug>#<rev>` of the home's `identity.json`, or for a session the ref its newest counted run carries; `None` when none is recorded (never invented) |
| `fingerprint` | an entity's `identity_fingerprint(manifest, book_head=...)` (`actor_id` from `entity_id`, `home_id`, `spark_hash`, served with the diary head); `None` for a session (its agent's fingerprint is abstractagent's) |
| `sessions`, `runs` | distinct sessions and counted runs |
| `successes`, `failures`, `cancelled`, `in_progress` | the outcomes above |
| `assessed` | feedback outcomes (0 today) |
| `tokens` | `{input, output, total}` of every `llm_call` of the counted runs and their sub-runs |
| `by_task_kind` | the same counts per kind, with `tokens_total`, `first_at`, `last_at`, `last_outcome_at` |
| `recent_outcomes` | the last 20 outcomes, oldest first |
| `as_of` | the newest time read (a host caches on it) |
| `basis` | `"completion-rate only"` |

Tokens use the SQLite ledger's `metrics_many` query (`tokens_input`, `tokens_output`, `tokens_total`); on
other ledgers the same rules run over the run's records (`token_counts_of_records`). Input is
`usage.input_tokens` (else `prompt_tokens`), output is `output_tokens` (else `completion_tokens`), total is
`total_tokens`.

## Reputation

For each task kind:

```
trust   = (successes + 1) / (successes + failures + 2)
support = successes + failures
```

So 4 successes and 1 failure read `trust 0.71 over 5 outcomes`. A kind with no outcome reads
`trust 0.5, support 0` and is shown as **"no experience yet"** — never as trusted. `overall` is the mean of
the kinds weighted by their support. `evidence` is the record's last 20 outcomes, each with its run (when
there is one), kind, outcome, time and signal, so every number can be checked. There is no recency
weighting and no learning: the same record always gives the same reputation.

## Cost

One run-index listing (the entity's whole home store, or the session's runs), one run load per candidate
turn root (its origin and stamp are in the run, not in the index), one token query per 300 counted runs
and sub-runs on SQLite (other ledgers are read run by run), one read of each automation's controller ledger
when it has a counted occurrence, and one read of each home file. A host serving it often caches the
answer per `as_of`.

A session's record covers the runs listed under that session; a sub-run started into another session is
outside it.

## Errors

`TrackRecordError` (a `ValueError`) is raised, never papered over, when a record is not in the shape its
writer guarantees: a run in the index that cannot be loaded, an unknown run status, an unknown stamp phase,
an occurrence decision naming a run that is not one of its attempts, a task-inbox line that is not JSON
(except the last line, which a crash mid-append can leave torn), a manifest without `slug`, `entity_id`,
`home_id` or `spark_hash`, a malformed run identity ref, or a session whose newest run names another
subject's ref. An unreadable `identity.json` raises `IdentityUnreadable`.

## Acquaintances

Whom an entity (or a session) has met: two lists, the other **entities** and the **humans**, each with
when it first and last met them, how many interactions, and on which kinds of task. Like the track record
it is computed from existing records and never stored (nothing goes into `identity.json`). Module:
`abstractruntime.identity.acquaintances`.

```python
from abstractruntime.identity import entity_acquaintances, session_acquaintances

acq = entity_acquaintances(home_dir, er.run_store)
for a in acq.entities:      # first met first
    print(a.id, a.first_seen, a.last_seen, a.interactions, dict(a.kinds))
for a in acq.humans:
    print(a.id, a.interactions)
print(dict(acq.not_counted))  # interactions with no usable identifier, by reason

session_acquaintances("sess_9f2a", run_store)  # a regular agent's session
```

Where an interaction is read (identifiers only; no text is ever read):

| Source | Gives |
|---|---|
| a run's door-stamped participants, `_runtime.entity.participants` (visits, summons, both legs of an entity meet) | each participant other than the entity itself: `entity:<slug>` → entities, `person:<user>` → humans; one interaction per run |
| a run a person started, `_meta.origin` with kind `person` | its `actor` (`<tenant>:<user>`) → humans |
| a task in `task_inbox.jsonl` | the `by` of its `added` event → humans (kind `work/task_inbox`) |

Only turn roots are read (never sub-runs, which carry their root's stamp; never the reserved `__…__` runs),
whatever their origin: a visit is an interaction even when no origin was recorded for it.

Not counted, and counted in `not_counted` instead: a participant of another kind (`agent:<name>`, a
shared-room agent, is neither an entity nor a human), a participant without a namespace, and a person-started
run whose origin names no actor. Work orders record no author, so they name no one. The two human id forms
are kept as recorded and never merged: the door stamp's `person:<user>` has no tenant, the origin's
`<tenant>:<user>` does, so one person can appear in each form.

`identity_ref` names the subject's identity as for the track record. Cost: one run-index listing, one run
load per turn root, one read of the task inbox.
