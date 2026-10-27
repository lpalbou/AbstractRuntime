# Replay view: one reconstruction of a run, every client

`abstractruntime.replay` turns a run tree's ledger records and durable snapshots into the view a
client shows: the status, the approval or question waiting for a person, the replies, the tool
calls and the steps. It is the only code that does this ([ADR 0009](adr/0009_one_replay_many_views.md)).
Hosts serve the view, for example AbstractGateway's `GET /api/gateway/runs/{id}/view` and its
stream. Clients render it and act through the host's commands. They never re-derive run state
from raw records.

```python
from abstractruntime.replay import replay, apply_delta

view = replay(
    {root_id: ledger.list(root_id), child_id: ledger.list(child_id)},  # records per run, in ledger order
    {root_id: run_store.load(root_id), child_id: run_store.load(child_id)},  # snapshots (RunState or dict)
    root_run_id=root_id,
)
doc = view.to_json()                       # replay_view_v1, JSON-safe

delta = view.apply(ledger.list_after(run_id=child_id, after=view.last_seq(child_id))[0])
client_copy = apply_delta(doc, delta)      # == view.to_json()
```

## Contract

- **Pure and deterministic.** There is no I/O and no clock. The input is records plus snapshots,
  and the order between runs does not matter. `replay(all)` is `apply` over every record, so it
  equals folding the same records in any chunks.
- **Idempotent.** A record whose `seq` is at or below the run's folded seq is ignored.
- **Incremental.** `ReplayView.apply(records, runs_by_id=)` folds newer records (and newer
  snapshots) and returns the **delta**. The scalars come whole; the changed items of `runs`,
  `messages`, `activities` and `artifacts` come keyed by id. Items are never removed.
  `apply_delta(view, delta)` is the client side (about 40 lines to port), and
  `diff_views(old, new)` is the delta between two views (a reconnect).
- **Versioned.** `version: "replay_view_v1"`. A change in the meaning of a field is a new
  version.

## The rules it owns

Each rule below was once a divergence between clients:

| Question | Rule |
|---|---|
| Which runs belong to the view? | The root, plus every run it delegated to: `subworkflow:<id>` wait keys, `sub_run_id` in a wait's details or a `start_subworkflow` result, and the snapshots' `parent_run_id`. A ledger link outranks a snapshot's. |
| When is a wait over? | When its run records a `resume` for its key, OR any later step of the same run exists (a timer often has no resume record; the runtime also appends the resume before it saves the run without its wait). Observability events (`abstract.status`, `abstract.progress`) are not steps. A terminal record ends it too. |
| What is the person asked? | The newest open person-facing wait anywhere in the tree: a tool approval, a question, or an event wait that carries a prompt or choices. A delegation, a timer, a pause or an automation's own wake never is. A snapshot's wait counts only when the ledger never mentioned its key (a windowed history). |
| What is the status? | A terminal snapshot, then the ledger's terminal record (`system:completion:*`, `system:terminal_status:<s>`, a run-level failure), then an open wait (`waiting`), then `running`. |
| What about `$slim` pointers? | They are resolved against the STARTED record of the same step. One that cannot be resolved (a window) is dropped and named in `warnings`. A client never sees one. |
| How many steps did a node take? | One activity per node execution, following AbstractFlow's `executionIdFor` rules (`replay/executions.py`): the same step; a resume joins the execution whose wait it answers; a repeated idempotency key; an Agent's structured-output call and the run's terminal record join the node's latest execution; a progress tick reports on its step. |
| What does the run execute? | `workflow_ref(workflow_id)`: `bundle_id@version:flow_id` gives `{bundle_id, bundle_version, flow_id}`. The root's flow is its `entrypoint`. A workflow outside a bundle has `bundle_id: null`. |

## The view (`to_json()`)

| Field | Meaning |
|---|---|
| `version`, `root_run_id`, `session_id` | the view's version, its root run and the root's session |
| `workflow` | `{id, bundle_id, bundle_version, flow_id, entrypoint}` of the root run |
| `status`, `display_status` | the derived status; the run's own status line |
| `cursor` | `{token, records, runs}`. `runs` maps each run id to its last folded `seq`. `token` (`<run id prefix>:<seq>.…`, header-safe) is the host's ETag and SSE id; `parse_cursor_token` reads it |
| `runs[]` | the tree, root first, children in the order they were started: `parent_run_id`, `depth`, `workflow`, `status`, `waiting`, `holds_interaction`, `records`, `first_seq`, `last_seq`, `order` |
| `interaction` | the person-facing wait, or `null`: `kind`, `run_id`, `wait_key`, `step_id`, `seq`, `prompt`, `choices`, `allow_free_text`, `details`, `tool_calls` (resolved), `activity_id`, and `command` (the resume a client sends to answer it) |
| `open_waits[]` | every open wait of the tree, with `person_facing` |
| `messages[]` | verbatim turns: `role`, `kind` (`prompt`, `answer`, `question`, `final`, `event`), `text`, provenance (`run_id`, `step_ids`, `seqs`, `artifact_ids`, `source`) |
| `activities[]` | one per execution: `kind`, `name`, `status`, `node_id`, `execution_id`, `seqs`, `step_ids`, `tool_calls[]` (each call's status, output, error), `wait` (`kind`, `wait_key`, `ended_by`: `resume`, `later_step`, `terminal`), `usage`, `child_run_id`, `progress` |
| `artifacts[]` | every `{"$artifact": id}` reference, with the run, step, seq and JSON path that produced it |
| `usage` | tokens and LLM calls summed over the tree |
| `warnings[]` | what the view could not show (records before a window, an unresolved `$slim`) |

Every id is stable and JSON-addressable: run ids, `<run_id>:<execution id>` for activities,
message ids, artifact ids, and the per-run `seq` of every record they cite. A cross-run index or
a push can point at them.

## The durable cursor

Every ledger store numbers a run's records 1, 2, 3, … (`seq`):

- `JsonlLedgerStore` writes it on each line under an advisory file lock, so two processes never
  take the same number. A line written before the field existed is numbered by its position.
- `SqliteLedgerStore` allocates it in the insert's transaction.
- `InMemoryLedgerStore` numbers records as they are appended.

`list(run_id)` returns records with their `seq`. `list_after(run_id=, after=, limit=)` returns
the records after a seq. On SQLite it is an indexed read, and on JSONL it seeks to the remembered
byte offset. `ObservableLedgerStore` passes `seq` to subscribers.
`SqliteLedgerStore.import_records(run_id, records)` copies a run's ledger in with its `seq`
values (a file-to-SQLite migration), so a cursor a client holds stays valid.

## Conformance

`tests/fixtures/replay/<case>/` holds the cases: redacted real ledgers and synthetic ones. Each
has its records, its snapshots and the expected view. The cases cover a delegated child
approval, a timer wait without a resume, a parallel tool batch, a cancellation, Agent
structured-output steps, a windowed history with `$slim`, a question and its answer, and a
denied approval.

`tests/test_replay_conformance.py` checks each expected view plus hand-written semantic
assertions. `tests/test_replay_invariants.py` checks that replay equals fold under any chunking
and interleaving, idempotence, delta round trips and cursor tokens. A client that renders runs
runs the same fixtures through its host in CI. `tests/fixtures/replay/regen.py` rewrites the
expected views after a reviewed reducer change.
