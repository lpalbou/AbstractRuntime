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
  `messages`, `activities` and `artifacts` come keyed by id. Items are never removed. A delta
  also lists its typed `changes`. Each change is `{kind, id, run_id, seq}`, where `kind` is
  one of:
  - `status`, which adds `status` and `previous`;
  - `interaction`, which adds `change: opened|closed`, `interaction_kind` and `wait_key`;
  - `run`, `message`, `activity` or `artifact`.

  A push dispatcher or a cross-run index subscribes to these changes. `apply_delta` ignores
  them.
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
| What is the person asked? | The newest open person-facing wait anywhere in the tree. `wait_kind` gives the kinds and agrees with `automations.attention`, which the gateway's attention stream uses. A tool approval is `tool_approval` (`details.mode == "approval_required"`). A USER wait is `ask_user`. An EVENT wait with a prompt or choices is `event`. None of these are person-facing: a delegation (`delegation`), a timer (`timer`), a job, a pause (`pause:<run>` or `details.kind == "pause"`), an automation's own wake (`automation_wake`) and an EVENT wait without a prompt (`signal`). The interaction carries `answer_payload`, the shape the resume payload takes. A snapshot's wait counts only when the ledger never mentioned its key (a windowed history). |
| What is the status? | A terminal snapshot, then the ledger's terminal record (`system:completion:*`, `system:terminal_status:<s>`, a run-level failure), then an open wait (`waiting`), then `running`. |
| What about `$slim` pointers? | They are resolved against the STARTED record of the step they name. An approved tool call completes under a new step id that points back at the waiting step. A pointer that cannot be resolved (a window) is dropped and named in `warnings`. A client never sees one. |
| What is the final answer? | One message, `<root>:final`, from the root's terminal record (or the snapshot's output when the ledger has none). When it repeats an answer the run already gave, it names that message in `repeats`, and a client shows the text once. |
| How many steps did a node take? | One activity per node execution, following AbstractFlow's `executionIdFor` rules (`replay/executions.py`): the same step; a resume joins the execution whose wait it answers; a repeated idempotency key; an Agent's structured-output call and the run's terminal record join the node's latest execution; a progress tick reports on its step. |
| What did a conversation's tool policy decide? | An `abstract.tool_policy_decision` record (session tool policy contract v1) is an activity of kind `tool_policy` (`decision`: `auto_approved` or `denied`, the policy's `revision`, `set_by`, `set_at`), never an interaction: such a call has no wait. The tool rows it names carry `decided_by: "session_policy"`, and a call the policy refused is `denied`, not failed. A decision written twice (a crash replay, same idempotency key) is one activity. A deny wins even over an approval gate answered "approved" before it: the batch reads `denied`, with `wait.outcome: denied` and `wait.denied_by: session_policy`. Every tool-approval activity says how its gate ended (`wait.outcome`: approved, denied, partly_denied). |
| What does the run execute? | `workflow_ref(workflow_id)`: `bundle_id@version:flow_id` gives `{bundle_id, bundle_version, flow_id}`. The root's flow is its `entrypoint`. A workflow outside a bundle has `bundle_id: null`. |

## The view (`to_json()`)

| Field | Meaning |
|---|---|
| `version`, `root_run_id`, `session_id` | the view's version, its root run and the root's session |
| `workflow` | `{id, bundle_id, bundle_version, flow_id, entrypoint}` of the root run |
| `status`, `display_status` | the derived status; the run's own status line |
| `cursor` | `{token, records, runs}`. `runs` maps each run id to its last folded `seq`. `token` (`<run id prefix>:<seq>.…`, header-safe) is the host's ETag and SSE id; `parse_cursor_token` reads it |
| `runs[]` | the tree in path order (a child after its parent, siblings in the order they were started): `parent_run_id`, `depth`, `workflow`, `status`, `waiting`, `holds_interaction`, `records`, `first_seq`, `last_seq`, `order` (the path from the root) |
| `interaction` | the person-facing wait, or `null`: `kind` (`tool_approval`, `ask_user`, `event`), `run_id`, `wait_key`, `step_id`, `seq`, `prompt`, `choices`, `allow_free_text`, `details`, `tool_calls` (resolved), `activity_id`, `answer_payload`, and `command` (the resume a client sends to answer it) |
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

`abstractruntime.replay.conformance` holds the cases, and they ship with the package
(`case_names()`, `load_case(name)`). The cases are redacted real ledgers and synthetic ones. Each
has its records, its snapshots and the expected view. The cases cover a delegated child
approval, a timer wait without a resume, a parallel tool batch, a cancellation, Agent
structured-output steps, a windowed history with `$slim`, a question and its answer, and a
denied approval, a session tool policy's decisions, plus every wait kind, an event answered then a pause, nested delegation, and an
answer repeated by the completion.

`tests/test_replay_conformance.py` checks each expected view plus hand-written semantic
assertions. `tests/test_replay_invariants.py` checks that replay equals fold under any chunking
and interleaving, idempotence, delta round trips and cursor tokens. A client that renders runs
runs the same fixtures through its host in CI (AbstractGateway serves every case through `/view`
in its tests). `scripts/regen_replay_conformance.py` reports which expected views a reducer
change alters, and `--write` rewrites them after review.
