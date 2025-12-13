# AbstractRuntime — Proposal (v0.1)

## Executive summary

**AbstractRuntime** is a low-level *durable workflow runtime*.

- It executes **workflow graphs** (state machines) where nodes can produce **effects** (LLM calls, tool calls, waits, ask-user, etc.).
- It supports **interrupt → checkpoint → resume** so a workflow can pause for hours/days without keeping Python stacks/coroutines alive.
- It records an append-only **execution journal** ("ledger" / *journal d’exécution*) so runs are observable, auditable, and debuggable.

**Scope boundary:** AbstractRuntime is not a UI builder and not an agent framework. It is the execution substrate that higher-level orchestration (e.g., AbstractFlow) and agents/memory pipelines can build on.

---

## Glossary (English ↔ Français)

- **Workflow**: a declared graph/state machine (nodes + transitions + a typed state).
  - FR: **workflow / flux** = un graphe d’exécution (machine à états).
- **Run**: one execution instance of a workflow (identified by `run_id`).
  - FR: **exécution**.
- **Interrupt**: a controlled pause point (waiting for user / timer / external event).
  - FR: **interruption / mise en attente**.
- **Resume**: continuing a paused run after receiving an external event.
  - FR: **reprise**.
- **Ledger**: execution journal / logbook. Append-only list of step records.
  - FR: **journal d’exécution**.
- **LedgerStore**: storage backend for the ledger.
  - FR: **stockage du journal d’exécution**.
- **RunStore**: storage backend for the current run state / checkpoints.
  - FR: **stockage des checkpoints / état de run**.
- **Snapshot**: named bookmark of run state (name/description/tags).
  - FR: **snapshot / marque-page**.
- **Effect**: a side-effect request produced by a node (LLM call, tool calls, wait, ask user, ...).
  - FR: **effet (action externe)**.

---

## Problem this solves

Once you allow **human-in-the-loop**, **timers**, and **external events**, you need durable semantics:

- a run can pause for days and survive restarts
- you can resume reliably on “user answered”, “job finished”, “time reached”, “webhook arrived”
- you can debug/audit: what step produced that tool call? what was the tool result? what did the LLM answer?

This cannot be done safely by keeping a Python stack/coroutine alive. You need persisted checkpoints + a journal.

---

## Design goals (v0.1)

- **Minimal kernel**: small API surface, few dependencies.
- **Durable by design**: pause/resume are explicit (`RunState.status=waiting` + `WaitState`).
- **Backend-agnostic stores**: in-memory + JSON/JSONL now; other stores later.
- **Layered coupling**: the kernel stays dependency-light; integrations (e.g. AbstractCore) live in a dedicated module.
- **Composable workflows**: workflow-as-node composition is supported by design (effect types exist).

## Non-goals (v0.1)

Avoid “Temporal-in-Python” (distributed orchestration backend) for now:
- worker leasing/heartbeats
- cluster-wide exactly-once progression
- global matching service / task queue protocols
- large-scale workflow versioning/migrations

We keep interfaces open so backends can evolve later.

---

## Core concepts (data model)

A workflow node returns a `StepPlan` which may:
- transition to another node (`next_node`)
- produce an `Effect` (side-effect to execute)
- complete the run (`complete_output`)

Key durable types:
- `RunState`: `run_id`, `workflow_id`, `status`, `current_node`, `vars`, `waiting`, `output`, `error`, `actor_id`
- `WaitState`: durable pause descriptor (`reason`, `wait_key`, `until`, `resume_to_node`, `result_key`, optional `details`)
- `StepRecord`: append-only ledger entry (`effect`, `result`, timestamps, status, actor_id)
- `EffectType`: `llm_call`, `tool_calls`, `wait_event`, `wait_until`, `ask_user`, ...

---

## Minimal runtime API

- `start(workflow, vars, actor_id) -> run_id`
- `tick(workflow, run_id) -> RunState` (progresses until waiting/completed/failed)
- `resume(workflow, run_id, wait_key, payload) -> RunState`
- `get_state(run_id) -> RunState`
- `get_ledger(run_id) -> list[dict]`

### Resume semantics (important)
When an effect yields a waiting outcome, the runtime stores `WaitState.resume_to_node`.
On resume, the runtime will continue execution **from that node**.

This avoids re-running the waiting node and is required for correctness.

---

## Persistence

The kernel persists context via:
- `RunStore`: latest checkpoint (`RunState`) — JSON file backend included
- `LedgerStore`: append-only step records — JSONL backend included
- `SnapshotStore`: named bookmarks of run state — JSON backend included

**ArtifactStore** (planned): store large payloads by reference instead of embedding in `RunState.vars`.

**Constraint (non-negotiable):** values stored in `RunState.vars` must be JSON-serializable (or referenced via artifacts).

---

## Integration with AbstractCore (LLM + tools)

AbstractCore already provides:
- provider-agnostic `create_llm(...).generate(...)`
- tool registry + execution
- an OpenAI-compatible server (`/v1/chat/completions`)

AbstractRuntime integrates via an explicit module:
- `abstractruntime.integrations.abstractcore`

Execution modes:
- **Local**: in-process AbstractCore provider + local tool execution
- **Remote**: HTTP to AbstractCore server + tool passthrough (default)
- **Hybrid**: remote LLM + local tools

Remote mode supports AbstractCore’s per-request dynamic routing (`base_url` in `/v1/chat/completions`).

---

## Provenance / AI fingerprint (v0.1)

The runtime supports attribution via `actor_id` on `RunState` and `StepRecord`.

For tamper-evidence:
- wrap a `LedgerStore` with `HashChainedLedgerStore`
- each record gets `prev_hash` + `record_hash`
- `verify_ledger_chain(records)` reports mismatches

Signatures (non-forgeability) are intentionally deferred to an optional extra.

---

## Relationship to AbstractAgent / AbstractMemory / AbstractFlow

- **AbstractCore**: inference plane (LLM + tools + server boundary)
- **AbstractRuntime**: durable execution substrate (pause/resume/ledger)
- **AbstractAgent**: agent logic (ReAct/CodeAct/state machines) built *on top* of runtime + core
- **AbstractMemory**: memory services and maintenance pipelines built *on top* of runtime + core
- **AbstractFlow**: high-level authoring/orchestration that composes workflows/agents

This avoids coupling the stateless router (AbstractCore server) with stateful long-running control loops.

---

## Status (implemented in this repository)

- Durable kernel: `RunState`, `WaitState`, `Runtime.start/tick/resume`
- Built-in waits: `wait_event`, `wait_until`, `ask_user`
- Persistence: in-memory + JSON checkpoints + JSONL ledger
- Snapshots: `SnapshotStore` (in-memory + JSON)
- Provenance: `HashChainedLedgerStore` + `verify_ledger_chain`
- AbstractCore integration module: local/remote/hybrid adapters
- Unit tests covering pause/resume, integration wiring, snapshots, and provenance
