# AbstractRuntime Roadmap

## Current Status: MVP Complete (v0.1)

AbstractRuntime has a functional MVP kernel with:
- Durable workflow execution (start/tick/resume)
- Explicit waiting states (wait_event, wait_until, ask_user)
- Persistence (in-memory + JSON/JSONL file backends)
- Snapshots/bookmarks for named checkpoints
- Tamper-evident provenance (hash-chained ledger)
- AbstractCore integration (local/remote/hybrid execution modes)
- **QueryableRunStore** for listing/filtering runs ✅
- **Built-in Scheduler** with zero-config operation ✅
- Test coverage for core functionality (81% overall, 57 tests)

---

## Why AbstractRuntime Exists

AbstractRuntime is the **durable execution substrate** that sits below both AbstractAgent and AbstractFlow:

```
AbstractFlow (UI graphs, multi-agent orchestration, templates)
     │
AbstractAgent (ReAct, CodeAct, specialized agents like DeepSearch)
     │
AbstractRuntime (interrupt/checkpoint/resume, ledger, snapshots)
     │
AbstractCore (LLM calls, tool execution, server API)
```

**Key insight**: You cannot keep Python stacks alive for days. When an agent needs to:
- Ask a user a question and wait hours/days for a response
- Wait until a scheduled time
- Wait for an external job to complete

...you need **durable state that survives process restarts**. This is what AbstractRuntime provides.

**Why not in AbstractFlow?** Because individual agents need these primitives directly. A ReAct agent that calls `ask_user` needs to pause and resume — that's not orchestration, that's the agent itself needing durability.

---

## Phase 1: Core Completeness ✅ COMPLETE

### 1.1 Built-in Scheduler (Zero-Config) ✅
**Status: COMPLETE** | **Backlog: 004**

**What shipped**:
- `WorkflowRegistry` for mapping workflow_id → WorkflowSpec
- `Scheduler` class with background polling thread
- `ScheduledRuntime` convenience wrapper
- `create_scheduled_runtime()` factory function
- Event ingestion via `scheduler.resume_event()`
- Stats tracking and callbacks

**Usage**:
```python
# Zero-config: defaults to in-memory, auto-starts scheduler
sr = create_scheduled_runtime()

# run() does start + tick in one call
run_id, state = sr.run(my_workflow)

# respond() auto-finds wait_key
if state.status.value == "waiting":
    state = sr.respond(run_id, {"answer": "yes"})

sr.stop()
```

---

## Phase 2: Composition and Examples (Next Priority)

### 2.1 Subworkflow Support
**Priority: High** | **Effort: Medium** | **Backlog: 011**

**Why**: `EffectType.START_SUBWORKFLOW` exists but has no handler. Without this:
- Multi-agent orchestration is impossible
- Workflow composition (DeepSearch as a node) doesn't work
- AbstractFlow cannot compose agents

**Deliverables**:
- Workflow registry interface
- `START_SUBWORKFLOW` effect handler
- Sync and async subworkflow modes
- Tests for parent/child workflow interaction

**Success criteria**: A workflow can invoke another workflow by ID and receive its output.

---

### 2.2 Examples and Documentation ✅ COMPLETE
**Priority: High** | **Effort: Low** | **Backlog: 010**

**What shipped**:
- `examples/` directory with 7 runnable examples
- 01_hello_world.py - Minimal workflow
- 02_ask_user.py - Pause/resume with user input
- 03_wait_until.py - Scheduled resumption
- 04_multi_step.py - Branching workflow
- 05_persistence.py - File-based storage
- 06_llm_integration.py - AbstractCore LLM call
- 07_react_agent.py - Full ReAct agent with tools

**Success criteria**: A developer can copy an example and have a working workflow in 5 minutes. ✅

---

## Phase 3: Production Readiness

### 3.1 Artifact Store
**Priority: Medium** | **Effort: Medium** | **Backlog: 009**

**Why**: Large payloads (documents, images, tool outputs) embedded in `RunState.vars` cause performance issues. The constraint that vars must be JSON-serializable becomes painful without by-reference storage.

**Deliverables**:
- `ArtifactStore` interface
- File-based implementation
- `ArtifactRef` type for referencing stored artifacts
- Integration with RunState serialization

**Success criteria**: A workflow can store a 10MB document without bloating the checkpoint.

---

### 3.2 Effect Retries and Idempotency
**Priority: Medium** | **Effort: Medium** | **Backlog: 013**

**Why**: At-least-once execution is a real risk. If a process crashes after an LLM call but before checkpointing, the call may be duplicated on restart.

**Deliverables**:
- `EffectPolicy` protocol (max_attempts, backoff, idempotency_key)
- Ledger-based deduplication
- Retry logic in runtime loop

**Success criteria**: A workflow survives a crash/restart without duplicating side effects.

---

## Phase 4: Advanced Features

### 4.1 Cryptographic Signatures
**Priority: Low** | **Effort: High** | **Backlog: 008**

**Why**: The hash chain provides tamper-evidence but not non-forgeability. For high-accountability scenarios (AI fingerprinting, regulatory compliance), cryptographic signatures are needed.

**Dependencies**: 007 (Hash Chain - complete)

**Deliverables**:
- Ed25519 keypair generation
- `actor_id` bound to public key
- Signed `StepRecord` entries
- Signature verification in `verify_ledger_chain`
- Key storage patterns (file, OS keychain, Vault)

**Success criteria**: A ledger can prove which actor produced each step.

---

### 4.2 Remote Tool Worker
**Priority: Low** | **Effort: Medium** | **Backlog: 014**

**Why**: Some deployments need centralized tool execution (thin clients, sandboxed environments). The current passthrough mode requires the host to execute tools; a worker service would handle this automatically.

**Deliverables**:
- Tool worker API contract
- `RemoteToolExecutor` implementation
- Job-based waiting semantics

**Success criteria**: A thin client can run workflows with tools by delegating to a worker service.

---

## Relationship to the Abstract Series

AbstractRuntime is the **durable execution substrate** that enables both agents and memory to have long-running, interruptible workflows.

```
AbstractFlow (UI graphs, multi-agent orchestration, templates)
     │
     ├── AbstractSwarm (swarm of agents - future)
     │
AbstractAgent (ReAct, CodeAct, DeepSearch, specialized agents)
     │
     ├── AbstractMemory (agentic memory - consolidation, reflection, forgetting)
     │
AbstractRuntime (interrupt/checkpoint/resume, ledger, snapshots)
     │
AbstractCore (LLM calls, tool execution, server API)
```

**Why this layering?**
- **AbstractAgent** needs durability primitives directly (an agent might `ask_user` and wait days)
- **AbstractMemory** needs the same primitives (memory consolidation might run on a schedule)
- **AbstractFlow** composes agents and workflows, using AbstractRuntime for execution
- **AbstractRuntime** provides the substrate; it doesn't know about "agents" or "memory" — just workflows

**Snapshots enable time-travel debugging:**
- Save a named checkpoint at any point
- Restore a multi-agent system to a previous state
- Rebuild context from ledger + snapshot

---

## Decision Log

| Decision | Rationale |
|----------|-----------|
| Kernel stays dependency-light | Enables portability, stability, and clear integration boundaries |
| AbstractCore integration is opt-in | Layered coupling prevents kernel from breaking when AbstractCore changes |
| Hash chain before signatures | Provides value immediately without key management complexity |
| Built-in scheduler (not external) | UX principle: simplify as much as possible; zero-config for simple cases |
| Graph representation for all workflows | A loop is a simple graph; graphs enable visualization, checkpointing, composition |

---

## Timeline (Estimated)

| Phase | Items | Status |
|-------|-------|--------|
| 1.1 | Scheduler + RunStore Query | ✅ Complete |
| 2.1 | Subworkflow | 2-3 days |
| 2.2 | Examples | 1-2 days |
| 3.1 | Artifact Store | 2-3 days |
| 3.2 | Retries/Idempotency | 3-4 days |
| 4.1 | Signatures | 1 week |
| 4.2 | Remote Worker | 3-4 days |

**Phase 1 (Core Completeness)**: ✅ Complete
**Phase 2 (Composition)**: ✅ Examples complete, Subworkflow complete
**Phase 3 (Production Readiness)**: ~1 week  
**Phase 4 (Advanced Features)**: ~2 weeks

Remaining: ~2-3 weeks for full roadmap.
