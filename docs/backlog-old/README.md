## AbstractRuntime v0.1 (temp backlog in AbstractCore)

This folder is a **temporary, implementation-grade backlog/spec** for the `abstractruntime` repository.

It exists because our environment can reliably modify files **inside this workspace** (`abstractcore/`) while writes outside the workspace were previously blocked/intermittent.

The content below consolidates **all prior architecture discussions, decisions, and designs** into a single, consistent implementation plan.

---

### 1) Context: what we are building (and what we are not)

#### **AbstractCore** (inference/routing plane)
- **Role**: thin, provider-agnostic LLM interface + server exposing OpenAI-compatible endpoints.
- **Design bias**: **stateless and horizontally scalable** routing layer.
- **Tools**: AbstractCore *can* execute tools (tool registry + execution), even if the server often runs in pass-through mode.

#### **AbstractAgent** (control plane)
- **Must be a separate repo** at `/Users/albou/projects/abstractagent/`.
- Hosts agentic orchestration patterns (ReAct/CodeAct/state-machine/Markov-style multi-agent), sandboxing policy, HITL UX.
- Needs **thin vs full client modes**:
  - **Thin**: HTTP client only (calls AbstractCore server)
  - **Full**: embeds AbstractCore locally (useful on capable machines)

#### **AbstractMemory** (state plane)
- Long-term memory is a complex system (episodic/semantic/vector/graph/etc.).
- **Key architectural decision**: treat `AbstractMemory` and `AbstractAgent` as **peers** that both build on top of AbstractCore capabilities.
  - Memory can internally use agentic workflows.
  - Agents can use memory as a tool/source.
  - Avoid a rigid “Memory powered by Agent” or “Agent powered by Memory” dependency that forces tight coupling.

#### **AbstractFlow** (high-level orchestration)
- Higher-level authoring/composition layer (UI graphs, templates, workflow-as-node composition, multi-agent composition).
- Owns the *product-level* orchestration experience.

#### **AbstractRuntime** (low-level durable execution substrate)
- This is the missing low-level primitive we converged on after the discussion:
  - Handles **interrupt → checkpoint → resume** reliably.
  - Models **elementary** long-running steps: `ask_user`, `wait_until`, `wait_event`, `wait_job`, `llm_call`, `tool_calls`.
  - Keeps an **append-only step ledger** for audit/debug/provenance.
- It is **not** an agent framework and **not** a UI builder.

---

### 2) Naming + coupling decisions (locked)

- **Package naming**:
  - Low-level kernel: `abstractruntime` (PyPI name reserved)
  - High-level orchestrator: `abstractflow` (PyPI placeholder owned)

- **Layered coupling with AbstractCore**:
  - `abstractruntime` core must stay dependency-light and **must not import AbstractCore**.
  - `abstractruntime` ships an **integration module** (`abstractruntime.integrations.abstractcore`) that imports AbstractCore and provides effect handlers.

- **Execution modes** (must both exist):
  - **Local**: in-process AbstractCore (direct `create_llm(...).generate(...)`, local tool registry)
  - **Remote**: HTTP to AbstractCore server (`/v1/chat/completions`), tools generally pass-through unless explicitly trusted
  - **Hybrid**: remote LLM + local tools

---

### 3) Core runtime primitives (already implemented in `abstractruntime` repo)

The kernel already contains:
- `RunState` / `WaitState` (durable state)
- `Effect` / `EffectType` (side-effect requests)
- `StepRecord` (append-only ledger entries)
- `Runtime.start()` / `Runtime.tick()` / `Runtime.resume()`
- File persistence: JSON checkpoints + JSONL ledger

This backlog focuses on adding:
- **AbstractCore integration module** for `llm_call` and `tool_calls`
- **Snapshots/bookmarks**
- **Tamper-evident provenance** (hash-chained ledger)
- **Tests** that don’t require live external services
- **Docs updates** to reflect reality (currently README still says “placeholder”)

---

### 4) What you will find in this backlog

- `001_integrations_abstractcore.md`
  - Concrete file layout + interfaces + payload schemas
  - Local vs remote vs hybrid wiring
  - Effect handlers for `EffectType.LLM_CALL` and `EffectType.TOOL_CALLS`

- `002_snapshots_bookmarks.md`
  - Snapshot model + store interface + JSON backend + search semantics

- `003_provenance_ledger_chain.md`
  - Hash-chained ledger design + verification utility
  - How it relates to ActorFingerprint
  - Signatures kept optional

- `004_tests.md`
  - Unit tests design (no live network)
  - Test scaffolding + recommended dependency injection seams

- `005_docs_updates.md`
  - Exact doc changes to apply to `abstractruntime/docs/proposal.md` and `abstractruntime/README.md`

---

### 5) Implementation order (matches the plan todos)

1. **Integrations (AbstractCore)**
2. **Snapshots/bookmarks**
3. **Provenance (hash chain)**
4. **Tests**
5. **Docs**


