# Changelog

All notable changes to AbstractRuntime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Durable prompt metadata for EVENT waits**:
  - `WAIT_EVENT` effects may include optional `prompt`, `choices`, and `allow_free_text` fields.
  - The runtime persists these fields onto `WaitState` so hosts (including remote/thin clients) can render a durable ask+wait UX without relying on in-process callbacks.
- **Rendering utilities** (`abstractruntime.rendering`):
  - `stringify_json(...)` + `JsonStringifyMode` to render JSON/JSON-ish values into strings with `none|beautify|minified` modes.
  - `render_agent_trace_markdown(...)` to render runtime-owned `node_traces` scratchpads into a complete, review-friendly Markdown timeline (tool args + results untruncated).

## [0.4.0] - 2025-01-06

### Added

- **Active Memory System** (`abstractruntime.memory.active_memory`): Complete MemAct agent memory module
  - Runtime-owned `ACTIVE_MEMORY_DELTA` effect for structured Active Memory updates (used by agents via `active_memory_delta` tool)
  - JSON-safe durable storage in `run.vars["_runtime"]["active_memory"]`
  - Memory modules: MY PERSONA, RELATIONSHIPS, MEMORY BLUEPRINTS, CURRENT TASKS, CURRENT CONTEXT, CRITICAL INSIGHTS, REFERENCES, HISTORY
  - Active Memory v9 format with natural-language markdown rendering (not YAML) to reduce syntax contamination
  - All components render into system prompt by default (prevents user-role pollution on native-tool providers)

- **MCP Worker** (`abstractruntime-mcp-worker`): Standalone stdio-based MCP server for AbstractRuntime tools
  - Exposes AbstractRuntime's default toolsets as MCP tools via stdio transport
  - Human-friendly logging to stderr with ANSI color support
  - Security: allowlist-based command execution safety (`TOOL_WAIT` effect for dangerous commands)
  - New optional dependency: `abstractruntime[mcp-worker]` (includes `abstractcore[tools]`)
  - Entry point: `abstractruntime-mcp-worker` CLI script

- **Evidence Capture System** (`abstractruntime.evidence.recorder`): Always-on provenance-first evidence recording
  - Automatically records evidence for external-boundary tools: `web_search`, `fetch_url`, `execute_command`
  - Evidence stored as artifact-backed records indexed as `kind="evidence"` in `RunState.vars["_runtime"]["memory_spans"]`
  - Runtime helpers: `Runtime.list_evidence(run_id)` and `Runtime.load_evidence(evidence_id)`
  - Keeps RunState JSON-safe by storing large payloads in ArtifactStore with refs

- **Ledger Subscriptions**: Real-time step append events via `Runtime.subscribe_ledger()`
  - `create_local_runtime`, `create_remote_runtime`, `create_hybrid_runtime` now wrap LedgerStore with `ObservableLedgerStore` by default
  - Hosts can receive real-time notifications when steps are appended to ledger

- **Durable Custom Events (Signals)**:
  - `EMIT_EVENT` effect to dispatch events and resume matching `WAIT_EVENT` runs
  - Extended `WAIT_EVENT` to accept `{scope, name}` payloads (runtime computes stable `wait_key`)
  - `Scheduler.emit_event(...)` host API for external event delivery (session-scoped by default)

- **Orchestrator-Owned Timeouts** (AbstractCore integration):
  - Default **LLM timeout**: 7200s per `LLM_CALL` (not per-workflow), enforced by `create_*_runtime` factories
  - Default **tool execution timeout**: 7200s per tool call (not per-workflow), enforced by ToolExecutor implementations

- **Tool Executor Enhancements** (`MappingToolExecutor`):
  - **Argument canonicalization**: Maps common parameter name variations (e.g., `file_path`/`filepath`/`path`) to canonical names
  - **Filename aliases**: Supports `target_file`, `file_path`, `filepath`, `path` as aliases for file operations
  - **Error output detection**: Detects structured error responses (`{"success": false, ...}`) from tools
  - **Argument sanitization**: Cleans and validates tool call arguments
  - **Timeout support**: Per-tool execution timeouts with configurable limits

- **Memory Query Enhancements** (`MEMORY_QUERY` effect):
  - Tag filters with **AND/OR** modes (`tags_mode=all|any`) and **multi-value** keys (`tags.person=["alice","bob"]`)
  - Metadata filters for **authors** (`created_by`) and **locations** (`location`, `tags.location`)
  - Span records now capture `created_by` for `conversation_span`, `active_memory_span`, `memory_note` when `actor_id` available
  - `MEMORY_NOTE` accepts optional `location` field
  - `MEMORY_NOTE` supports `keep_in_context=true` flag to immediately rehydrate stored note into `context.messages`

- **Package Dependencies**:
  - New optional dependency: `abstractruntime[abstractcore]` (enables `abstractruntime.integrations.abstractcore.*`)
  - New optional dependency: `abstractruntime[mcp-worker]` (includes `abstractcore[tools]>=2.6.8`)

### Changed

- **LLM Client Enhancements**:
  - Tool call parsing refactored for better robustness and error handling
  - Streaming support with timing metrics (TTFT, generation time)
  - Response normalization preserves JSON-safe `raw_response` for debugging
  - Always attaches exact provider request payload under `result.metadata._provider_request` for every `LLM_CALL` step

- **Runtime Core** (902 lines changed):
  - Enhanced resume handling for paused/cancelled runs
  - Improved subworkflow execution with async+wait support
  - Better observable ledger integration

### Fixed

- **Cancellation is Terminal**: `Runtime.tick()` now treats `RunStatus.CANCELLED` as terminal and will not progress cancelled runs
- **Control-Plane Safety**: `Runtime.tick()` stops without overwriting externally persisted pause/cancel state (used by AbstractFlow Web)
- **Atomic Run Checkpoints**: `JsonFileRunStore.save()` writes via temp file + atomic rename to prevent partial/corrupt JSON under concurrent writes
- **START_SUBWORKFLOW async+wait**: Support for `async=true` + `wait=true` to start child run without blocking parent tick, while keeping parent in durable SUBWORKFLOW wait
- **ArtifactStore Run-Scoped Addressing**: Artifact IDs namespaced to run when `run_id` provided (prevents cross-run collisions, preserves purge-by-run semantics)
- **AbstractCore Integration Imports**: `LocalAbstractCoreLLMClient` imports `create_llm` robustly in monorepo namespace-package layouts
- **Token Limit Metadata**: `_limits.max_output_tokens` falls back to model capabilities when not configured (runtime surfaces explicit per-step output budget)
- **Token-Cap Normalization Boundary**: Removed local `max_tokens → max_output_tokens` aliasing from AbstractRuntime's AbstractCore client (AbstractCore providers own this mapping)

### Testing

- **25 new/modified test files** covering:
  - Active Memory functionality
  - MCP worker (logging, security, stdio communication)
  - Evidence recorder
  - Memory query rich filters
  - Tool executor (canonicalization, filename aliases, timeouts, error detection)
  - LLM client tool call parsing
  - Runtime configuration and subworkflow handling
  - Packaging extras validation

### Statistics

- **33 commits** improving memory systems, MCP integration, evidence capture, and tool execution
- **45 files changed**: 5,788 insertions, 286 deletions
- **6,074 total lines changed** across the codebase
- **3 new modules**: `active_memory.py`, `evidence/recorder.py`, `mcp_worker.py`

## [0.2.0] - 2025-12-17

### Added

#### Core Runtime Features
- **Durable Workflow Execution**: Start/tick/resume semantics for long-running workflows that survive process restarts
- **WorkflowSpec**: Graph-based workflow definitions with node handlers keyed by ID
- **RunState**: Durable state management (`current_node`, `vars`, `waiting`, `status`)
- **Effect System**: Side-effect requests including `LLM_CALL`, `TOOL_CALLS`, `ASK_USER`, `WAIT_EVENT`, `WAIT_UNTIL`, `START_SUBWORKFLOW`
- **StepPlan**: Node execution plans that define effects and state transitions
- **Explicit Waiting States**: First-class support for pausing execution (`WaitReason`, `WaitState`)

#### Scheduler & Automation
- **Built-in Scheduler**: Zero-config background scheduler with polling thread for automatic run resumption
- **WorkflowRegistry**: Mapping from workflow_id to WorkflowSpec for dynamic workflow resolution
- **ScheduledRuntime**: High-level wrapper combining Runtime + Scheduler with simplified API
- **create_scheduled_runtime()**: Factory function for zero-config scheduler creation
- **Event Ingestion**: Support for external event delivery via `scheduler.resume_event()`
- **Scheduler Stats**: Built-in statistics tracking and callback support

#### Storage & Persistence
- **Append-only Ledger**: Execution journal with `StepRecord` entries for audit/debug/provenance
- **InMemoryRunStore**: In-memory run state storage for development and testing
- **InMemoryLedgerStore**: In-memory ledger storage for development and testing
- **JsonFileRunStore**: File-based persistent run state storage (one file per run)
- **JsonlLedgerStore**: JSONL-based persistent ledger storage
- **QueryableRunStore**: Interface for listing and filtering runs by status, workflow_id, actor_id, and time range
- **Artifacts System**: Storage for large payloads (documents, images, tool outputs) to avoid bloating checkpoints
  - `ArtifactStore` interface with in-memory and file-based implementations
  - `ArtifactRef` type for referencing stored artifacts
  - Helper functions: `artifact_ref()`, `is_artifact_ref()`, `get_artifact_id()`, `resolve_artifact()`, `compute_artifact_id()`

#### Snapshots & Bookmarks
- **Snapshot System**: Named, searchable checkpoints of run state for debugging and experimentation
- **SnapshotStore**: Storage interface for snapshots with metadata (name, description, tags, timestamps)
- **InMemorySnapshotStore**: In-memory snapshot storage for development
- **JsonSnapshotStore**: File-based snapshot storage (one file per snapshot)
- **Snapshot Search**: Filter by run_id, tag, or substring match in name/description

#### Provenance & Accountability
- **Hash-Chained Ledger**: Tamper-evident ledger with `prev_hash` and `record_hash` for each step
- **HashChainedLedgerStore**: Decorator for adding hash chain verification to any ledger store
- **verify_ledger_chain()**: Verification function that detects modifications or reordering of ledger records
- **Actor Identity**: `ActorFingerprint` for attribution of workflow execution to specific actors
- **actor_id tracking**: Support for actor_id in both RunState and StepRecord for accountability

#### AbstractCore Integration
- **LLM_CALL Effect Handler**: Execute LLM calls via AbstractCore providers
- **TOOL_CALLS Effect Handler**: Execute tool calls with support for multiple execution modes
- **Three Execution Modes**:
  - **Local**: In-process AbstractCore providers with local tool execution
  - **Remote**: HTTP to AbstractCore server (`/v1/chat/completions`) with tool passthrough
  - **Hybrid**: Remote LLM calls with local tool execution
- **Convenience Factories**: `create_local_runtime()`, `create_remote_runtime()`, `create_hybrid_runtime()`
- **Tool Execution Modes**:
  - Executed mode (trusted local) with results
  - Passthrough mode (untrusted/server) with waiting semantics
- **Layered Coupling**: AbstractCore integration as opt-in module to keep kernel dependency-light

#### Effect Policies & Reliability
- **EffectPolicy Protocol**: Configurable retry and idempotency policies for effects
- **DefaultEffectPolicy**: Default implementation with no retries
- **RetryPolicy**: Configurable retry behavior with max_attempts and backoff
- **NoRetryPolicy**: Explicit no-retry policy
- **compute_idempotency_key()**: Ledger-based deduplication to prevent duplicate side effects after crashes

#### Examples & Documentation
- **7 Runnable Examples**:
  - `01_hello_world.py`: Minimal workflow demonstration
  - `02_ask_user.py`: Pause/resume with user input
  - `03_wait_until.py`: Scheduled resumption with time-based waiting
  - `04_multi_step.py`: Branching workflow with conditional logic
  - `05_persistence.py`: File-based storage demonstration
  - `06_llm_integration.py`: AbstractCore LLM call integration
  - `07_react_agent.py`: Full ReAct agent implementation with tools
- **Comprehensive Documentation**:
  - Architecture Decision Records (ADRs) for key design choices
  - Integration guides for AbstractCore
  - Detailed documentation for snapshots and provenance
  - Limits and constraints documentation
  - ROADMAP with prioritized next steps

### Technical Details

#### Architecture
- **Layered Design**: Clear separation between kernel, storage, integrations, and identity
- **Dependency-Light Kernel**: Core runtime remains stable with minimal dependencies
- **Graph-Based Execution**: All workflows represented as state machines/graphs for visualization and composition
- **JSON-Serializable State**: All run state and vars must be JSON-serializable for persistence

#### Test Coverage
- **81% Overall Coverage**: Comprehensive test suite with 57+ tests
- **Integration Tests**: Tests for AbstractCore integration, subworkflows, trace propagation
- **Core Tests**: Scheduler, snapshots, artifacts, pause/resume, retry/idempotency, ledger chain
- **Storage Tests**: Queryable run store, durable toolsets

#### Compatibility
- **Python 3.10+**: Supports Python 3.10, 3.11, 3.12, and 3.13
- **Development Status**: Planning/Alpha (moving toward Beta with 0.2.0)

### Known Limitations

- Snapshot restore does not guarantee safety if workflow spec or node code has changed
- Subworkflow support (`START_SUBWORKFLOW`) is implemented but undergoing refinement
- Cryptographic signatures (non-forgeability) not yet implemented - current hash chain provides tamper-evidence only
- Remote tool worker service not yet implemented

### Design Decisions

- **Kernel stays dependency-light**: Enables portability, stability, and clear integration boundaries
- **AbstractCore integration is opt-in**: Layered coupling prevents kernel breakage when AbstractCore changes
- **Hash chain before signatures**: Provides immediate value without key management complexity
- **Built-in scheduler (not external)**: Zero-config UX for simple cases
- **Graph representation for all workflows**: Enables visualization, checkpointing, and composition

### Notes

AbstractRuntime is the durable execution substrate designed to pair with AbstractCore, AbstractAgent, and AbstractFlow. It enables workflows to interrupt, checkpoint, and resume across process restarts, making it suitable for long-running agent workflows that need to wait for user input, scheduled events, or external job completion.

## [0.0.1] - Initial Development

Initial development version with basic proof-of-concept features.

[0.2.0]: https://github.com/lpalbou/abstractruntime/releases/tag/v0.2.0
[0.0.1]: https://github.com/lpalbou/abstractruntime/releases/tag/v0.0.1
