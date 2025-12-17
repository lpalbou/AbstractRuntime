# Changelog

All notable changes to AbstractRuntime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
