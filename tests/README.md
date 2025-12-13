# AbstractRuntime Test Suite

## Overview

The test suite validates AbstractRuntime's core functionality and its integration with AbstractCore. Tests are organized by component and capability.

## Test Structure

```
tests/
├── test_runtime.py              # Core runtime: tick, resume, effects
├── test_scheduler.py            # Scheduler: auto-resume, polling
├── test_subworkflow.py          # Subworkflows: composition, cancellation
├── test_artifacts.py            # Artifact store: storage, references
├── test_retry_idempotency.py    # Retry logic, idempotency
├── test_integration_abstractcore.py  # AbstractCore integration (mocked)
├── test_real_integration.py     # Real LLM tests with Ollama
└── README.md                    # This file
```

## Running Tests

```bash
# All unit tests (fast, no LLM required)
pytest --ignore=tests/test_real_integration.py

# Real integration tests (requires Ollama running)
pytest tests/test_real_integration.py -v

# All tests
pytest -v
```

## Test Categories

### 1. Core Runtime Tests (`test_runtime.py`)

Tests the fundamental tick/resume execution loop.

```
┌─────────────────────────────────────────────────────────────┐
│                         Runtime                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ start() │───►│ tick()  │───►│ resume()│───►│ tick()  │  │
│  └─────────┘    └────┬────┘    └────┬────┘    └────┬────┘  │
│                      │              │              │        │
│                      ▼              ▼              ▼        │
│                 ┌─────────┐   ┌─────────┐   ┌─────────┐    │
│                 │ RUNNING │   │ WAITING │   │COMPLETED│    │
│                 └─────────┘   └─────────┘   └─────────┘    │
└─────────────────────────────────────────────────────────────┘

Tests:
- test_simple_workflow_completes
- test_effectful_workflow
- test_wait_and_resume
- test_workflow_with_vars
```

### 2. Subworkflow Tests (`test_subworkflow.py`)

Tests workflow composition and parent-child relationships.

```
┌─────────────────────────────────────────────────────────────┐
│                    Parent Workflow                           │
│  ┌─────────┐    ┌──────────────────┐    ┌─────────┐        │
│  │ start   │───►│ START_SUBWORKFLOW│───►│  done   │        │
│  └─────────┘    └────────┬─────────┘    └─────────┘        │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                      │
│              │    Child Workflow     │                      │
│              │  ┌─────┐   ┌─────┐   │                      │
│              │  │start│──►│done │   │                      │
│              │  └─────┘   └─────┘   │                      │
│              └───────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘

Tests:
- test_sync_child_completes (child runs to completion)
- test_sync_child_waits (parent waits when child waits)
- test_async_parent_continues (fire-and-forget)
- test_nested_subworkflows (3 levels deep)
- test_cancel_with_children (cascading cancellation)
```

### 3. Artifact Tests (`test_artifacts.py`)

Tests large payload storage and retrieval.

```
┌─────────────────────────────────────────────────────────────┐
│                     Artifact Flow                            │
│                                                              │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │ Content │───►│ store_json() │───►│ ArtifactMetadata│    │
│  │ (large) │    └──────────────┘    │ {artifact_id}   │    │
│  └─────────┘                        └────────┬────────┘    │
│                                              │              │
│                                              ▼              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RunState.vars                           │   │
│  │  {"result": {"$artifact": "abc123..."}}             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                              │              │
│                                              ▼              │
│  ┌──────────────┐    ┌─────────┐                           │
│  │resolve_artifact│──►│ Content │                           │
│  └──────────────┘    └─────────┘                           │
└─────────────────────────────────────────────────────────────┘

Tests:
- test_store_and_load_bytes
- test_content_addressed_id (deduplication)
- test_store_json / test_load_json
- test_artifact_ref / test_resolve_artifact
- test_list_by_run (cleanup support)
```

### 4. Retry & Idempotency Tests (`test_retry_idempotency.py`)

Tests production reliability features.

```
┌─────────────────────────────────────────────────────────────┐
│                    Retry Flow                                │
│                                                              │
│  Attempt 1        Attempt 2        Attempt 3                │
│  ┌───────┐        ┌───────┐        ┌───────┐               │
│  │ FAIL  │──1s───►│ FAIL  │──2s───►│SUCCESS│               │
│  └───────┘        └───────┘        └───────┘               │
│      │                │                │                    │
│      ▼                ▼                ▼                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Ledger                            │   │
│  │  {attempt: 1, status: failed, idempotency_key: X}   │   │
│  │  {attempt: 2, status: failed, idempotency_key: X}   │   │
│  │  {attempt: 3, status: completed, idempotency_key: X}│   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 Idempotency Flow                             │
│                                                              │
│  First Run                    Restart (same run_id)         │
│  ┌───────────┐                ┌───────────┐                 │
│  │ LLM_CALL  │                │ LLM_CALL  │                 │
│  │ (execute) │                │ (skip!)   │                 │
│  └─────┬─────┘                └─────┬─────┘                 │
│        │                            │                        │
│        ▼                            ▼                        │
│  ┌───────────┐                ┌───────────┐                 │
│  │  Ledger   │                │  Ledger   │                 │
│  │ COMPLETED │───────────────►│ (lookup)  │                 │
│  └───────────┘                └───────────┘                 │
│                                     │                        │
│                                     ▼                        │
│                               Reuse prior result             │
└─────────────────────────────────────────────────────────────┘

Tests:
- test_no_retry_by_default
- test_retry_with_policy (3 attempts)
- test_exponential_backoff
- test_skip_reexecution_on_restart (idempotency)
- test_ledger_records_attempt_number
```

### 5. Real Integration Tests (`test_real_integration.py`)

Tests with actual LLM calls through AbstractCore + Ollama.

```
┌─────────────────────────────────────────────────────────────┐
│                  Integration Architecture                    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  AbstractRuntime                     │   │
│  │  ┌─────────┐    ┌─────────────┐    ┌─────────────┐  │   │
│  │  │ Runtime │───►│ LLM_CALL    │───►│ Effect      │  │   │
│  │  │ tick()  │    │ Effect      │    │ Handler     │  │   │
│  │  └─────────┘    └─────────────┘    └──────┬──────┘  │   │
│  └───────────────────────────────────────────┼──────────┘   │
│                                              │              │
│  ┌───────────────────────────────────────────┼──────────┐   │
│  │                  AbstractCore             │          │   │
│  │  ┌──────────────────────────────────────┐│          │   │
│  │  │     LocalAbstractCoreLLMClient       ││          │   │
│  │  │                                      ◄┘          │   │
│  │  │  create_llm(provider, model)         │           │   │
│  │  └──────────────────┬───────────────────┘           │   │
│  │                     │                                │   │
│  │  ┌──────────────────▼───────────────────┐           │   │
│  │  │         Ollama Provider              │           │   │
│  │  │   gemma3:1b-it-q4_K_M                │           │   │
│  │  └──────────────────────────────────────┘           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

Tests:
- test_simple_math_question (2+2=4)
- test_classification_task (sentiment analysis)
- test_json_extraction (structured output)
- test_think_then_answer (chain of thought)
- test_parent_delegates_to_specialist_child (subworkflow + LLM)
- test_store_large_llm_output_as_artifact (artifact + LLM)
- test_llm_then_wait_then_llm (wait/resume + LLM)
- test_simple_react_pattern (Reason → Act → Observe)
```

### 6. ReAct Agent Pattern Test

The `test_simple_react_pattern` demonstrates a complete agent loop:

```
┌─────────────────────────────────────────────────────────────┐
│                    ReAct Agent Workflow                      │
│                                                              │
│  Task: "What is the capital of France?"                     │
│                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ REASON  │───►│   ACT   │───►│ OBSERVE │───►│  DONE   │  │
│  └────┬────┘    └────┬────┘    └────┬────┘    └─────────┘  │
│       │              │              │                       │
│       ▼              ▼              ▼                       │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ LLM_CALL│    │ LLM_CALL│    │ LLM_CALL│                 │
│  │"Think   │    │"Provide │    │"Is this │                 │
│  │ about   │    │ answer" │    │ complete│                 │
│  │ task"   │    │         │    │ ?"      │                 │
│  └─────────┘    └─────────┘    └─────────┘                 │
│       │              │              │                       │
│       ▼              ▼              ▼                       │
│  reasoning:     action:        observation:                 │
│  "I need to     "Paris"        "COMPLETE"                  │
│  recall..."                                                 │
└─────────────────────────────────────────────────────────────┘
```

## Test Coverage by Component

| Component | Test File | Tests |
|-----------|-----------|-------|
| Runtime core | test_runtime.py | ~30 |
| Scheduler | test_scheduler.py | ~15 |
| Subworkflows | test_subworkflow.py | 18 |
| Artifacts | test_artifacts.py | 44 |
| Retry/Idempotency | test_retry_idempotency.py | 20 |
| AbstractCore Integration | test_integration_abstractcore.py | 4 |
| Real LLM Integration | test_real_integration.py | 11 |
| **Total** | | **~155** |

## Requirements

### Unit Tests
- Python 3.10+
- pytest

### Integration Tests
- Ollama running locally (`ollama serve`)
- Model: `gemma3:1b-it-q4_K_M`
- AbstractCore installed

## Adding New Tests

1. **Unit tests**: Add to appropriate `test_*.py` file
2. **Integration tests**: Add to `test_real_integration.py`
3. **New component**: Create new `test_<component>.py` file

Follow existing patterns:
- Use `WorkflowSpec` for workflow definitions
- Use `create_test_runtime()` for integration tests
- Assert on `RunStatus` and `state.output`
