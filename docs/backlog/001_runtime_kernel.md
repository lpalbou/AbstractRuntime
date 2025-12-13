# 001_runtime_kernel

> Legacy note: preserved for history. See `docs/backlog/completed/001_runtime_kernel.md`.

## Goal
Implement the minimal AbstractRuntime kernel that can execute a workflow graph until:
- completion
- failure
- or an interrupt (waiting state)

## Scope
- Workflow spec representation (nodes + transitions)
- Run state representation
- Effect model
- Runtime loop: `start`, `tick`, `resume`

## Non-goals
- persistence (RunStore/LedgerStore) beyond in-memory
- distributed workers
- UI

## Deliverables
- `abstractruntime/core/spec.py`
- `abstractruntime/core/state.py`
- `abstractruntime/core/effects.py`
- `abstractruntime/core/runtime.py`

## Acceptance criteria
- A workflow can:
  - run deterministic nodes
  - emit `wait_event` and pause
  - resume with an event payload and continue

## Notes
Keep the kernel dependency-light and avoid coupling to AbstractCore.


