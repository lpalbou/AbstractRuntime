---
name: AbstractRuntime docs final
overview: Reconcile acore-runtime discussion, local backlog, and abstractruntime docs/backlog into a single, accurate, implementation-aligned documentation set in the abstractruntime repo (README + proposal + updated backlog items).
todos:
  - id: read-acore-runtime
    content: Re-read key sections of `acore-runtime.md` (definitions, modes, snapshots, provenance) and extract the final canonical statements to carry into docs.
    status: completed
  - id: reconcile-backlogs
    content: Compare `abstractcore/runtime/backlog/*` with `abstractruntime/docs/backlog/*` and reconcile discrepancies (file names, scope, statuses).
    status: completed
  - id: update-readme-proposal
    content: Update `abstractruntime/README.md` and `abstractruntime/docs/proposal.md` to be accurate and to include integration/snapshots/provenance.
    status: completed
  - id: update-existing-backlog
    content: Update `abstractruntime/docs/backlog/001-006_*.md` in place to match current implementation + refined designs.
    status: in_progress
  - id: add-snapshots-backlog
    content: Add `abstractruntime/docs/backlog/007_snapshots_and_bookmarks.md` as the missing slice.
    status: pending
---

# AbstractRuntime documentation finalization (source-of-truth)

## What I will produce (in `/Users/albou/projects/abstractruntime/`)

- Update **top-level docs** to match reality + final design:
- `README.md`
- `docs/proposal.md`
- Update **existing backlog docs in place** (your preference) so they reflect current code + the refined designs:
- `docs/backlog/001_runtime_kernel.md`
- `docs/backlog/002_persistence_and_ledger.md`
- `docs/backlog/003_wait_resume_and_scheduler.md`
- `docs/backlog/004_effect_handlers_and_integrations.md`
- `docs/backlog/005_examples_and_composition.md`
- `docs/backlog/006_ai_fingerprint_and_provenance.md`
- Add one missing backlog slice (because snapshots/bookmarks became a first-class requirement in the discussion and current plan):
- `docs/backlog/007_snapshots_and_bookmarks.md`

## Key design reconciliations I will enforce in docs

- **Layered coupling** (consistent with `acore-runtime.md`):
- `abstractruntime.core` stays dependency-light.
- `abstractruntime.integrations.abstractcore` is where AbstractCore imports live (LLM + tool execution + logging adapter).
- **Durability invariants**:
- `RunState.vars` must remain JSON-serializable; large payloads are future `ArtifactStore` references.
- Resume requires workflow spec + node handlers to be available.
- **Execution modes** (same workflow spec, different wiring):
- Local (in-process AbstractCore)
- Remote (HTTP to AbstractCore server)
- Hybrid (remote LLM + local tools)
- **Tool execution policy**:
- Remote/untrusted defaults to pass-through tool calls → WAITING + resume.
- Local/trusted can execute tools in-process.
- **Provenance**:
- v0.1: tamper-evident hash-chained ledger; signatures explicitly optional.
- Align `docs/backlog/006` with current implementation (`ActorFingerprint` is hash-based today) while preserving the longer-term signed mode.

## Concrete edits (file-by-file)

### `README.md`

- Remove “placeholder/no implementation” claim.
- Add minimal Quick Start (start → tick → waiting → resume).
- Add “How it fits” section: AbstractCore (inference plane) vs AbstractRuntime (durable runner) vs AbstractFlow/Agent.
- Link to `docs/proposal.md` and backlog items.

### `docs/proposal.md`

- Keep the current proposal core.
- Add/expand sections for:
- Integration module location and local/remote/hybrid wiring.
- Snapshots/bookmarks (and restore semantics).
- Provenance: hash chain now, signatures optional later.
- Operational model: scheduler/event-ingress as external processes (no Temporal clone).

### `docs/backlog/001_runtime_kernel.md`

- Update deliverable file names to match the repo (`core/models.py`, `core/spec.py`, `core/runtime.py`).
- Add “Status: implemented” and point to tests.

### `docs/backlog/002_persistence_and_ledger.md`

- Update deliverables to match current storage modules (`storage/base.py`, `storage/in_memory.py`, `storage/json_files.py`).
- Keep `ArtifactStore` as planned (explicitly “not implemented yet”).
- Cross-link snapshots backlog `007`.

### `docs/backlog/003_wait_resume_and_scheduler.md`

- Align with what exists today (wait_event, wait_until, ask_user are implemented in `core/runtime.py`).
- Keep scheduler as planned: document minimal worker loop patterns + why it’s intentionally out-of-scope for v0.1.

### `docs/backlog/004_effect_handlers_and_integrations.md`

- Expand to include:
- Stable effect payload schemas (`llm_call`, `tool_calls`).
- Concrete integration module file layout.
- Remote client request seam (testable without live HTTP).
- Optional forwarding of AbstractCore server per-request `base_url` (since AbstractCore supports it for openai-compatible routing).

### `docs/backlog/005_examples_and_composition.md`

- Specify a minimal set of example workflows to add later (ask_user, wait_until, composition/subworkflow).
- Keep “wait_job” explicitly as a planned placeholder.

### `docs/backlog/006_ai_fingerprint_and_provenance.md`

- Reconcile the doc with current state:
- ActorFingerprint is hash-based now.
- Add hash-chained ledger as the immediate primitive.
- Keep Ed25519/signatures as an optional extra and document key-management open questions.

### `docs/backlog/007_snapshots_and_bookmarks.md` (new)

- Add the snapshot model + store interface + JSON backend + basic search semantics.

## Validation / success criteria

- Docs do not contradict the repository state (README no longer says “no implementation”).
- Backlog items reflect current file names and actual implemented capabilities.
- New doc set matches the refined design in `acore-runtime.md` and the local `abstractcore/runtime/backlog/*`.

## Contingency (if writes to `/Users/albou/projects/abstractruntime/` are blocked again)

- I will still produce the exact final markdown content inside this workspace (e.g. under `runtime/backlog/final_for_abstractruntime/`) and provide a single `git apply`-style patch file so you can apply it from your shell in the abstractruntime repo.