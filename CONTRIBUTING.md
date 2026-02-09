# Contributing to AbstractRuntime

Thanks for your interest in contributing!

AbstractRuntime is a **durable workflow runtime** (interrupt → checkpoint → resume) with an append-only execution ledger.

## Quick start (dev setup)

Prereqs: **Python 3.10+**.

Recommended (workspace checkout): develop inside the [AbstractFramework](https://github.com/lpalbou/AbstractFramework) workspace.  
The test bootstrap (`tests/conftest.py`) will auto-wire sibling projects on `sys.path` (e.g., `abstractcore/`, `abstractmemory/`, `abstractsemantics/`, `abstractflow/`).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# Full dev install (kernel + optional integrations used by tests/examples)
python -m pip install -e ".[abstractcore,mcp-worker]"

python -m pytest -q
```

If you cloned **only** this repo (without the AbstractFramework workspace), make sure the sibling packages above are importable (install them or clone them next to this repo) before running the full test suite.

## Repo map (source of truth)

- Public exports: `src/abstractruntime/__init__.py` (keep this consistent with `docs/api.md`)
- Core kernel (durable semantics): `src/abstractruntime/core/`
- Durability backends: `src/abstractruntime/storage/`
- Driver loop (in-process): `src/abstractruntime/scheduler/`
- Optional integrations (extras): `src/abstractruntime/integrations/`
- Tests: `tests/`

Docs entrypoints:
- `README.md` → `docs/getting-started.md`
- Docs index: `docs/README.md`
- Architecture: `docs/architecture.md`

## Change guidelines

### Code

- Preserve durability invariants: values stored in `RunState.vars` must stay JSON-serializable (`src/abstractruntime/core/models.py`).
- Add/adjust tests for new behavior (see `tests/`).
- If you touch effect semantics, update `docs/architecture.md` and ensure handlers and models stay aligned.

### Documentation

Docs should be **user-facing**, **actionable**, and anchored to code (prefer referencing `src/...` paths for claims).

When behavior changes, update:
- `docs/api.md` (public API surface + imports)
- `docs/getting-started.md` (onboarding examples)
- `docs/architecture.md` (semantics/invariants)
- `CHANGELOG.md` (user-visible changes)

## Releases

- Bump `version` in `pyproject.toml`
- Add a dated section to `CHANGELOG.md` (Keep a Changelog format)
