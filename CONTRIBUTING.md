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

# Full dev install (runtime + docs/test tooling)
python -m pip install -e ".[test,docs]"

python -m pytest -q
```

Inside the AbstractFramework workspace, prefer `python -P -m pytest tests -q`: `-P` keeps the current folder off
`sys.path`, so a workspace folder named like a package (for example `abstractflow/`) cannot shadow it.

If you cloned **only** this repo (without the AbstractFramework workspace), make sure the sibling packages above are importable (install them or clone them next to this repo) before running the full test suite.

### Live email tests (opt-in)

`tests/live_email/` runs the email integration against a **real test mailbox** (framework backlog 0992 WP2): an
`email.received@1` automation admits one uniquely tagged self-sent message exactly once (not again on a second poll,
a later wake or a restarted runtime), and a `send_email` tool call to the account's own address runs unattended while
one to another address parks on a `tool_approval` wait and is never sent. The tests carry the `live_email` marker, are
not `basic` (CI never selects them), and are skipped unless every variable below is set.

The test harness reads the mailbox from environment variables (the tests' input only; the runtime still gets its
account from the host resolver): `AF_TEST_EMAIL_IMAP_HOST`, `AF_TEST_EMAIL_IMAP_PORT`, `AF_TEST_EMAIL_IMAP_SECURITY`,
`AF_TEST_EMAIL_SMTP_HOST`, `AF_TEST_EMAIL_SMTP_PORT`, `AF_TEST_EMAIL_SMTP_SECURITY`, `AF_TEST_EMAIL_USERNAME`,
`AF_TEST_EMAIL_ADDRESS`, `AF_TEST_EMAIL_PASSWORD`. Keep them in a private file (mode 0600) and load it into the test
process only:

```bash
set -a; . ~/.config/abstractframework-test/email.env; set +a; \
  python -P -m pytest tests/live_email -m live_email -s --durations=0
```

No credential value is printed (redacting objects, scrubbed failure reports), mail only goes to the test account's
own address (a guard refuses any other recipient before `MAIL FROM`), and messages are small and tagged
`[af-live-test]`: the mailbox is read-only, so nothing can be cleaned up. Use a dedicated test mailbox. AbstractCore's
`tests/email/live/` covers the mail library itself.

## Repo map (source of truth)

- Public exports: `src/abstractruntime/__init__.py` (keep this consistent with `docs/api.md`)
- Core kernel (durable semantics): `src/abstractruntime/core/`
- Durability backends: `src/abstractruntime/storage/`
- Driver loop (in-process): `src/abstractruntime/scheduler/`
- Runtime integrations: `src/abstractruntime/integrations/`
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

List every `docs/*.md` page in `docs/README.md`. Keep `llms.txt` (the hand-curated index) and
`llms-full.txt` in step with the documentation in the same change. `llms-full.txt` is generated:
run `python scripts/generate_llms_full.py` after editing any page it includes, and
`python scripts/generate_llms_full.py --check` to confirm it is current (it exits 1 when the file
is stale). Add a page to the script's `DOCUMENTS` list when it joins the core set.

## Releases

- Bump `version` in `pyproject.toml`
- Add a dated section to `CHANGELOG.md` (Keep a Changelog format)
