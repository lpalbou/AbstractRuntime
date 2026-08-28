"""AbstractRuntime test bootstrap for a multi-repo workspace checkout.

Why this exists:
- In this workspace, sibling projects live under a shared parent directory
  (e.g. `abstractcore/`, `abstractruntime/`, ...).
- When tests are invoked from the workspace root, Python's default `sys.path`
  includes the CWD (""), which makes directories like `abstractcore/` appear as
  namespace packages (PEP 420) and *shadow* the actual installable package
  located at `abstractcore/abstractcore/`.

This breaks imports for the AbstractRuntime↔AbstractCore integration, e.g.:
`from abstractcore import create_llm`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _prepend_sys_path(path: Path) -> None:
    p = str(path)
    if p and p not in sys.path:
        sys.path.insert(0, p)


HERE = Path(__file__).resolve()
ABSTRACTRUNTIME_ROOT = HERE.parents[1]  # .../abstractruntime
MONOREPO_ROOT = HERE.parents[2]  # .../abstractframework

# Ensure `abstractcore` resolves to .../abstractcore/abstractcore (has __init__.py)
_prepend_sys_path(MONOREPO_ROOT / "abstractcore")

# Ensure `abstractflow` resolves to .../abstractflow/abstractflow (has __init__.py)
_prepend_sys_path(MONOREPO_ROOT / "abstractflow")

# Ensure `abstractruntime` resolves to .../abstractruntime/src/abstractruntime (src-layout)
_prepend_sys_path(ABSTRACTRUNTIME_ROOT / "src")

# Keep sibling src-layout packages stable as well.
_prepend_sys_path(MONOREPO_ROOT / "abstractagent" / "src")
_prepend_sys_path(MONOREPO_ROOT / "abstractmemory" / "src")
_prepend_sys_path(MONOREPO_ROOT / "abstractsemantics" / "src")


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _hermetic_host_residency_sweep(monkeypatch: pytest.MonkeyPatch):
    """Local residency listings merge core's live model-server sweep. This
    host may run real local servers (Ollama/LM Studio), whose resident models
    would leak into hermetic residency assertions. Sweep-merge tests
    monkeypatch their own fakes on top."""
    from abstractruntime.integrations.abstractcore import llm_client

    monkeypatch.setattr(llm_client, "_sweep_host_loaded_models", lambda: [])
    yield
