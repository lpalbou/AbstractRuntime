"""AbstractRuntime test bootstrap for monorepo layouts.

Why this exists:
- In this repo, sibling projects live at the monorepo root (e.g. `abstractcore/`,
  `abstractruntime/`, ...).
- When tests are invoked from the monorepo root, Python's default `sys.path`
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
MONOREPO_ROOT = HERE.parents[2]         # .../abstractframework

# Ensure `abstractcore` resolves to .../abstractcore/abstractcore (has __init__.py)
_prepend_sys_path(MONOREPO_ROOT / "abstractcore")

# Ensure `abstractruntime` resolves to .../abstractruntime/src/abstractruntime (src-layout)
_prepend_sys_path(ABSTRACTRUNTIME_ROOT / "src")

# Keep sibling src-layout packages stable as well.
_prepend_sys_path(MONOREPO_ROOT / "abstractagent" / "src")
