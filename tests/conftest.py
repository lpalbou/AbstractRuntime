"""Test configuration for the src/ layout.

This repository contains an `abstractruntime/` sub-project using the common
`src/` package layout. When tests are invoked from the monorepo root, Python's
import path may resolve an installed `abstractruntime` package instead.

We explicitly prioritize the local `abstractruntime/src` directory to ensure
we test the code in this repository.
"""

from __future__ import annotations

import sys
from pathlib import Path


_SRC = (Path(__file__).resolve().parents[1] / "src").as_posix()
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

