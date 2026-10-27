"""The replay conformance cases: ledgers with the view every implementation must produce.

Each case is a directory holding these files:

- `records.json`: `{run_id: [ledger records in order, each with its seq]}`.
- `runs.json`: `{run_id: run snapshot}`.
- `manifest.json`: the case, what it pins, its provenance and `root_run_id`.
- `expected_view.json`: `replay(...).to_json()`.

The cases are redacted ledgers from a real gateway plus synthetic ones. They ship with the
package, so a host or a client can run them without the runtime's repository. The gateway serves
them through `/view` in its own tests. A client port of the reducer, or of `apply_delta`, can
check itself against them (ADR 0009).

    from abstractruntime.replay.conformance import case_names, load_case
    for name in case_names():
        case = load_case(name)
        # case["records"], case["runs"], case["manifest"], case["expected_view"]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent


def case_names() -> List[str]:
    """Every case, sorted by name."""
    return sorted(p.name for p in HERE.iterdir() if p.is_dir() and (p / "manifest.json").exists())


def case_dir(name: str) -> Path:
    d = HERE / name
    if not (d / "manifest.json").exists():
        raise KeyError(f"no replay conformance case named {name!r}")
    return d


def load_case(name: str) -> Dict[str, Any]:
    """`{records, runs, manifest, expected_view}` of one case."""
    d = case_dir(name)
    return {
        key: json.loads((d / f"{key}.json").read_text(encoding="utf-8"))
        for key in ("records", "runs", "manifest", "expected_view")
    }


__all__ = ["case_names", "case_dir", "load_case"]
