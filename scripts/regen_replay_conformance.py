"""Regenerate the replay conformance goldens (`expected_view.json`) from each fixture's inputs.

    python scripts/regen_replay_conformance.py            # report which goldens would change (no write)
    python scripts/regen_replay_conformance.py --write    # rewrite them (all cases, or the named ones)
    python scripts/regen_replay_conformance.py --write a_delegated_child_approval

A fixture is a directory holding `records.json` ({run_id: [ledger records in order, each with its
seq]}), `runs.json` ({run_id: run snapshot}), `manifest.json` (case, what it pins, provenance,
`root_run_id`) and `expected_view.json` (`replay(...).to_json()`, pretty, sorted keys).

Regenerate ONLY after reviewing the diff: a golden that changes is a change of the replay
contract every client renders. `tests/test_replay_conformance.py` also checks each case's meaning
by hand, so a golden regenerated over a bug still fails there.
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HERE = ROOT / "src" / "abstractruntime" / "replay" / "conformance"

from abstractruntime.replay import replay  # noqa: E402


def case_dirs():
    return sorted(p for p in HERE.iterdir() if p.is_dir() and (p / "manifest.json").exists())


def load_case(d: Path):
    records = json.loads((d / "records.json").read_text(encoding="utf-8"))
    runs = json.loads((d / "runs.json").read_text(encoding="utf-8"))
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    return records, runs, manifest


def render(view_json) -> str:
    return json.dumps(view_json, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def expected_text(d: Path) -> str:
    records, runs, manifest = load_case(d)
    return render(replay(records, runs, root_run_id=manifest["root_run_id"]).to_json())


def main(argv) -> int:
    write = "--write" in argv
    names = {a for a in argv if not a.startswith("--")}
    changed = 0
    for d in case_dirs():
        if names and d.name not in names:
            continue
        new = expected_text(d)
        path = d / "expected_view.json"
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if new == old:
            continue
        changed += 1
        diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), f"{d.name}/expected_view.json (old)",
                                         f"{d.name}/expected_view.json (new)", lineterm="", n=2))
        print("\n".join(diff[:200]) + ("\n..." if len(diff) > 200 else ""))
        if write:
            path.write_text(new, encoding="utf-8")
    print(f"{changed} golden(s) {'rewritten' if write else 'would change'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
