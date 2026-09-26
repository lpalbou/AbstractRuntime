#!/usr/bin/env python3
"""Regenerate llms-full.txt: the documentation pages below, concatenated in full.

Run from anywhere (paths resolve from this file):

    python scripts/generate_llms_full.py          # rewrite llms-full.txt
    python scripts/generate_llms_full.py --check  # exit 1 when llms-full.txt is stale

The manifest is explicit rather than a glob, and the output depends only on
the listed files (no timestamps, UTF-8, LF line endings), so it is
byte-reproducible. A missing page fails loudly. `llms.txt` is the curated
index and is edited by hand. Release history (`CHANGELOG.md`), ADR bodies and
backlog items are left out: they are linked from `llms.txt` and `docs/README.md`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "llms-full.txt"

DOCUMENTS = [
    "README.md",
    "docs/getting-started.md",
    "docs/architecture.md",
    "docs/api.md",
    "docs/faq.md",
    "docs/troubleshooting.md",
    "docs/integrations/abstractcore.md",
    "docs/proposal.md",
    "docs/limits.md",
    "docs/artifacts.md",
    "docs/tool-approval.md",
    "docs/tools-comms.md",
    "docs/entity-runtime.md",
    "docs/mcp-worker.md",
    "docs/evidence.md",
    "docs/snapshots.md",
    "docs/provenance.md",
    "docs/workflow-bundles.md",
    "docs/manual_testing.md",
    "docs/adr/README.md",
    "examples/README.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "ACKNOWLEDGMENTS.md",
    "ROADMAP.md",
    "docs/README.md",
]

HEADER = """# AbstractRuntime — full documentation

> Durable workflow runtime for AbstractFramework: workflows run as a persisted
> state machine (interrupt, checkpoint, resume) with explicit waits and an
> append-only execution ledger. The AbstractCore integration adds LLM calls,
> tools, live token streaming, model residency and media generation.

This file concatenates the documentation pages listed below, in full. It
contains no source code; the repository's code and tests are the source of
truth. Relative links inside each page are relative to that page's own path.
See llms.txt for the linked index.

## Document index

{index}
"""


def render() -> str:
    missing = [name for name in DOCUMENTS if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("missing documents: " + ", ".join(missing))
    index = "\n".join(f"- {name}" for name in DOCUMENTS)
    parts = [HEADER.format(index=index)]
    for name in DOCUMENTS:
        text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        parts.append(f"\n\n{'=' * 78}\n# FILE: {name}\n{'=' * 78}\n\n")
        parts.append(text.rstrip() + "\n")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail when llms-full.txt is stale")
    args = parser.parse_args()
    expected = render()
    if args.check:
        current = OUTPUT.read_bytes().decode("utf-8") if OUTPUT.is_file() else None
        if current != expected:
            print("llms-full.txt is stale; run python scripts/generate_llms_full.py", file=sys.stderr)
            return 1
        print("llms-full.txt is current")
        return 0
    OUTPUT.write_bytes(expected.encode("utf-8"))
    print(f"wrote {OUTPUT.name} from {len(DOCUMENTS)} documents ({OUTPUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
