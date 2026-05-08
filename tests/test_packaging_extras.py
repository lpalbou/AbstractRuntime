from __future__ import annotations

import re
from pathlib import Path


def _extract_optional_dependency_block(text: str, *, key: str) -> str:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"{key} = ["):
            start = i
            break
    assert start is not None, f"Missing optional-dependencies entry: {key}"

    block: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() == "]":
            break
        block.append(line)
    return "\n".join(block)


def _extract_optional_dependency_keys(text: str) -> set[str]:
    in_section = False
    keys: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[project.optional-dependencies]":
            in_section = True
            continue
        if in_section and stripped.startswith("[") and stripped.endswith("]"):
            break
        if not in_section:
            continue
        match = re.match(r"^([A-Za-z0-9_-]+)\s*=\s*\[", line)
        if match:
            keys.add(match.group(1))
    return keys


def test_runtime_exposes_abstractcore_and_worker_extras_with_gateway_aligned_floor() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert "[project.optional-dependencies]" in text
    assert "abstractcore = [" in text
    assert "mcp-worker = [" in text
    assert '"abstractcore>=2.13.11"' in text
    assert '"abstractcore[media,openai,vision,voice,audio]>=2.13.11"' in text

    worker_block = _extract_optional_dependency_block(text, key="mcp-worker")
    assert '"abstractcore[tools]>=2.13.11"' in worker_block


def test_runtime_does_not_expose_hardware_profile_extras() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    optional_keys = _extract_optional_dependency_keys(text)

    assert "apple" not in optional_keys
    assert "gpu" not in optional_keys
    assert "all-apple" not in optional_keys
    assert "all-gpu" not in optional_keys
