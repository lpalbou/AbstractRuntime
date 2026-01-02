from __future__ import annotations

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


def test_runtime_exposes_abstractcore_and_worker_extras() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert "[project.optional-dependencies]" in text
    assert "abstractcore = [" in text
    assert "mcp-worker = [" in text

    worker_block = _extract_optional_dependency_block(text, key="mcp-worker")
    assert "abstractcore[tools]" in worker_block

