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

    assert '"abstractsemantics>=0.0.3"' in text
    assert '"AbstractMemory>=0.2.6"' in text
    assert '"AbstractMemory[lancedb]' not in text
    assert "[project.optional-dependencies]" in text
    assert "abstractcore = [" in text
    assert "mcp-worker = [" in text
    assert '"abstractcore>=2.13.23"' in text
    assert '"abstractcore[remote,vision,voice,audio]>=2.13.23"' in text
    assert '"openai<2.0.0,>=1.109.1"' in text
    assert '"httpx<1.0.0,>=0.28.1"' in text
    assert '"anyio<5.0.0,>=4.12.1"' in text
    assert '"Pillow<13.0.0,>=10.0.0"' in text
    assert '"unstructured[docx,odt,pptx,rtf,xlsx]<0.19.0,>=0.18.32"' in text
    assert '"python-pptx<2.0.0,>=1.0.2"' in text

    worker_block = _extract_optional_dependency_block(text, key="mcp-worker")
    assert '"abstractcore[tools]>=2.13.23"' in worker_block


def test_runtime_exposes_python_install_hardware_profile_cascades() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    optional_keys = _extract_optional_dependency_keys(text)

    assert "apple" in optional_keys
    assert "gpu" in optional_keys
    assert "all-apple" in optional_keys
    assert "all-gpu" in optional_keys

    apple_block = _extract_optional_dependency_block(text, key="apple")
    gpu_block = _extract_optional_dependency_block(text, key="gpu")
    all_apple_block = _extract_optional_dependency_block(text, key="all-apple")
    all_gpu_block = _extract_optional_dependency_block(text, key="all-gpu")

    assert '"abstractcore[apple]>=2.13.23"' in apple_block
    assert '"abstractcore[gpu]>=2.13.23"' in gpu_block
    assert '"abstractcore[all-apple]>=2.13.23"' in all_apple_block
    assert '"abstractcore[all-gpu]>=2.13.23"' in all_gpu_block
    assert '"setuptools<82.0.0,>=80.10.2"' in apple_block
    assert '"setuptools<82.0.0,>=80.10.2"' in gpu_block
    assert '"setuptools<82.0.0,>=80.10.2"' in all_apple_block
    assert '"setuptools<82.0.0,>=80.10.2"' in all_gpu_block
