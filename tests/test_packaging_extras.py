from __future__ import annotations

import re
from pathlib import Path

import pytest


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


def test_runtime_base_is_remote_light_with_multimodal_and_mcp_support() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert '"abstractsemantics>=0.0.5"' in text
    assert '"AbstractMemory>=0.3.0"' in text
    assert '"AbstractMemory[lancedb]' not in text
    assert "[project.optional-dependencies]" in text
    assert "abstractcore = [" not in text
    assert "multimodal = [" not in text
    assert "mcp-worker = [" not in text
    assert '"abstractcore>=2.25.1"' in text
    # Only AbstractCore's three install settings: light (no extra), apple, gpu.
    assert re.findall(r'"abstractcore\[([^\]]*)\]', text) == ["apple", "gpu"]
    assert '"openai<3.0.0,>=1.109.1"' in text
    assert '"httpx<1.0.0,>=0.28.1"' in text
    assert '"anyio<5.0.0,>=4.12.1"' in text
    assert '"Pillow<13.0.0,>=10.0.0"' in text
    assert '"pypdf<7.0.0,>=6.0.0"' in text
    assert '"reportlab<5.0.0,>=4.0.0"' in text
    assert '"unstructured[docx,odt,pptx,rtf,xlsx]<0.19.0,>=0.18.32"' in text
    assert '"python-pptx<2.0.0,>=1.0.2"' in text
    assert "pymupdf" not in text.lower()
    assert "pymupdf4llm" not in text
    assert "pymupdf-layout" not in text
    assert '"torch' not in text
    assert '"sentence-transformers' not in text
    assert '"vllm' not in text
    assert '"mlx' not in text


def test_runtime_exposes_only_apple_and_gpu_user_install_profiles() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    optional_keys = _extract_optional_dependency_keys(text)

    assert "apple" in optional_keys
    assert "gpu" in optional_keys
    assert "all-apple" not in optional_keys
    assert "all-gpu" not in optional_keys
    assert "abstractcore" not in optional_keys
    assert "multimodal" not in optional_keys
    assert "mcp-worker" not in optional_keys

    apple_block = _extract_optional_dependency_block(text, key="apple")
    gpu_block = _extract_optional_dependency_block(text, key="gpu")

    assert '"abstractcore[apple]>=2.25.1"' in apple_block
    assert '"abstractcore[gpu]>=2.25.1"' in gpu_block
    assert '"setuptools<82.0.0,>=80.10.2"' in apple_block
    assert '"setuptools<82.0.0,>=77.0.3"' in gpu_block
    assert "pymupdf" not in apple_block.lower()
    assert "pymupdf" not in gpu_block.lower()


def test_sibling_abstractcore_profiles_keep_pypdf_and_exclude_pymupdf() -> None:
    """Workspace-only check: the AbstractCore profiles Runtime's apple/gpu extras select
    install the permissive pypdf reader and never PyMuPDF. Skipped outside the monorepo."""

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    core_pyproject = pyproject.resolve().parents[1] / "abstractcore" / "pyproject.toml"
    if not core_pyproject.exists():
        pytest.skip("no sibling abstractcore checkout")
    tomllib = pytest.importorskip("tomllib")
    core = tomllib.loads(core_pyproject.read_text(encoding="utf-8"))["project"]
    extras = core["optional-dependencies"]

    def resolve(profile: str, seen: tuple[str, ...] = ()) -> list[str]:
        # Follow self-referencing aliases such as all-apple = ["abstractcore[apple]"].
        requirements: list[str] = []
        for req in extras[profile]:
            match = re.fullmatch(r"abstractcore\[([^\]]+)\]", req.strip())
            if match:
                for inner in match.group(1).split(","):
                    inner = inner.strip()
                    assert inner not in seen, f"alias cycle through {inner}"
                    requirements += resolve(inner, seen + (profile,))
            else:
                requirements.append(req)
        return requirements

    base = core["dependencies"]
    for core_profile in ("apple", "gpu"):
        installed = base + resolve(core_profile)
        assert any(req.replace(" ", "").startswith("pypdf>=6.0.0") for req in installed), core_profile
        assert not any("pymupdf" in req.lower() for req in installed), core_profile
