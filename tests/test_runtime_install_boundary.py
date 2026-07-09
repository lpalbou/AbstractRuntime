from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
KERNEL_DIRS = (
    SRC / "abstractruntime" / "core",
    SRC / "abstractruntime" / "storage",
    SRC / "abstractruntime" / "scheduler",
    SRC / "abstractruntime" / "evidence",
    SRC / "abstractruntime" / "identity",
    SRC / "abstractruntime" / "rendering",
    SRC / "abstractruntime" / "workflow_bundle",
)
OPTIONAL_STACK_ROOTS = {"abstractcore", "abstractvision", "abstractvoice", "abstractmemory", "abstractmusic"}


def _import_root(module: str) -> str:
    return module.split(".", 1)[0]


def test_runtime_kernel_does_not_import_optional_capability_stacks() -> None:
    """MODULE-LEVEL optional-stack imports are forbidden in kernel dirs.

    Contract ruling (landing audit, 2026-07-07): the boundary's INTENT is
    import-time cleanliness — `import abstractruntime` must never pull the
    optional stacks (the companion subprocess test enforces exactly that).
    FUNCTION-LEVEL (lazy) imports are the sanctioned mechanism for kernel
    features that USE optional stacks when present (the identity drivers:
    chat/life/tools/digest) — they are deliberate, labeled, and only run
    when the operator invokes those features. The static scan therefore
    checks module scope only: an optional import that executes at import
    time is a violation; one inside a function body is the design.
    """
    violations: list[str] = []
    for base in KERNEL_DIRS:
        for path in sorted(base.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            # Collect module-level statements only (direct children of the
            # Module node, plus module-level try/if blocks — the common
            # guarded-import patterns that still execute at import time).
            def _module_level_nodes(root: ast.Module):
                stack = list(root.body)
                while stack:
                    stmt = stack.pop()
                    yield stmt
                    if isinstance(stmt, (ast.Try, ast.If, ast.With)):
                        for field in ("body", "orelse", "finalbody", "handlers"):
                            for child in getattr(stmt, field, []) or []:
                                if isinstance(child, ast.ExceptHandler):
                                    stack.extend(child.body)
                                else:
                                    stack.append(child)

            for node in _module_level_nodes(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if _import_root(alias.name) in OPTIONAL_STACK_ROOTS:
                            violations.append(f"{path.relative_to(ROOT)} imports {alias.name} at module level")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if _import_root(node.module) in OPTIONAL_STACK_ROOTS:
                        violations.append(f"{path.relative_to(ROOT)} imports {node.module} at module level")

    assert violations == []


def test_package_root_import_does_not_touch_optional_capability_stacks() -> None:
    script = f"""
import importlib.abc
import sys

sys.path.insert(0, {str(SRC)!r})

class OptionalStackBlocker(importlib.abc.MetaPathFinder):
    blocked = {sorted(OPTIONAL_STACK_ROOTS)!r}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise AssertionError(f"unexpected optional import: {{fullname}}")
        return None

sys.meta_path.insert(0, OptionalStackBlocker())
import abstractruntime
print(abstractruntime.Runtime.__name__)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "Runtime"
