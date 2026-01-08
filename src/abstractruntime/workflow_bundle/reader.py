"""WorkflowBundle reader (directory or .flow zip).

This module focuses on *reading* bundles. Writing/packing bundles is expected to
be performed by authoring tooling (e.g., AbstractFlow).
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .models import WorkflowBundleError, WorkflowBundleManifest, workflow_bundle_manifest_from_dict


@dataclass(frozen=True)
class WorkflowBundle:
    """An opened WorkflowBundle source."""

    source: Path
    manifest: WorkflowBundleManifest

    def _is_zip(self) -> bool:
        return self.source.is_file()

    def read_bytes(self, relpath: str) -> bytes:
        p = str(relpath or "").strip()
        if not p:
            raise WorkflowBundleError("read_bytes requires a non-empty relpath")

        if self._is_zip():
            with zipfile.ZipFile(self.source, "r") as zf:
                try:
                    return zf.read(p)
                except KeyError as e:
                    raise FileNotFoundError(f"Bundle file not found: {p}") from e
        else:
            abs_p = (self.source / p).resolve()
            # Ensure relpath cannot escape the bundle directory.
            try:
                abs_p.relative_to(self.source.resolve())
            except Exception as e:
                raise WorkflowBundleError(f"Unsafe relpath outside bundle dir: {p}") from e
            if not abs_p.exists():
                raise FileNotFoundError(f"Bundle file not found: {p}")
            return abs_p.read_bytes()

    def read_text(self, relpath: str, *, encoding: str = "utf-8") -> str:
        return self.read_bytes(relpath).decode(encoding)

    def read_json(self, relpath: str) -> Any:
        return json.loads(self.read_text(relpath))


def _read_manifest_from_dir(dir_path: Path) -> WorkflowBundleManifest:
    p = (dir_path / "manifest.json").resolve()
    if not p.exists():
        raise FileNotFoundError(f"manifest.json not found in bundle dir: {dir_path}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return workflow_bundle_manifest_from_dict(raw)


def _read_manifest_from_zip(zip_path: Path) -> WorkflowBundleManifest:
    with zipfile.ZipFile(zip_path, "r") as zf:
        try:
            raw_bytes = zf.read("manifest.json")
        except KeyError as e:
            raise FileNotFoundError(f"manifest.json not found in bundle: {zip_path}") from e
    raw = json.loads(raw_bytes.decode("utf-8"))
    return workflow_bundle_manifest_from_dict(raw)


def open_workflow_bundle(source: str | Path) -> WorkflowBundle:
    """Open a WorkflowBundle from a directory or a `.flow` zip file."""
    p = Path(source).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Bundle source not found: {p}")
    if p.is_dir():
        manifest = _read_manifest_from_dir(p)
        return WorkflowBundle(source=p, manifest=manifest)
    # File: treat as zip bundle.
    manifest = _read_manifest_from_zip(p)
    return WorkflowBundle(source=p, manifest=manifest)


