"""WorkflowBundle manifest models (stdlib-only).

Bundle layout (zip):
  manifest.json
  flows/<id>.json            (VisualFlow JSON sources)
  assets/...                 (optional)

Notes:
- A bundle is a *distribution unit*. Hosts may namespace workflow ids at load time.
- `manifest.artifacts` remains as a legacy field for backward compatibility, but modern
  hosts compile from `manifest.flows` using the AbstractRuntime VisualFlow compiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


WORKFLOW_BUNDLE_FORMAT_VERSION_V1 = "1"


class WorkflowBundleError(ValueError):
    """Raised when a WorkflowBundle manifest is invalid or unsupported."""


def _is_safe_relpath(p: str) -> bool:
    s = str(p or "").strip()
    if not s:
        return False
    # Disallow absolute paths and traversal.
    if s.startswith(("/", "\\")):
        return False
    if ":" in s.split("/", 1)[0]:
        # Prevent "C:\..." style and other drive-like prefixes.
        return False
    parts = [x for x in s.replace("\\", "/").split("/") if x]
    if any(x in {".", ".."} for x in parts):
        return False
    return True


@dataclass(frozen=True)
class WorkflowBundleEntrypoint:
    """An entrypoint workflow exposed by a bundle."""

    flow_id: str
    name: Optional[str] = None
    description: str = ""
    interfaces: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowBundleManifest:
    """Bundle manifest (manifest.json)."""

    bundle_format_version: str
    bundle_id: str
    bundle_version: str = "0.0.0"
    created_at: str = ""

    entrypoints: List[WorkflowBundleEntrypoint] = field(default_factory=list)

    # Maps flow_id -> relative path in the bundle (optional; used by UIs).
    flows: Dict[str, str] = field(default_factory=dict)
    # Legacy/deprecated: maps flow_id -> relative path to WorkflowArtifact JSON.
    artifacts: Dict[str, str] = field(default_factory=dict)
    # Optional assets mapping (logical name -> relative path).
    assets: Dict[str, str] = field(default_factory=dict)

    # Free-form JSON-safe metadata (tags, author, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if str(self.bundle_format_version or "").strip() != WORKFLOW_BUNDLE_FORMAT_VERSION_V1:
            raise WorkflowBundleError(
                f"Unsupported WorkflowBundle bundle_format_version '{self.bundle_format_version}'. "
                f"Supported: {WORKFLOW_BUNDLE_FORMAT_VERSION_V1}"
            )
        if not isinstance(self.bundle_id, str) or not self.bundle_id.strip():
            raise WorkflowBundleError("bundle_id must be a non-empty string")
        if not isinstance(self.bundle_version, str) or not self.bundle_version.strip():
            raise WorkflowBundleError("bundle_version must be a non-empty string")

        if not isinstance(self.entrypoints, list) or not self.entrypoints:
            raise WorkflowBundleError("manifest.entrypoints must be a non-empty list")

        seen: set[str] = set()
        for ep in self.entrypoints:
            if not isinstance(ep.flow_id, str) or not ep.flow_id.strip():
                raise WorkflowBundleError("entrypoint.flow_id must be a non-empty string")
            fid = ep.flow_id.strip()
            if fid in seen:
                raise WorkflowBundleError(f"Duplicate entrypoint flow_id '{fid}'")
            seen.add(fid)
            if not isinstance(ep.description, str):
                raise WorkflowBundleError(f"entrypoint '{fid}' description must be a string")
            if not isinstance(ep.interfaces, list):
                raise WorkflowBundleError(f"entrypoint '{fid}' interfaces must be a list")
            for it in ep.interfaces:
                if not isinstance(it, str):
                    raise WorkflowBundleError(f"entrypoint '{fid}' interfaces must be strings")

        for mapping_name, mapping in (("flows", self.flows), ("artifacts", self.artifacts), ("assets", self.assets)):
            if not isinstance(mapping, dict):
                raise WorkflowBundleError(f"manifest.{mapping_name} must be a dict")
            for k, v in mapping.items():
                if not isinstance(k, str) or not k.strip():
                    raise WorkflowBundleError(f"manifest.{mapping_name} keys must be non-empty strings")
                if not isinstance(v, str) or not _is_safe_relpath(v):
                    raise WorkflowBundleError(f"manifest.{mapping_name} entry '{k}' has unsafe path '{v}'")

        if not isinstance(self.metadata, dict):
            raise WorkflowBundleError("manifest.metadata must be an object")

    def artifact_path_for(self, flow_id: str) -> Optional[str]:
        fid = str(flow_id or "").strip()
        if not fid:
            return None
        p = self.artifacts.get(fid)
        return str(p) if isinstance(p, str) and p.strip() else None

    def flow_path_for(self, flow_id: str) -> Optional[str]:
        fid = str(flow_id or "").strip()
        if not fid:
            return None
        p = self.flows.get(fid)
        return str(p) if isinstance(p, str) and p.strip() else None


def workflow_bundle_manifest_from_dict(raw: Dict[str, Any]) -> WorkflowBundleManifest:
    if not isinstance(raw, dict):
        raise WorkflowBundleError("manifest.json must be a JSON object")

    version = str(raw.get("bundle_format_version") or raw.get("bundleFormatVersion") or "").strip()
    bundle_id = str(raw.get("bundle_id") or raw.get("bundleId") or "").strip()
    bundle_version = str(raw.get("bundle_version") or raw.get("bundleVersion") or "0.0.0").strip() or "0.0.0"
    created_at = str(raw.get("created_at") or raw.get("createdAt") or "")

    entrypoints_raw = raw.get("entrypoints")
    entrypoints: list[WorkflowBundleEntrypoint] = []
    if isinstance(entrypoints_raw, list):
        for ep in entrypoints_raw:
            if not isinstance(ep, dict):
                continue
            flow_id = str(ep.get("flow_id") or ep.get("flowId") or "").strip()
            name = ep.get("name")
            name_s = str(name).strip() if isinstance(name, str) and name.strip() else None
            description = str(ep.get("description") or "")
            interfaces_raw = ep.get("interfaces")
            interfaces: list[str] = []
            if isinstance(interfaces_raw, list):
                for x in interfaces_raw:
                    if isinstance(x, str) and x.strip():
                        interfaces.append(x.strip())
            entrypoints.append(
                WorkflowBundleEntrypoint(
                    flow_id=flow_id,
                    name=name_s,
                    description=description,
                    interfaces=interfaces,
                )
            )

    flows_raw = raw.get("flows")
    flows: Dict[str, str] = dict(flows_raw) if isinstance(flows_raw, dict) else {}

    artifacts_raw = raw.get("artifacts")
    artifacts: Dict[str, str] = dict(artifacts_raw) if isinstance(artifacts_raw, dict) else {}

    assets_raw = raw.get("assets")
    assets: Dict[str, str] = dict(assets_raw) if isinstance(assets_raw, dict) else {}

    metadata_raw = raw.get("metadata")
    metadata: Dict[str, Any] = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}

    man = WorkflowBundleManifest(
        bundle_format_version=version,
        bundle_id=bundle_id,
        bundle_version=bundle_version,
        created_at=created_at,
        entrypoints=entrypoints,
        flows=flows,
        artifacts=artifacts,
        assets=assets,
        metadata=metadata,
    )
    man.validate()
    return man


def workflow_bundle_manifest_to_dict(manifest: WorkflowBundleManifest) -> Dict[str, Any]:
    manifest.validate()
    return {
        "bundle_format_version": str(manifest.bundle_format_version),
        "bundle_id": str(manifest.bundle_id),
        "bundle_version": str(manifest.bundle_version),
        "created_at": str(manifest.created_at or ""),
        "entrypoints": [
            {
                "flow_id": ep.flow_id,
                "name": ep.name,
                "description": str(ep.description or ""),
                "interfaces": list(ep.interfaces or []),
            }
            for ep in list(manifest.entrypoints or [])
        ],
        "flows": dict(manifest.flows or {}),
        "artifacts": dict(manifest.artifacts or {}),
        "assets": dict(manifest.assets or {}),
        "metadata": dict(manifest.metadata or {}),
    }

