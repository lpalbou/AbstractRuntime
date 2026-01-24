"""WorkflowBundleRegistry (disk) for installed `.flow` bundles.

This registry is a host-side convenience layer:
- stores `.flow` bundles in a directory
- provides interface discovery from bundle manifests
- resolves bundle refs like `bundle_id@version` (or latest version)

Design goals:
- stdlib-only (to keep AbstractRuntime minimal)
- no global state; callers choose the directory
- correctness-first (scan bundles; optional caching can be added later)
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import WorkflowBundleEntrypoint, WorkflowBundleManifest
from .reader import open_workflow_bundle


class WorkflowBundleRegistryError(ValueError):
    """Raised when registry operations fail (install/remove/resolve)."""


_BUNDLE_ID_SAFE_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_BUNDLE_VERSION_SAFE_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def sanitize_bundle_id(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    s = _BUNDLE_ID_SAFE_RE.sub("-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def sanitize_bundle_version(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    s = _BUNDLE_VERSION_SAFE_RE.sub("-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def _try_parse_semver(v: str) -> Optional[Tuple[int, int, int]]:
    s = str(v or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(".")]
    if not parts or any(not p for p in parts):
        return None
    nums: list[int] = []
    for p in parts:
        if not p.isdigit():
            return None
        nums.append(int(p))
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


def default_workflow_bundles_dir() -> Path:
    """Resolve the default bundles directory for `.flow` bundles.

    Priority:
    1) `ABSTRACTFRAMEWORK_WORKFLOWS_DIR` (shared/cross-package)
    2) `ABSTRACTGATEWAY_FLOWS_DIR` (gateway bundle host)
    3) `ABSTRACTFLOW_PUBLISH_DIR` (authoring host publish)
    4) `ABSTRACTFLOW_FLOWS_DIR` (legacy)
    5) repo/dev fallback: `./flows/bundles/` if it exists
    6) user default: `~/.abstractframework/workflows/`
    """
    env_candidates = (
        "ABSTRACTFRAMEWORK_WORKFLOWS_DIR",
        "ABSTRACTGATEWAY_FLOWS_DIR",
        "ABSTRACTFLOW_PUBLISH_DIR",
        "ABSTRACTFLOW_FLOWS_DIR",
    )
    for name in env_candidates:
        v = os.getenv(name)
        if isinstance(v, str) and v.strip():
            return Path(v.strip()).expanduser().resolve()

    dev = Path("flows") / "bundles"
    if dev.exists() and dev.is_dir():
        return dev.resolve()

    return (Path.home() / ".abstractframework" / "workflows").resolve()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class InstalledWorkflowBundle:
    bundle_id: str
    bundle_version: str
    path: Path
    manifest: WorkflowBundleManifest
    sha256: Optional[str] = None

    @property
    def bundle_ref(self) -> str:
        return f"{self.bundle_id}@{self.bundle_version}"


@dataclass(frozen=True)
class WorkflowEntrypointRef:
    bundle_id: str
    bundle_version: str
    flow_id: str
    name: str = ""
    description: str = ""
    interfaces: List[str] = field(default_factory=list)
    is_default: bool = False

    @property
    def bundle_ref(self) -> str:
        return f"{self.bundle_id}@{self.bundle_version}"

    @property
    def workflow_id(self) -> str:
        return f"{self.bundle_ref}:{self.flow_id}"


def _split_bundle_ref(raw: str) -> tuple[str, Optional[str]]:
    s = str(raw or "").strip()
    if not s:
        return ("", None)
    # Allow passing a bundle ref with a `.flow` suffix (e.g. "basic-agent.flow")
    # when referring to an installed bundle id (not a filesystem path).
    if s.lower().endswith(".flow") and "/" not in s and "\\" not in s:
        s = s[:-5]
    if "@" not in s:
        return (s, None)
    a, b = s.split("@", 1)
    a = a.strip()
    b = b.strip()
    if not a:
        return ("", None)
    if not b:
        return (a, None)
    return (a, b)


def _pick_latest_version(bundles_by_version: Dict[str, InstalledWorkflowBundle]) -> Optional[str]:
    items = [(str(ver), b) for ver, b in (bundles_by_version or {}).items() if isinstance(ver, str)]
    if not items:
        return None

    if all(_try_parse_semver(ver) is not None for ver, _ in items):
        return max(items, key=lambda x: _try_parse_semver(x[0]) or (0, 0, 0))[0]

    # Fallback: prefer newest created_at.
    def _key(x: tuple[str, InstalledWorkflowBundle]) -> tuple[str, str]:
        ver, b = x
        created = str(getattr(getattr(b, "manifest", None), "created_at", "") or "")
        return (created, ver)

    return max(items, key=_key)[0]


class WorkflowBundleRegistry:
    """A directory-backed registry for `.flow` bundles."""

    def __init__(self, bundles_dir: str | Path | None = None) -> None:
        base = Path(bundles_dir).expanduser() if bundles_dir is not None else default_workflow_bundles_dir()
        self.bundles_dir = base.expanduser().resolve()

    def ensure_dir(self) -> Path:
        self.bundles_dir.mkdir(parents=True, exist_ok=True)
        return self.bundles_dir

    def scan(self) -> List[InstalledWorkflowBundle]:
        """Scan the bundles directory for `.flow` bundles (best-effort)."""
        if not self.bundles_dir.exists() or not self.bundles_dir.is_dir():
            return []

        out: list[InstalledWorkflowBundle] = []
        for path in sorted(self.bundles_dir.glob("*.flow")):
            if not path.is_file():
                continue
            try:
                bundle = open_workflow_bundle(path)
            except Exception:
                continue
            man = bundle.manifest
            bid = str(getattr(man, "bundle_id", "") or "").strip()
            bver = str(getattr(man, "bundle_version", "") or "0.0.0").strip() or "0.0.0"
            if not bid:
                continue
            out.append(InstalledWorkflowBundle(bundle_id=bid, bundle_version=bver, path=path.resolve(), manifest=man))
        return out

    def bundles_by_id(self) -> Dict[str, Dict[str, InstalledWorkflowBundle]]:
        out: Dict[str, Dict[str, InstalledWorkflowBundle]] = {}
        for b in self.scan():
            out.setdefault(b.bundle_id, {})[b.bundle_version] = b
        return out

    def resolve_bundle(self, bundle_ref: str) -> InstalledWorkflowBundle:
        """Resolve `bundle_id[@version]` to an installed bundle (prefers latest)."""
        bid, ver = _split_bundle_ref(bundle_ref)
        if not bid:
            raise WorkflowBundleRegistryError("bundle_ref must be 'bundle_id' or 'bundle_id@version'")

        bundles = self.bundles_by_id().get(bid) or {}
        if not bundles:
            raise WorkflowBundleRegistryError(f"Bundle '{bid}' is not installed in {self.bundles_dir}")

        if ver:
            b = bundles.get(ver)
            if b is None:
                available = ", ".join(sorted(bundles.keys()))
                raise WorkflowBundleRegistryError(f"Bundle '{bid}@{ver}' not found (available: {available})")
            return b

        latest = _pick_latest_version(bundles)
        if not latest:
            raise WorkflowBundleRegistryError(f"Bundle '{bid}' has no versions installed")
        return bundles[latest]

    def resolve_entrypoint(
        self,
        ref: str,
        *,
        interface: Optional[str] = None,
    ) -> WorkflowEntrypointRef:
        """Resolve a workflow reference to a bundle entrypoint.

        Supported refs:
        - `bundle_id` (uses manifest.default_entrypoint if set, else first entrypoint)
        - `bundle_id@version`
        - `bundle_id[:flow_id]` or `bundle_id@version:flow_id`
        - `entrypoint_name` (unique match across bundles, best-effort)
        """
        s = str(ref or "").strip()
        if not s:
            raise WorkflowBundleRegistryError("workflow reference is required")

        bundle_flow_id: Optional[str] = None
        # Support bundle_id:flow_id (avoid clobbering Windows drive letters).
        if ":" in s and not re.match(r"^[A-Za-z]:[\\\\/]", s):
            left, right = s.split(":", 1)
            if left.strip() and right.strip():
                s = left.strip()
                bundle_flow_id = right.strip()

        # If the caller passed a `.flow` filename (not found relative to CWD),
        # try resolving it within the registry directory.
        if s.lower().endswith(".flow") and "/" not in s and "\\" not in s:
            candidate = (self.bundles_dir / s).expanduser()
            if candidate.exists() and candidate.is_file():
                try:
                    b = open_workflow_bundle(candidate)
                    man = b.manifest
                    return self._entrypoint_from_manifest(
                        manifest=man,
                        bundle_id=str(getattr(man, "bundle_id", "") or "").strip(),
                        bundle_version=str(getattr(man, "bundle_version", "") or "0.0.0").strip() or "0.0.0",
                        flow_id=bundle_flow_id,
                        interface=interface,
                    )
                except Exception:
                    pass

        # Fast path: resolve by bundle ref.
        try:
            bundle = self.resolve_bundle(s)
            man = bundle.manifest
            return self._entrypoint_from_manifest(
                manifest=man,
                bundle_id=bundle.bundle_id,
                bundle_version=bundle.bundle_version,
                flow_id=bundle_flow_id,
                interface=interface,
            )
        except WorkflowBundleRegistryError:
            pass

        # Best-effort: resolve by entrypoint name.
        needle = s.casefold()
        matches: list[WorkflowEntrypointRef] = []
        for b in self.scan():
            man = b.manifest
            for ep in list(getattr(man, "entrypoints", None) or []):
                ep_name = str(getattr(ep, "name", "") or "").strip()
                if not ep_name or ep_name.casefold() != needle:
                    continue
                epr = self._entrypoint_from_entrypoint(
                    ep=ep,
                    bundle_id=b.bundle_id,
                    bundle_version=b.bundle_version,
                    is_default=str(getattr(man, "default_entrypoint", "") or "").strip() == str(getattr(ep, "flow_id", "") or "").strip(),
                )
                if interface and interface not in epr.interfaces:
                    continue
                matches.append(epr)

        if not matches:
            raise WorkflowBundleRegistryError(f"Workflow '{ref}' not found in {self.bundles_dir}")

        if len(matches) > 1:
            options = ", ".join(sorted({m.bundle_ref for m in matches}))
            raise WorkflowBundleRegistryError(
                f"Workflow name '{ref}' matches multiple bundles ({options}); use bundle_id[@version] instead"
            )
        return matches[0]

    def list_entrypoints(
        self,
        *,
        interface: Optional[str] = None,
        latest_only: bool = True,
    ) -> List[WorkflowEntrypointRef]:
        """List available entrypoints (optionally filtered by interface)."""
        bundles = self.bundles_by_id()
        selected: Iterable[InstalledWorkflowBundle]
        if latest_only:
            latest: list[InstalledWorkflowBundle] = []
            for bid, versions in bundles.items():
                latest_ver = _pick_latest_version(versions)
                if latest_ver:
                    latest.append(versions[latest_ver])
            selected = latest
        else:
            selected = [b for versions in bundles.values() for b in versions.values()]

        out: list[WorkflowEntrypointRef] = []
        for b in selected:
            man = b.manifest
            default_fid = str(getattr(man, "default_entrypoint", "") or "").strip()
            for ep in list(getattr(man, "entrypoints", None) or []):
                epr = self._entrypoint_from_entrypoint(
                    ep=ep,
                    bundle_id=b.bundle_id,
                    bundle_version=b.bundle_version,
                    is_default=bool(default_fid and str(getattr(ep, "flow_id", "") or "").strip() == default_fid),
                )
                if interface and interface not in epr.interfaces:
                    continue
                out.append(epr)

        out.sort(key=lambda e: (e.bundle_id, e.bundle_version, e.name or e.flow_id))
        return out

    def install(self, source: str | Path, *, overwrite: bool = False) -> InstalledWorkflowBundle:
        """Install a `.flow` bundle into the registry directory."""
        src = Path(source).expanduser().resolve()
        if not src.exists() or not src.is_file():
            raise WorkflowBundleRegistryError(f"Bundle not found: {src}")

        bundle = open_workflow_bundle(src)
        man = bundle.manifest
        bid = str(getattr(man, "bundle_id", "") or "").strip()
        bver = str(getattr(man, "bundle_version", "") or "0.0.0").strip() or "0.0.0"
        if not bid:
            raise WorkflowBundleRegistryError(f"Bundle '{src}' has empty manifest.bundle_id")

        safe_id = sanitize_bundle_id(bid)
        safe_ver = sanitize_bundle_version(bver)
        if safe_id != bid:
            raise WorkflowBundleRegistryError(
                f"Bundle id '{bid}' contains unsafe characters. "
                f"Publish with a safe bundle_id (suggested: '{safe_id}')."
            )
        if safe_ver != bver:
            raise WorkflowBundleRegistryError(
                f"Bundle version '{bver}' contains unsafe characters. "
                f"Publish with a safe bundle_version (suggested: '{safe_ver}')."
            )

        self.ensure_dir()
        dest = (self.bundles_dir / f"{bid}@{bver}.flow").resolve()
        if dest.exists():
            if not overwrite:
                raise WorkflowBundleRegistryError(f"Bundle already installed: {dest}")
            try:
                dest.unlink()
            except Exception as e:
                raise WorkflowBundleRegistryError(f"Failed removing existing bundle: {dest} ({e})") from e

        try:
            shutil.copy2(src, dest)
        except Exception as e:
            raise WorkflowBundleRegistryError(f"Failed installing bundle to {dest}: {e}") from e

        try:
            bundle2 = open_workflow_bundle(dest)
            sha = _sha256_file(dest)
            return InstalledWorkflowBundle(
                bundle_id=str(bundle2.manifest.bundle_id),
                bundle_version=str(bundle2.manifest.bundle_version),
                path=dest,
                manifest=bundle2.manifest,
                sha256=sha,
            )
        except Exception:
            # Best-effort: return what we know even if re-open/hash fails.
            return InstalledWorkflowBundle(bundle_id=bid, bundle_version=bver, path=dest, manifest=man)

    def install_bytes(
        self,
        content: bytes,
        *,
        filename_hint: str = "upload.flow",
        overwrite: bool = False,
    ) -> InstalledWorkflowBundle:
        """Install bundle bytes into the registry directory.

        This is primarily intended for hosts that receive `.flow` bytes over the network.
        """
        data = bytes(content or b"")
        if not data:
            raise WorkflowBundleRegistryError("Bundle content is empty")

        self.ensure_dir()
        suffix = ".flow" if str(filename_hint or "").lower().endswith(".flow") else ".flow"
        try:
            fd, tmp_path_str = tempfile.mkstemp(prefix=".upload_", suffix=suffix, dir=str(self.bundles_dir))
            tmp_path = Path(tmp_path_str)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        except Exception as e:
            raise WorkflowBundleRegistryError(f"Failed writing upload temp file: {e}") from e

        try:
            bundle = open_workflow_bundle(tmp_path)
            man = bundle.manifest
            bid = str(getattr(man, "bundle_id", "") or "").strip()
            bver = str(getattr(man, "bundle_version", "") or "0.0.0").strip() or "0.0.0"
            if not bid:
                raise WorkflowBundleRegistryError("Uploaded bundle has empty manifest.bundle_id")

            safe_id = sanitize_bundle_id(bid)
            safe_ver = sanitize_bundle_version(bver)
            if safe_id != bid:
                raise WorkflowBundleRegistryError(
                    f"Bundle id '{bid}' contains unsafe characters. "
                    f"Publish with a safe bundle_id (suggested: '{safe_id}')."
                )
            if safe_ver != bver:
                raise WorkflowBundleRegistryError(
                    f"Bundle version '{bver}' contains unsafe characters. "
                    f"Publish with a safe bundle_version (suggested: '{safe_ver}')."
                )

            dest = (self.bundles_dir / f"{bid}@{bver}.flow").resolve()
            if dest.exists() and not overwrite:
                raise WorkflowBundleRegistryError(f"Bundle already installed: {dest.name}")

            try:
                tmp_path.replace(dest)
            except Exception as e:
                raise WorkflowBundleRegistryError(f"Failed installing bundle to {dest}: {e}") from e

            try:
                bundle2 = open_workflow_bundle(dest)
                sha = _sha256_file(dest)
                return InstalledWorkflowBundle(
                    bundle_id=str(bundle2.manifest.bundle_id),
                    bundle_version=str(bundle2.manifest.bundle_version),
                    path=dest,
                    manifest=bundle2.manifest,
                    sha256=sha,
                )
            except Exception:
                return InstalledWorkflowBundle(bundle_id=bid, bundle_version=bver, path=dest, manifest=man)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    def remove(self, bundle_ref: str) -> int:
        """Remove installed bundle(s).

        Args:
            bundle_ref: `bundle_id` removes all versions, `bundle_id@version` removes that version.

        Returns:
            Number of removed files.
        """
        bid, ver = _split_bundle_ref(bundle_ref)
        if not bid:
            raise WorkflowBundleRegistryError("bundle_ref must be 'bundle_id' or 'bundle_id@version'")

        bundles = self.bundles_by_id().get(bid) or {}
        if not bundles:
            return 0

        removed = 0
        if ver:
            target = bundles.get(ver)
            if target is None:
                return 0
            try:
                target.path.unlink()
            except Exception as e:
                raise WorkflowBundleRegistryError(f"Failed removing {target.path}: {e}") from e
            return 1

        for b in bundles.values():
            try:
                b.path.unlink()
                removed += 1
            except Exception as e:
                raise WorkflowBundleRegistryError(f"Failed removing {b.path}: {e}") from e
        return removed

    @staticmethod
    def _entrypoint_from_entrypoint(
        *,
        ep: WorkflowBundleEntrypoint,
        bundle_id: str,
        bundle_version: str,
        is_default: bool,
    ) -> WorkflowEntrypointRef:
        fid = str(getattr(ep, "flow_id", "") or "").strip()
        name = str(getattr(ep, "name", "") or "").strip()
        desc = str(getattr(ep, "description", "") or "")
        interfaces = [str(x).strip() for x in list(getattr(ep, "interfaces", None) or []) if isinstance(x, str) and x.strip()]
        return WorkflowEntrypointRef(
            bundle_id=bundle_id,
            bundle_version=bundle_version,
            flow_id=fid,
            name=name,
            description=desc,
            interfaces=interfaces,
            is_default=bool(is_default),
        )

    def _entrypoint_from_manifest(
        self,
        *,
        manifest: WorkflowBundleManifest,
        bundle_id: str,
        bundle_version: str,
        flow_id: Optional[str],
        interface: Optional[str],
    ) -> WorkflowEntrypointRef:
        eps = list(getattr(manifest, "entrypoints", None) or [])
        if not eps:
            raise WorkflowBundleRegistryError(f"Bundle '{bundle_id}@{bundle_version}' has no entrypoints")

        default_fid = str(getattr(manifest, "default_entrypoint", "") or "").strip()
        chosen = None
        if flow_id:
            chosen = next((ep for ep in eps if str(getattr(ep, "flow_id", "") or "").strip() == flow_id), None)
            if chosen is None:
                available = ", ".join(sorted({str(getattr(ep, 'flow_id', '') or '').strip() for ep in eps if str(getattr(ep, 'flow_id', '') or '').strip()}))
                raise WorkflowBundleRegistryError(
                    f"Entrypoint '{flow_id}' not found in bundle '{bundle_id}@{bundle_version}' (available: {available})"
                )
        elif default_fid:
            chosen = next((ep for ep in eps if str(getattr(ep, "flow_id", "") or "").strip() == default_fid), None)

        if chosen is None:
            chosen = eps[0]

        epr = self._entrypoint_from_entrypoint(
            ep=chosen,
            bundle_id=bundle_id,
            bundle_version=bundle_version,
            is_default=bool(default_fid and str(getattr(chosen, "flow_id", "") or "").strip() == default_fid),
        )
        if interface and interface not in epr.interfaces:
            raise WorkflowBundleRegistryError(
                f"Selected entrypoint '{epr.workflow_id}' does not implement interface '{interface}'"
            )
        return epr
