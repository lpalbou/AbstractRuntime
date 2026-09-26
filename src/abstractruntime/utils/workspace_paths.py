"""Shared workspace-path canonicalization helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple, Optional

_MOUNT_NAME_RE: set[str] = set("abcdefghijklmnopqrstuvwxyz0123456789_-")


class WorkspacePathError(ValueError):
    """Raised when a workspace path cannot be normalized safely."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


class WorkspacePathResolution(NamedTuple):
    resolved_path: Path
    virtual_path: str
    mount_name: Optional[str]
    root_path: Path


def resolve_no_strict(path: Path) -> Path:
    """Resolve without requiring the path to exist."""
    try:
        return path.resolve(strict=False)
    except TypeError:  # pragma: no cover
        return path.resolve()


def is_under_path(child: Path, parent: Path) -> bool:
    try:
        resolve_no_strict(child).relative_to(resolve_no_strict(parent))
        return True
    except Exception:
        return False


def slug_workspace_mount_name(name: str) -> str:
    """Return a stable mount slug (<= 32 chars, lower-case, [a-z0-9_-])."""
    raw = str(name or "").strip().lower()
    if not raw:
        return "mount"
    out: list[str] = []
    prev_dash = False
    for ch in raw:
        if ch in _MOUNT_NAME_RE:
            out.append(ch)
            prev_dash = ch == "-"
            continue
        if not prev_dash:
            out.append("-")
            prev_dash = True
    slug = "".join(out).strip("-")
    if not slug:
        return "mount"
    return slug[:32]


def _mount_digest(path: Path) -> str:
    return hashlib.sha256(str(resolve_no_strict(path)).encode("utf-8")).hexdigest()


def build_workspace_mounts(*, allowed_dirs: Iterable[Path], used_names: set[str]) -> dict[str, Path]:
    """Build deterministic mount aliases for allowed roots outside the base workspace."""
    unique_resolved: list[Path] = []
    seen: set[str] = set()
    for item in allowed_dirs:
        if not isinstance(item, Path):
            continue
        resolved = resolve_no_strict(item)
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique_resolved.append(resolved)

    grouped: dict[str, list[Path]] = {}
    for resolved in unique_resolved:
        base = slug_workspace_mount_name(resolved.name)
        grouped.setdefault(base, []).append(resolved)

    out: dict[str, Path] = {}
    for base in sorted(grouped.keys()):
        paths = sorted(grouped[base], key=lambda item: str(item))
        needs_digest = len(paths) > 1 or base in used_names
        for resolved in paths:
            if not needs_digest and base not in used_names:
                name = base
            else:
                digest = _mount_digest(resolved)
                name = ""
                for width in (8, 12, 16, 20, 24, 28, 32):
                    suffix = digest[:width]
                    trim = max(1, 32 - (1 + len(suffix)))
                    candidate = f"{base[:trim]}_{suffix}"
                    if candidate not in used_names:
                        name = candidate
                        break
                if not name:
                    idx = 2
                    while True:  # pragma: no branch
                        suffix = f"_{idx}"
                        trim = max(1, 32 - len(suffix))
                        candidate = f"{base[:trim]}{suffix}"
                        if candidate not in used_names:
                            name = candidate
                            break
                        idx += 1
            used_names.add(name)
            out[name] = resolved
    return out


def resolve_workspace_path(
    *,
    base: Path,
    mounts: Mapping[str, Path],
    raw_path: str,
    workspace_root_name: Optional[str] = None,
) -> WorkspacePathResolution:
    """Resolve a user-supplied server path and return its canonical virtual form."""

    def _normalize_rel(path: Path) -> str:
        rel = path.as_posix()
        return "" if rel in {"", "."} else rel

    text = str(raw_path or "").strip()
    if text.startswith("@"):
        text = text[1:].lstrip()
    if not text:
        raise WorkspacePathError("path is required", kind="required")

    normalized = text.replace("\\", "/")
    candidate = Path(normalized).expanduser()

    if candidate.is_absolute():
        try:
            resolved = resolve_no_strict(candidate)
        except Exception as exc:  # pragma: no cover
            raise WorkspacePathError("invalid absolute path", kind="invalid_absolute_path") from exc

        matches: list[tuple[int, Optional[str], Path]] = []
        if is_under_path(resolved, base):
            matches.append((len(str(base)), None, base))
        for mount_name, root in mounts.items():
            if not isinstance(root, Path):
                continue
            if is_under_path(resolved, root):
                matches.append((len(str(root)), str(mount_name), root))
        if not matches:
            raise WorkspacePathError("path is outside workspace roots", kind="outside_workspace_roots")
        matches.sort(key=lambda item: item[0], reverse=True)
        _len, mount_name, root = matches[0]
        rel = _normalize_rel(resolved.relative_to(root))
        virt = f"{mount_name}/{rel}" if mount_name and rel else (mount_name or rel)
        return WorkspacePathResolution(resolved, virt, mount_name, root)

    while normalized.startswith("./"):
        normalized = normalized[2:]

    parts = [segment for segment in normalized.split("/") if segment not in ("", ".")]
    mount_name: Optional[str] = None
    root = base
    rel_part = normalized

    if parts:
        head = parts[0]
        if head in mounts:
            mount_name = head
            root = mounts[head]
            rel_part = "/".join(parts[1:])
        elif workspace_root_name and head == workspace_root_name and not (base / head).exists():
            rel_part = "/".join(parts[1:])

    try:
        resolved = resolve_no_strict(root / Path(rel_part))
    except Exception as exc:  # pragma: no cover
        raise WorkspacePathError("invalid path", kind="invalid_path") from exc
    if not is_under_path(resolved, root):
        raise WorkspacePathError("path escapes workspace root", kind="path_escape")

    rel = _normalize_rel(resolved.relative_to(root))
    virt = f"{mount_name}/{rel}" if mount_name and rel else (mount_name or rel)
    return WorkspacePathResolution(resolved, virt, mount_name, root)


BUILTIN_DENY_KEY = "workspace_builtin_deny_prefixes"
BUILTIN_ALLOW_KEY = "workspace_builtin_allow"


def _path_entries(raw: object) -> list[str]:
    """A path list as hosts send it: a list, a JSON array string, or newline-separated text."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if isinstance(x, str) and x.strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("["):
            import json

            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()]
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    return []


def merge_builtin_workspace_protection(parent: Mapping, child: Mapping) -> dict:
    """The host's built-in protection a CHILD scope may carry, given its parent.

    The host (the gateway) sets `workspace_builtin_deny_prefixes` and
    `workspace_builtin_allow` on the run it starts; they are authoritative for
    the whole run tree. A child (a subworkflow's vars, a VisualFlow node's
    inputs) may ADD deny prefixes but never remove one, and may never widen the
    allow list: the result is

    - deny  = parent's deny prefixes, then the child's additional ones;
    - allow = parent's allow entries, plus the child's own `workspace_root`
      only when that folder already lies under one of the parent's allow
      entries (a child cannot allow itself into another folder of the data
      directory by pointing its root there). The child's own allow entries
      are ignored.

    Returns the two keys to set on the child, or {} when the parent carries no
    built-in deny (nothing to protect; the child's own values stand).
    """

    parent_deny = _path_entries(parent.get(BUILTIN_DENY_KEY))
    if not parent_deny:
        return {}
    deny = list(dict.fromkeys(parent_deny + _path_entries(child.get(BUILTIN_DENY_KEY))))
    parent_allow = _path_entries(parent.get(BUILTIN_ALLOW_KEY))
    allow = list(dict.fromkeys(parent_allow))
    child_root = child.get("workspace_root")
    if isinstance(child_root, str) and child_root.strip():
        root_path = Path(child_root.strip()).expanduser()
        if root_path.is_absolute() and any(is_under_path(root_path, Path(a).expanduser()) for a in parent_allow):
            if child_root.strip() not in allow:
                allow.append(child_root.strip())
    return {BUILTIN_DENY_KEY: deny, BUILTIN_ALLOW_KEY: allow}


__all__ = [
    "BUILTIN_ALLOW_KEY",
    "BUILTIN_DENY_KEY",
    "merge_builtin_workspace_protection",
    "WorkspacePathError",
    "WorkspacePathResolution",
    "build_workspace_mounts",
    "is_under_path",
    "resolve_no_strict",
    "resolve_workspace_path",
    "slug_workspace_mount_name",
]
