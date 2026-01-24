"""Workspace-scoped tool execution helpers.

This module provides utilities to scope filesystem-ish tool calls (files + shell)
to a workspace policy, driven by run `vars` / `input_data`.

Key concepts:
- `workspace_root`: base directory for resolving relative paths (and default cwd for `execute_command`).
- `workspace_access_mode`:
  - `workspace_only` (default): absolute paths must remain under `workspace_root`
  - `all_except_ignored`: absolute paths may escape `workspace_root` unless blocked by `workspace_ignored_paths`
  - `workspace_or_allowed`: absolute paths may escape `workspace_root` only when under `workspace_allowed_paths`
- `workspace_ignored_paths`: denylist of directories (absolute or relative-to-workspace_root).
- `workspace_allowed_paths`: allowlist of directories (absolute or relative-to-workspace_root).

Important limitations:
- `execute_command` is not a sandbox; commands can still write outside via absolute paths / `cd ..`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

WorkspaceAccessMode = str  # "workspace_only" | "all_except_ignored" | "workspace_or_allowed"

_VALID_ACCESS_MODES: set[str] = {"workspace_only", "all_except_ignored", "workspace_or_allowed"}
_MOUNT_NAME_RE: set[str] = set("abcdefghijklmnopqrstuvwxyz0123456789_-")


def _resolve_no_strict(path: Path) -> Path:
    """Resolve without requiring the path to exist (best-effort across py versions)."""
    try:
        return path.resolve(strict=False)
    except TypeError:  # pragma: no cover (older python)
        return path.resolve()


def _find_repo_root_from_here(*, start: Path, max_hops: int = 10) -> Optional[Path]:
    """Best-effort monorepo root detection for local/dev runs."""
    cur = _resolve_no_strict(start)
    for _ in range(max_hops):
        docs = cur / "docs" / "KnowledgeBase.md"
        if docs.exists():
            return cur
        if (cur / "abstractflow").exists() and (cur / "abstractcore").exists() and (cur / "abstractruntime").exists():
            return cur
        nxt = cur.parent
        if nxt == cur:
            break
        cur = nxt
    return None


def resolve_workspace_base_dir() -> Path:
    """Base directory against which relative workspace roots are resolved.

    Priority:
    - `ABSTRACT_WORKSPACE_BASE_DIR` env var, if set.
    - `ABSTRACTFLOW_WORKSPACE_BASE_DIR` env var, if set (backward compat).
    - Best-effort monorepo root detection from this file location.
    - Current working directory.
    """
    env = os.getenv("ABSTRACT_WORKSPACE_BASE_DIR") or os.getenv("ABSTRACTFLOW_WORKSPACE_BASE_DIR")
    if isinstance(env, str) and env.strip():
        return _resolve_no_strict(Path(env.strip()).expanduser())

    here_dir = Path(__file__).resolve().parent
    guessed = _find_repo_root_from_here(start=here_dir)
    if guessed is not None:
        return guessed

    return _resolve_no_strict(Path.cwd())


def _normalize_access_mode(raw: Any) -> WorkspaceAccessMode:
    text = str(raw or "").strip().lower()
    if not text:
        return "workspace_only"
    if text in _VALID_ACCESS_MODES:
        return text
    raise ValueError(f"Invalid workspace_access_mode: '{raw}' (expected one of: {sorted(_VALID_ACCESS_MODES)})")


def _parse_ignored_paths(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        out: list[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        # Tolerate users pasting a JSON array into a text field.
        if text.startswith("["):
            try:
                import json

                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if isinstance(x, str) and str(x).strip()]
            except Exception:
                pass
        # Newline-separated entries (UI-friendly).
        lines = [ln.strip() for ln in text.splitlines()]
        return [ln for ln in lines if ln]
    return []


def _parse_allowed_paths(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        out: list[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        # Tolerate users pasting a JSON array into a text field.
        if text.startswith("["):
            try:
                import json

                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if isinstance(x, str) and str(x).strip()]
            except Exception:
                pass
        # Newline-separated entries (UI-friendly).
        lines = [ln.strip() for ln in text.splitlines()]
        return [ln for ln in lines if ln]
    return []


def _resolve_ignored_paths(*, root: Path, ignored: Iterable[str]) -> Tuple[Path, ...]:
    out: list[Path] = []
    for raw in ignored:
        s = str(raw or "").strip()
        if not s:
            continue
        p = Path(s).expanduser()
        if not p.is_absolute():
            p = root / p
        out.append(_resolve_no_strict(p))
    # Stable ordering for deterministic error messages/tests.
    return tuple(dict.fromkeys(out))


def _resolve_allowed_paths(*, root: Path, allowed: Iterable[str]) -> Tuple[Path, ...]:
    out: list[Path] = []
    for raw in allowed:
        s = str(raw or "").strip()
        if not s:
            continue
        p = Path(s).expanduser()
        if not p.is_absolute():
            p = root / p
        out.append(_resolve_no_strict(p))
    return tuple(dict.fromkeys(out))


def _is_under(child: Path, parent: Path) -> bool:
    try:
        _resolve_no_strict(child).relative_to(_resolve_no_strict(parent))
        return True
    except Exception:
        return False


def _slug_mount_name(name: str) -> str:
    """Return a stable mount name (<= 32 chars, lower-case, [a-z0-9_-])."""
    s = str(name or "").strip().lower()
    if not s:
        return "mount"
    out: list[str] = []
    for ch in s:
        if ch in _MOUNT_NAME_RE:
            out.append(ch)
        else:
            out.append("-")
    slug = "".join(out).strip("-")
    if not slug:
        return "mount"
    return slug[:32]


def _mounts_from_allowed_paths(*, allowed_dirs: Iterable[Path], used_names: set[str]) -> Dict[str, Path]:
    """Build a deterministic {mount_name -> root} map for allowed roots outside workspace_root."""
    import hashlib

    out: Dict[str, Path] = {}
    for p in allowed_dirs:
        try:
            resolved = p.resolve()
        except Exception:
            resolved = p
        base = _slug_mount_name(resolved.name)
        name = base
        if name in used_names:
            digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
            trim = max(1, 32 - (1 + len(digest)))
            name = f"{base[:trim]}_{digest}"
        i = 2
        while name in used_names:
            suffix = f"_{i}"
            trim = max(1, 32 - len(suffix))
            name = f"{base[:trim]}{suffix}"
            i += 1
        used_names.add(name)
        out[name] = resolved
    return out


def _resolve_virtual_mount_relative_path(*, scope: "WorkspaceScope", raw: str) -> tuple[Path, str]:
    """Resolve a relative path that may be a virtual mount path.

    Supported forms:
      - "rel/path.txt" (workspace_root)
      - "mount/rel/path.txt" (allowed root mount; only when access_mode==workspace_or_allowed)
      - "<workspace_root_name>/rel/path.txt" (best-effort redundant prefix stripping)
      - Optional leading "@", tolerated for UX across clients ("@mount/rel/path.txt")

    Returns:
      (root_used, rel_part) where rel_part is a relative path to join under root_used.
    """
    text = str(raw or "").strip().replace("\\", "/")
    if text.startswith("@"):
        text = text[1:].lstrip()
    while text.startswith("./"):
        text = text[2:]

    parts = [seg for seg in text.split("/") if seg not in ("", ".")]
    if len(parts) < 2:
        return (scope.root, text)

    first = parts[0]

    # Mounts: allow a "mount/..." prefix for allowed roots outside workspace_root.
    if scope.access_mode == "workspace_or_allowed" and scope.allowed_paths:
        used: set[str] = set()
        allowed_outside = [p for p in scope.allowed_paths if isinstance(p, Path) and not _is_under(p, scope.root)]
        mounts = _mounts_from_allowed_paths(allowed_dirs=allowed_outside, used_names=used)
        if first in mounts:
            root = mounts[first]
            rel = "/".join(parts[1:])
            return (root, rel)

    # Convenience: if the path redundantly begins with the workspace directory name, strip it
    # when it does not exist as a real child directory (common with "repo-name/..." patterns).
    try:
        if first == scope.root.name and not (scope.root / first).exists():
            rel = "/".join(parts[1:])
            return (scope.root, rel)
    except Exception:
        pass

    return (scope.root, text)


def _ensure_allowed(*, path: Path, scope: "WorkspaceScope") -> None:
    for blocked in scope.ignored_paths:
        if _is_under(path, blocked) or _resolve_no_strict(path) == _resolve_no_strict(blocked):
            raise ValueError(f"Path is blocked by workspace_ignored_paths: '{path}'")


def _resolve_under_root_strict(*, root: Path, user_path: str) -> Path:
    """Resolve under root and ensure it doesn't escape (used for relative paths always)."""
    p = Path(str(user_path or "").strip()).expanduser()
    if p.is_absolute():
        raise ValueError("Internal error: strict under-root resolver received absolute path")
    resolved = _resolve_no_strict(root / p)
    if not _is_under(resolved, root):
        raise ValueError(f"Path escapes workspace_root: '{user_path}'")
    return resolved


def resolve_user_path(*, scope: "WorkspaceScope", user_path: str) -> Path:
    """Resolve a user path according to workspace policy."""
    raw = str(user_path or "").strip()
    if not raw:
        raise ValueError("Empty path")

    # Tolerate "@path" handles (used by attachments and some UIs) for filesystem-ish tools.
    if raw.startswith("@"):
        raw = raw[1:].lstrip()

    p = Path(raw).expanduser()
    if p.is_absolute():
        resolved = _resolve_no_strict(p)
        if scope.access_mode == "workspace_only":
            if not _is_under(resolved, scope.root):
                raise ValueError(f"Path escapes workspace_root: '{user_path}'")
        elif scope.access_mode == "workspace_or_allowed":
            if not _is_under(resolved, scope.root) and not any(_is_under(resolved, p) for p in scope.allowed_paths):
                raise ValueError(f"Path is outside workspace roots: '{user_path}'")
        _ensure_allowed(path=resolved, scope=scope)
        return resolved

    # Relative paths normally resolve under workspace_root, but we also support a
    # conservative "mount/..." convention for allowed roots (mirrors gateway file endpoints).
    root_used, rel_part = _resolve_virtual_mount_relative_path(scope=scope, raw=raw)
    resolved = _resolve_under_root_strict(root=root_used, user_path=rel_part)
    _ensure_allowed(path=resolved, scope=scope)
    return resolved


def _normalize_arguments(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    # Some models emit JSON strings for args.
    if isinstance(raw, str) and raw.strip():
        import json

        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


@dataclass(frozen=True)
class WorkspaceScope:
    root: Path
    access_mode: WorkspaceAccessMode = "workspace_only"
    ignored_paths: Tuple[Path, ...] = ()
    allowed_paths: Tuple[Path, ...] = ()

    @classmethod
    def from_input_data(
        cls,
        input_data: Dict[str, Any],
        *,
        key: str = "workspace_root",
        base_dir: Optional[Path] = None,
    ) -> Optional["WorkspaceScope"]:
        raw = input_data.get(key)
        if not isinstance(raw, str) or not raw.strip():
            return None

        base = base_dir or resolve_workspace_base_dir()
        root = Path(raw.strip()).expanduser()
        if not root.is_absolute():
            root = base / root
        root = _resolve_no_strict(root)
        if root.exists() and not root.is_dir():
            raise ValueError(f"workspace_root must be a directory (got file): {raw}")
        root.mkdir(parents=True, exist_ok=True)

        access_mode = _normalize_access_mode(input_data.get("workspace_access_mode") or input_data.get("workspaceAccessMode"))
        ignored = _parse_ignored_paths(input_data.get("workspace_ignored_paths") or input_data.get("workspaceIgnoredPaths"))
        ignored_paths = _resolve_ignored_paths(root=root, ignored=ignored)
        allowed = _parse_allowed_paths(input_data.get("workspace_allowed_paths") or input_data.get("workspaceAllowedPaths"))
        allowed_paths = _resolve_allowed_paths(root=root, allowed=allowed)

        return cls(root=root, access_mode=access_mode, ignored_paths=ignored_paths, allowed_paths=allowed_paths)


class WorkspaceScopedToolExecutor:
    """Wrap another ToolExecutor and scope filesystem-ish tool calls to a workspace policy."""

    def __init__(self, *, scope: WorkspaceScope, delegate: Any):
        self._scope = scope
        self._delegate = delegate

    def set_timeout_s(self, timeout_s: Optional[float]) -> None:  # pragma: no cover (depends on delegate)
        setter = getattr(self._delegate, "set_timeout_s", None)
        if callable(setter):
            setter(timeout_s)

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Preprocess: rewrite and pre-block invalid calls so we don't crash the whole run.
        blocked: Dict[Tuple[int, str], Dict[str, Any]] = {}
        to_execute: List[Dict[str, Any]] = []

        for i, tc in enumerate(tool_calls or []):
            name = str(tc.get("name", "") or "")
            call_id = str(tc.get("call_id") or tc.get("id") or f"call_{i}")
            args = _normalize_arguments(tc.get("arguments"))

            try:
                rewritten_args = self._rewrite_args(tool_name=name, args=args)
            except Exception as e:
                blocked[(i, call_id)] = {
                    "call_id": call_id,
                    "name": name,
                    "success": False,
                    "output": None,
                    "error": str(e),
                }
                continue

            rewritten = dict(tc)
            rewritten["name"] = name
            rewritten["call_id"] = call_id
            rewritten["arguments"] = rewritten_args
            to_execute.append(rewritten)

        delegate_result = self._delegate.execute(tool_calls=to_execute)

        # If the delegate didn't execute tools, we can't merge blocked results meaningfully.
        if not isinstance(delegate_result, dict) or delegate_result.get("mode") != "executed":
            return delegate_result

        results = delegate_result.get("results")
        if not isinstance(results, list):
            results = []

        by_id: Dict[str, Dict[str, Any]] = {}
        for r in results:
            if not isinstance(r, dict):
                continue
            rid = str(r.get("call_id") or "")
            if rid:
                by_id[rid] = r

        merged: List[Dict[str, Any]] = []
        for i, tc in enumerate(tool_calls or []):
            call_id = str(tc.get("call_id") or tc.get("id") or f"call_{i}")
            key = (i, call_id)
            if key in blocked:
                merged.append(blocked[key])
                continue
            r = by_id.get(call_id)
            if r is None:
                merged.append(
                    {
                        "call_id": call_id,
                        "name": str(tc.get("name", "") or ""),
                        "success": False,
                        "output": None,
                        "error": "Tool result missing (internal error)",
                    }
                )
                continue
            merged.append(r)

        return {"mode": "executed", "results": merged}

    def _rewrite_args(self, *, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return rewrite_tool_arguments(tool_name=tool_name, args=args, scope=self._scope)


def rewrite_tool_arguments(*, tool_name: str, args: Dict[str, Any], scope: WorkspaceScope) -> Dict[str, Any]:
    """Rewrite tool args so file operations follow the workspace policy."""
    root = scope.root
    out = dict(args or {})

    def _alias_field(preferred: str, aliases: Iterable[str]) -> None:
        if preferred in out and out.get(preferred) is not None:
            return
        for a in aliases:
            if a in out and out.get(a) is not None:
                out[preferred] = out.get(a)
                return

    def _rewrite_path_field(field: str, *, default_to_root: bool = False) -> None:
        raw = out.get(field)
        if (raw is None or (isinstance(raw, str) and not raw.strip())) and default_to_root:
            out[field] = str(_resolve_no_strict(root))
            return
        if raw is None:
            return
        if not isinstance(raw, str):
            raw = str(raw)
        resolved = resolve_user_path(scope=scope, user_path=raw)
        out[field] = str(resolved)

    def _rewrite_path_list_field(field: str) -> None:
        raw = out.get(field)
        if raw is None:
            return

        items: list[Any]
        if isinstance(raw, list):
            items = list(raw)
        elif isinstance(raw, tuple):
            items = list(raw)
        else:
            # Accept a single string (or scalar) and let the underlying tool parse it.
            items = [raw]

        rewritten: list[str] = []
        for it in items:
            s = str(it or "").strip()
            if not s:
                continue
            resolved = resolve_user_path(scope=scope, user_path=s)
            rewritten.append(str(resolved))

        out[field] = rewritten

    # Filesystem-ish tools (AbstractCore common tools)
    if tool_name == "list_files":
        _rewrite_path_field("directory_path", default_to_root=True)
        return out
    if tool_name == "search_files":
        _rewrite_path_field("path", default_to_root=True)
        return out
    if tool_name == "analyze_code":
        _alias_field("file_path", ["path", "filename", "file"])
        _rewrite_path_field("file_path")
        if "file_path" not in out:
            raise ValueError("analyze_code requires file_path")
        return out
    if tool_name == "read_file":
        _alias_field("file_path", ["path", "filename", "file"])
        _rewrite_path_field("file_path")
        if "file_path" not in out:
            raise ValueError("read_file requires file_path")
        return out
    if tool_name == "write_file":
        _alias_field("file_path", ["path", "filename", "file"])
        _rewrite_path_field("file_path")
        if "file_path" not in out:
            raise ValueError("write_file requires file_path")
        return out
    if tool_name == "edit_file":
        _alias_field("file_path", ["path", "filename", "file"])
        _rewrite_path_field("file_path")
        if "file_path" not in out:
            raise ValueError("edit_file requires file_path")
        return out
    if tool_name == "execute_command":
        _rewrite_path_field("working_directory", default_to_root=True)
        return out
    if tool_name == "skim_files":
        _alias_field("paths", ["path", "file_path", "filename", "file"])
        _rewrite_path_list_field("paths")
        if "paths" not in out:
            raise ValueError("skim_files requires paths")
        return out
    if tool_name == "skim_folders":
        _alias_field("paths", ["path", "directory_path", "folder"])
        _rewrite_path_list_field("paths")
        if "paths" not in out:
            raise ValueError("skim_folders requires paths")
        return out

    return out


__all__ = [
    "WorkspaceAccessMode",
    "WorkspaceScope",
    "WorkspaceScopedToolExecutor",
    "rewrite_tool_arguments",
    "resolve_workspace_base_dir",
    "resolve_user_path",
]
