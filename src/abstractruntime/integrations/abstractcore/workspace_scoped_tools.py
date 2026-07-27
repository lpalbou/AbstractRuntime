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
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_logger = logging.getLogger(__name__)

from abstractruntime.utils.workspace_paths import (
    WorkspacePathError,
    WorkspacePathResolution,
    build_workspace_mounts,
    is_under_path,
    resolve_no_strict,
    resolve_workspace_path as resolve_canonical_workspace_path,
)

WorkspaceAccessMode = str  # "workspace_only" | "all_except_ignored" | "workspace_or_allowed"

_VALID_ACCESS_MODES: set[str] = {"workspace_only", "all_except_ignored", "workspace_or_allowed"}


def _find_repo_root_from_here(*, start: Path, max_hops: int = 10) -> Optional[Path]:
    """Best-effort monorepo root detection for local/dev runs."""
    cur = resolve_no_strict(start)
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
        return resolve_no_strict(Path(env.strip()).expanduser())

    here_dir = Path(__file__).resolve().parent
    guessed = _find_repo_root_from_here(start=here_dir)
    if guessed is not None:
        return guessed

    return resolve_no_strict(Path.cwd())


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
        out.append(resolve_no_strict(p))
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
        out.append(resolve_no_strict(p))
    return tuple(dict.fromkeys(out))


def _is_under(child: Path, parent: Path) -> bool:
    return is_under_path(child, parent)


def _mounts_from_allowed_paths(*, allowed_dirs: Iterable[Path], used_names: set[str]) -> Dict[str, Path]:
    """Build a deterministic {mount_name -> root} map for allowed roots outside workspace_root."""
    return build_workspace_mounts(allowed_dirs=allowed_dirs, used_names=used_names)


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
    mounts: Dict[str, Path] = {}
    if scope.access_mode == "workspace_or_allowed" and scope.allowed_paths:
        used: set[str] = set()
        allowed_outside = [p for p in scope.allowed_paths if isinstance(p, Path) and not _is_under(p, scope.root)]
        mounts = _mounts_from_allowed_paths(allowed_dirs=allowed_outside, used_names=used)
    try:
        resolved = resolve_canonical_workspace_path(
            base=scope.root,
            mounts=mounts,
            raw_path=raw,
            workspace_root_name=scope.root.name,
        )
    except WorkspacePathError as exc:
        if exc.kind == "path_escape":
            raise ValueError(f"Path escapes workspace_root: '{raw}'") from exc
        raise ValueError(str(exc)) from exc
    rel_part = resolved.resolved_path.relative_to(resolved.root_path).as_posix()
    return (resolved.root_path, rel_part)


def _ensure_allowed(*, path: Path, scope: "WorkspaceScope") -> None:
    for blocked in scope.ignored_paths:
        if _is_under(path, blocked) or resolve_no_strict(path) == resolve_no_strict(blocked):
            raise ValueError(f"Path is blocked by workspace_ignored_paths: '{path}'")


def _resolve_under_root_strict(*, root: Path, user_path: str) -> Path:
    """Resolve under root and ensure it doesn't escape (used for relative paths always)."""
    p = Path(str(user_path or "").strip()).expanduser()
    if p.is_absolute():
        raise ValueError("Internal error: strict under-root resolver received absolute path")
    resolved = resolve_no_strict(root / p)
    if not _is_under(resolved, root):
        raise ValueError(f"Path escapes workspace_root: '{user_path}'")
    return resolved


# Bounded suffix scan for re-anchoring (µs-class; only runs on the refusal path).
_REANCHOR_MAX_SUFFIXES = 16


def _reanchor_absolute(*, scope: "WorkspaceScope", resolved: Path, roots: Tuple[Path, ...]) -> Optional[Path]:
    """Re-anchor an absolute path that FAILED containment onto a workspace root.

    Models frequently fabricate a plausible-but-wrong absolute PREFIX for a
    file that genuinely lives inside the workspace (live incident 2026-07-12:
    `/Users/x/projects/mnemosyne/...` named while the real tree was
    `<root>/mnemosyne/...` — the relative retry succeeded, the absolute form
    refused). Rule (adversarially reviewed):

    - Path EXISTS on disk: substitution is forbidden UNLESS identity is
      provable — a candidate under a root must be the SAME FILE (inode) as the
      named path (covers case-insensitive-filesystem aliases; suffix depth ≥1
      is fine because samefile proves identity).
    - Path does NOT exist: fabricated-prefix recovery — try suffixes of the
      named path under each root, LONGEST suffix first (most of the model's
      stated intent), root before mounts on ties; the candidate must EXIST and
      suffix depth must be ≥2 (a basename-only match is no evidence of shared
      identity). Writes to new files deliberately never re-anchor (the
      resolver is tool-agnostic; a parent-exists rule would create files the
      model never named).
    - Every candidate is re-resolved and containment-rechecked (kills
      symlink-out) and ignored-path candidates are skipped silently (no
      blacklist oracle).

    Returns the re-anchored path, or None (caller refuses with the unified
    error — one string for both branches, so refusals never become a
    filesystem-existence oracle for outside paths).
    """
    parts = resolved.parts
    if len(parts) < 2:
        return None

    try:
        named_exists = resolved.exists()
    except Exception:
        named_exists = False

    min_depth = 1 if named_exists else 2
    max_suffix = len(parts) - 1  # never re-join the full anchor'd path
    suffix_lengths = [k for k in range(max_suffix, min_depth - 1, -1)][: _REANCHOR_MAX_SUFFIXES]

    for k in suffix_lengths:
        suffix = Path(*parts[-k:])
        for root in roots:
            try:
                candidate = resolve_no_strict(root / suffix)
            except Exception:
                continue
            if not _is_under(candidate, root):
                continue  # symlink-out or .. games — skip, keep scanning
            try:
                if not candidate.exists():
                    continue
            except Exception:
                continue
            # Ignored candidates are skipped silently (no blacklist oracle);
            # _ensure_allowed backstops after return.
            blocked = False
            for ign in scope.ignored_paths:
                if _is_under(candidate, ign) or resolve_no_strict(candidate) == resolve_no_strict(ign):
                    blocked = True
                    break
            if blocked:
                continue
            if named_exists:
                # Identity required: the named path is a REAL file elsewhere;
                # only accept the candidate when it IS that file (same inode —
                # case-alias/hardlink), never a lookalike substitution.
                try:
                    if not os.path.samefile(str(candidate), str(resolved)):
                        continue
                except Exception:
                    continue
            _logger.warning(
                "workspace re-anchor: '%s' -> '%s' (fabricated/mis-cased absolute prefix)",
                str(resolved),
                str(candidate),
            )
            return candidate
    return None


_REANCHOR_TEACHING_SUFFIX = (
    " — if you meant a file inside the workspace, retry with a path relative to '{root}' "
    "(absolute paths must stay under it; the host can grant outside directories)."
)


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
        resolved = resolve_no_strict(p)
        if scope.access_mode == "workspace_only":
            if not _is_under(resolved, scope.root):
                reanchored = _reanchor_absolute(scope=scope, resolved=resolved, roots=(scope.root,))
                if reanchored is None:
                    raise ValueError(
                        f"Path escapes workspace_root: '{user_path}'"
                        + _REANCHOR_TEACHING_SUFFIX.format(root=scope.root)
                    )
                resolved = reanchored
        elif scope.access_mode == "workspace_or_allowed":
            if not _is_under(resolved, scope.root) and not any(_is_under(resolved, p) for p in scope.allowed_paths):
                roots = (scope.root, *tuple(scope.allowed_paths))
                reanchored = _reanchor_absolute(scope=scope, resolved=resolved, roots=roots)
                if reanchored is None:
                    raise ValueError(
                        f"Path is outside workspace roots: '{user_path}'"
                        + _REANCHOR_TEACHING_SUFFIX.format(root=scope.root)
                    )
                resolved = reanchored
        _ensure_allowed(path=resolved, scope=scope)
        return resolved

    # Relative paths normally resolve under workspace_root, but we also support a
    # conservative "mount/..." convention for allowed roots (mirrors gateway file endpoints).
    root_used, rel_part = _resolve_virtual_mount_relative_path(scope=scope, raw=raw)
    resolved = _resolve_under_root_strict(root=root_used, user_path=rel_part)
    _ensure_allowed(path=resolved, scope=scope)
    return resolved


def resolve_user_workspace_path(*, scope: "WorkspaceScope", user_path: str) -> WorkspacePathResolution:
    """Resolve a user path and return its canonical workspace-path representation."""
    raw = str(user_path or "").strip()
    if not raw:
        raise ValueError("Empty path")

    if raw.startswith("@"):
        raw = raw[1:].lstrip()

    mounts: Dict[str, Path] = {}
    if scope.allowed_paths:
        used: set[str] = set()
        allowed_outside = [p for p in scope.allowed_paths if isinstance(p, Path) and not _is_under(p, scope.root)]
        mounts = _mounts_from_allowed_paths(allowed_dirs=allowed_outside, used_names=used)

    if scope.access_mode == "all_except_ignored":
        try:
            resolved = resolve_canonical_workspace_path(
                base=scope.root,
                mounts=mounts,
                raw_path=raw,
                workspace_root_name=scope.root.name,
            )
        except WorkspacePathError as exc:
            if exc.kind == "path_escape":
                raise ValueError(f"Path escapes workspace_root: '{user_path}'") from exc
            if exc.kind != "outside_workspace_roots":
                raise ValueError(str(exc)) from exc
            resolved_path = resolve_user_path(scope=scope, user_path=raw)
            return WorkspacePathResolution(
                resolved_path=resolved_path,
                virtual_path=str(resolved_path),
                mount_name=None,
                root_path=resolved_path.parent,
            )
        _ensure_allowed(path=resolved.resolved_path, scope=scope)
        return resolved

    try:
        resolved = resolve_canonical_workspace_path(
            base=scope.root,
            mounts=mounts,
            raw_path=raw,
            workspace_root_name=scope.root.name,
        )
    except WorkspacePathError as exc:
        if exc.kind == "path_escape":
            raise ValueError(f"Path escapes workspace_root: '{user_path}'") from exc
        raise ValueError(str(exc)) from exc
    _ensure_allowed(path=resolved.resolved_path, scope=scope)
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
        root = resolve_no_strict(root)
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
            out[field] = str(resolve_no_strict(root))
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
    if tool_name == "shell_exec":
        # Pins the INITIAL cwd of a persistent shell session (backlog 0220). Like
        # execute_command, this is policy for the starting point, not a sandbox: once
        # running, the session can `cd` anywhere (stated in the tool schema).
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
    if tool_name == "browser_probe":
        # Core's render-verification tool (c4872 note 1): the arg is named
        # `target` and carries EITHER a URL or a local file path — a new
        # spelling this rewriter did not cover, so local-file probes
        # bypassed the workspace wall. URLs pass through untouched (egress
        # policy is the approval lane's, not the wall's); file:// URLs and
        # bare paths wall exactly like read_file's file_path.
        raw_target = out.get("target")
        if isinstance(raw_target, str) and raw_target.strip():
            t = raw_target.strip()
            lowered = t.lower()
            if lowered.startswith(("http://", "https://")):
                pass  # network target: not the wall's jurisdiction
            elif lowered.startswith("file://"):
                inner = t[len("file://"):]
                resolved = resolve_user_path(scope=scope, user_path=inner)
                out["target"] = f"file://{resolved}"
            else:
                out["target"] = str(resolve_user_path(scope=scope, user_path=t))
        return out

    return out


__all__ = [
    "WorkspaceAccessMode",
    "WorkspaceScope",
    "WorkspaceScopedToolExecutor",
    "rewrite_tool_arguments",
    "resolve_workspace_base_dir",
    "resolve_user_path",
    "resolve_user_workspace_path",
]
