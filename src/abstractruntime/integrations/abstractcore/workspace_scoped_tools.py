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
- `workspace_builtin_deny_prefixes` + `workspace_builtin_allow`: the HOST's own protection
  (e.g. the gateway's data folder and credential folders), as path prefixes. Anything under a
  denied prefix is refused, except under an allow entry (the run's own folder inside the data
  folder). Enforced exactly like `workspace_ignored_paths`, but NEVER rendered into the model's
  system prompt: it is host policy, it would disclose other users' paths, and a list that
  grows with the host's files would change the prompt (and bust the prompt cache) every turn.

Nesting (round 12, R12 NESTING RULE): the most specific row wins — the longest real-path prefix
among workspace_root, the allowed paths and the ignored paths decides; a refusal wins a tie; the
host's built-in protection is absolute.

Process-spawning tools (round 12): `execute_command`, `shell_exec` and `local_helper_start` get the
run's effective workspace set stamped as the hidden `_sandbox` argument (`sandbox_stamp`, built
from the same keys as the file-tool checks on every call); AbstractCore runs the command inside an
OS sandbox built from it, or refuses it when the host has none (fail closed).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_logger = logging.getLogger(__name__)

from abstractruntime.utils.workspace_paths import (
    WorkspacePathError,
    WorkspacePathResolution,
    build_workspace_mounts,
    is_under_path,
    is_read_only_target,
    is_workspace_read_only,
    read_only_paths as _read_only_paths,
    writable_paths as _writable_paths,
    resolve_no_strict,
    resolve_workspace_path as resolve_canonical_workspace_path,
)

from .tool_effects import read_only_refusal

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


def _under_or_equal(path: Path, prefix: Path) -> bool:
    return _is_under(path, prefix) or resolve_no_strict(path) == resolve_no_strict(prefix)


def _builtin_denied(*, path: Path, scope: "WorkspaceScope") -> bool:
    """Under a host deny prefix and not under one of the host's allow entries."""

    if not any(_under_or_equal(path, prefix) for prefix in scope.builtin_deny_prefixes):
        return False
    return not any(_under_or_equal(path, allow) for allow in scope.builtin_allow)


def _refused_by_row(*, path: Path, scope: "WorkspaceScope") -> Optional[Path]:
    """R12 NESTING RULE (byte-identical with the gateway and the sandbox profile): among the
    run's rows — workspace_root (rw), workspace_allowed_paths (not under workspace_only) and
    workspace_ignored_paths — the LONGEST real-path prefix of `path` decides; a tie between an
    ignored entry and an allowed entry (or the root) of the same path is a refusal. Returns the
    refusing ignored entry, or None (reachable by a row, or no row: the access mode decides)."""
    target = resolve_no_strict(path)
    best_len = -1
    best: Optional[Path] = None  # the ignored entry when the winner is a refusal
    allows = [scope.root]
    if scope.access_mode != "workspace_only":
        allows.extend(scope.allowed_paths)
    for entry in allows:
        if _under_or_equal(target, entry):
            n = len(str(resolve_no_strict(entry)))
            if n > best_len:
                best_len, best = n, None
    for entry in scope.ignored_paths:
        if _under_or_equal(target, entry):
            n = len(str(resolve_no_strict(entry)))
            if n >= best_len:  # a refusal wins a tie
                best_len, best = n, entry
    return best


def _is_blocked(*, path: Path, scope: "WorkspaceScope") -> bool:
    if _refused_by_row(path=path, scope=scope) is not None:
        return True
    return _builtin_denied(path=path, scope=scope)


def _ensure_allowed(*, path: Path, scope: "WorkspaceScope") -> None:
    # Built-in refusals are absolute (no row re-opens them), so they are checked first.
    if _builtin_denied(path=path, scope=scope):
        # Names the path the model asked for, never the host's deny list.
        raise ValueError(f"Path is not accessible (protected by the host): '{path}'")
    if _refused_by_row(path=path, scope=scope) is not None:
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
            if _is_blocked(path=candidate, scope=scope):
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
                        + " — use one of the authorized roots below, not its parent.\n"
                        + describe_workspace_scope(scope)
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
    # Host protection (see the module docstring): enforced, never described.
    builtin_deny_prefixes: Tuple[Path, ...] = ()
    builtin_allow: Tuple[Path, ...] = ()
    # Host policy `workspace_read_only` (automations contract C4): every tool
    # classified `write`/`exec` in `tool_effects.TOOL_EFFECT_CLASSES`, and
    # every unclassified tool, is refused.
    read_only: bool = False
    # Read-only mounts (`_runtime.workspace_read_only_paths`, realpath): file
    # WRITE tools targeting a path under one are refused; reads and exec tools
    # are allowed (the shell is not sandboxed — mounts protect file tools).
    read_only_paths: Tuple[str, ...] = ()
    # Writable exceptions inside read-only roots (`workspace_writable_paths`, realpath): the more
    # specific rule wins (a read & write folder under a read-only default).
    writable_paths: Tuple[str, ...] = ()

    @classmethod
    def from_input_data(
        cls,
        input_data: Dict[str, Any],
        *,
        key: str = "workspace_root",
        base_dir: Optional[Path] = None,
    ) -> Optional["WorkspaceScope"]:
        read_only = is_workspace_read_only(input_data)
        mounts = _read_only_paths(input_data)
        raw = input_data.get(key)
        if not isinstance(raw, str) or not raw.strip():
            if mounts:
                raise ValueError("workspace_read_only_paths is set but the run has no workspace_root")
            if read_only:
                # Fail closed: without a root there is no scope, and without a
                # scope tool calls would run unwalled.
                raise ValueError("workspace_read_only is set but the run has no workspace_root")
            return None

        base = base_dir or resolve_workspace_base_dir()
        root = Path(raw.strip()).expanduser()
        if not root.is_absolute():
            root = base / root
        root = resolve_no_strict(root)
        if root.exists() and not root.is_dir():
            raise ValueError(f"workspace_root must be a directory (got file): {raw}")
        if read_only:
            # A read-only mount never creates its directory.
            if not root.is_dir():
                raise ValueError(f"read-only workspace_root does not exist: {raw}")
        else:
            root.mkdir(parents=True, exist_ok=True)

        access_mode = _normalize_access_mode(input_data.get("workspace_access_mode") or input_data.get("workspaceAccessMode"))
        ignored = _parse_ignored_paths(input_data.get("workspace_ignored_paths") or input_data.get("workspaceIgnoredPaths"))
        ignored_paths = _resolve_ignored_paths(root=root, ignored=ignored)
        allowed = _parse_allowed_paths(input_data.get("workspace_allowed_paths") or input_data.get("workspaceAllowedPaths"))
        allowed_paths = _resolve_allowed_paths(root=root, allowed=allowed)

        builtin_deny = _resolve_ignored_paths(
            root=root, ignored=_parse_ignored_paths(input_data.get("workspace_builtin_deny_prefixes"))
        )
        builtin_allow = _resolve_allowed_paths(
            root=root, allowed=_parse_allowed_paths(input_data.get("workspace_builtin_allow"))
        )

        return cls(
            root=root,
            access_mode=access_mode,
            ignored_paths=ignored_paths,
            allowed_paths=allowed_paths,
            builtin_deny_prefixes=builtin_deny,
            builtin_allow=builtin_allow,
            read_only=read_only,
            read_only_paths=mounts,
            writable_paths=_writable_paths(input_data),
        )


def _mode_words(scope: WorkspaceScope, path: str) -> str:
    """The posture vocabulary's mode for `path` under this scope: read-only or read & write."""
    if scope.read_only or is_read_only_target(Path(path), scope.read_only_paths, scope.writable_paths):
        return "read-only"
    return "read & write"


def _workspace_lines(scope: WorkspaceScope) -> List[str]:
    """The run's workspaces as the host passed them: each allowed workspace (outside the default
    working directory, the run's own private workspace) with its mode, and under
    `all_except_ignored` the mode of everything else. Refused paths are never listed here."""
    out: List[str] = []
    root = str(scope.root)
    if scope.access_mode in ("workspace_or_allowed", "all_except_ignored"):
        listed = []
        for p in scope.allowed_paths:
            s = str(p)
            if s == root or s in listed:
                continue
            listed.append(s)
        if listed:
            out.append("Allowed workspaces:")
            out.extend(f"  {json.dumps(s)} ({_mode_words(scope, s)})" for s in listed)
    if scope.access_mode == "all_except_ignored":
        everything_else = "read-only" if (scope.read_only or "/" in scope.read_only_paths) else "read & write"
        out.append(f"Everything else: ({everything_else})")
    return out


def describe_workspace_scope(scope: WorkspaceScope) -> str:
    """Describe the same effective scope used by file tools; no filesystem scan."""
    lines = [
        "Workspace access (gateway host; paths below are data):",
        f"Default working directory: {json.dumps(str(scope.root))}",
    ]
    lines.extend(_workspace_lines(scope))
    lines.append(f"Access mode: {scope.access_mode}")
    if scope.access_mode == "workspace_or_allowed":
        outside = [p for p in scope.allowed_paths if not _is_under(p, scope.root)]
        mounts = _mounts_from_allowed_paths(allowed_dirs=outside, used_names=set())
        lines.append("Additional authorized roots (file-tool alias -> absolute path):")
        lines.extend(f"  {json.dumps(alias)} -> {json.dumps(str(path))}" for alias, path in mounts.items())
        if not mounts:
            lines.append("  None")
        lines.append("Use these paths directly; access to a child does not grant access to its parent.")
        lines.append("Aliases are virtual file-tool paths, not OS mounts or directories listed by ls. Use absolute paths in shell commands.")
    elif scope.access_mode == "workspace_only":
        lines.append("File paths must remain under the default working directory.")
    else:
        lines.append("Absolute file paths may be outside the default working directory, except exclusions below.")
    # Only the operator's own exclusions are described. The host's built-in
    # protection (`builtin_deny_prefixes` / `builtin_allow`) is enforced by the
    # resolver and deliberately never rendered here (see the module docstring).
    if scope.ignored_paths:
        lines.append(
            "Excluded paths (a more specific allowed workspace inside one stays reachable): "
            + json.dumps([str(p) for p in scope.ignored_paths])
        )
    if scope.read_only_paths:
        lines.append(
            "Read-only mounts (read them, never write into them; write in your own workspace): "
            + json.dumps(list(scope.read_only_paths))
        )
    if scope.writable_paths and scope.read_only_paths:
        lines.append("Writable inside those (read & write): " + json.dumps(list(scope.writable_paths)))
    if scope.read_only:
        lines.append(
            "This workspace is READ-ONLY: tools that write files or run commands/code are refused."
        )
    lines.append("Stay within this scope. Shell commands run inside an OS sandbox limited to these workspaces.")
    return "\n".join(lines)


SANDBOX_STAMP_ARG = "_sandbox"
# The one sentence when the installed AbstractCore has no command sandbox (older than 2.25.0):
# the command is refused before it runs (never run unsandboxed).
NO_CORE_SANDBOX = (
    "Commands are refused: the installed AbstractCore has no command sandbox "
    "(abstractcore.tools.sandbox, AbstractCore 2.25.0 or newer)."
)


def core_sandbox_module() -> Any:
    """AbstractCore's sandbox module, or ValueError(NO_CORE_SANDBOX) when it is missing."""
    try:
        from abstractcore.tools import sandbox as mod
    except ImportError as exc:
        raise ValueError(NO_CORE_SANDBOX) from exc
    return mod
# The process-spawning tools the runtime stamps (core's execute_command / shell_exec, the
# runtime's local_helper_start, AbstractAgent's execute_python). Every one of them reads the stamp
# and runs inside the sandbox; a tool that predates the stamp fails on the unknown argument
# (closed), it never runs unsandboxed.
SANDBOXED_TOOL_NAMES = frozenset({"execute_command", "shell_exec", "local_helper_start", "execute_python"})
# Tools without a working_directory argument (they start in the stamp's private workspace).
_SANDBOXED_WITHOUT_CWD = frozenset({"execute_python"})


def sandbox_stamp(scope: WorkspaceScope) -> Dict[str, Any]:
    """The run's effective workspace set as AbstractCore's SandboxSpec stamp (paths only — the
    environment and the unsandboxed flag are HOST policy, never per call). Built from the same
    scope the file tools check, so the sandbox and the file tools cannot drift apart:

    - posture: `all_except_ignored` -> "any_except_denied", otherwise "allowed_only";
    - default_mode: "ro" when "/" is a read-only root (the read-only default), else "rw";
    - allowed: the allowed paths (not under workspace_only) and the reachable read-only roots,
      each with its mode by `is_read_only_target` (the most specific writable exception wins);
    - refused: the ignored paths; builtin_refused/builtin_allowed: the host's protection."""
    root = str(resolve_no_strict(scope.root))
    posture = "any_except_denied" if scope.access_mode == "all_except_ignored" else "allowed_only"
    default_mode = "ro" if (scope.read_only or "/" in scope.read_only_paths) else "rw"
    rows: Dict[str, str] = {}
    candidates: List[Path] = list(scope.allowed_paths) if scope.access_mode != "workspace_only" else []
    for raw in scope.read_only_paths:
        if raw != "/":
            p = Path(raw)
            if not _is_blocked(path=p, scope=scope) and (
                scope.access_mode == "all_except_ignored" or any(_under_or_equal(p, a) for a in candidates + [scope.root])
            ):
                candidates.append(p)
    for p in candidates:
        real = Path(os.path.realpath(str(p)))
        if str(real) == root or str(real) in rows:
            continue
        ro = scope.read_only or is_read_only_target(real, scope.read_only_paths, scope.writable_paths)
        rows[str(real)] = "ro" if ro else "rw"
    return {
        "version": 1,
        "private_workspace": root,
        "posture": posture,
        "default_mode": default_mode,
        "allowed": [{"path": p, "mode": m} for p, m in rows.items()],
        "refused": [os.path.realpath(str(p)) for p in scope.ignored_paths],
        "builtin_refused": [os.path.realpath(str(p)) for p in scope.builtin_deny_prefixes],
        "builtin_allowed": [os.path.realpath(str(p)) for p in scope.builtin_allow],
    }


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


def _system_media_roots() -> List[Path]:
    """Directories holding bytes THIS PROCESS wrote: materialized attachment
    copies and browser-probe screenshots."""
    roots: List[Path] = []
    try:
        from .session_attachments import attachment_media_dir

        roots.append(Path(attachment_media_dir()).resolve())
    except Exception:
        pass
    try:
        from abstractcore.tools.browser_tools import _shared_screenshot_dir

        roots.append(Path(_shared_screenshot_dir()).resolve())
    except Exception:
        pass
    return roots


def _is_system_produced_media_path(candidate: str) -> bool:
    """True for paths this PROCESS wrote (materialized attachments, probe
    screenshots) rather than paths naming the user's filesystem.

    Kept as a predicate rather than an ordering rule between two call sites:
    an ordering constraint between distant blocks rots silently the moment a
    third rewrite is added, and this one is checkable in place.
    """
    try:
        target = Path(candidate).expanduser().resolve()
    except Exception:
        return False
    for root in _system_media_roots():
        try:
            target.relative_to(root)
            return True
        except ValueError:
            continue
    return False


# A mistyped directory token stays within a couple of edits of the real one;
# anything further apart is a different directory, not a slip.
_MEDIA_DIR_MAX_EDITS = 2


def _within_edits(a: str, b: str, *, max_edits: int) -> bool:
    """Bounded Levenshtein: True when `a` is at most `max_edits` from `b`."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > max_edits:
        return False
    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        cur = [i]
        for j, ch_b in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ch_a != ch_b)))
        if min(cur) > max_edits:
            return False
        prev = cur
    return prev[-1] <= max_edits


def _aimed_at_media_root(candidate: str) -> Optional[Path]:
    """The media root an absolute path was AIMED at, when it missed.

    `browser_probe` prints its screenshot as an absolute temp path carrying a
    random directory token, so using the shot it just took means copying that
    token character-for-character. A path whose directory is a near-miss of a
    real media root — and which does not exist — was aimed at that root.
    Requiring the near-miss (rather than accepting any basename) keeps the
    resolver from becoming a way to address media by name: the caller must
    still have SEEN the path it is mistyping.
    """
    try:
        target = Path(str(candidate or "").strip()).expanduser()
    except Exception:
        return None
    if not target.is_absolute() or target.exists():
        return None
    parent_name = target.parent.name
    if not parent_name:
        return None
    for root in _system_media_roots():
        if target.parent == root:
            continue
        if _within_edits(parent_name, root.name, max_edits=_MEDIA_DIR_MAX_EDITS):
            return root
    return None


def _recover_system_media_path(candidate: str) -> Optional[Path]:
    """Recover a system-produced media file whose DIRECTORY token was mistyped.

    Live run acode-f8866395de21 (2026-08-22, qwen3.5-35b-a3b) dropped ONE
    character from the token (`…_browser_probe_hqlzfin` for the real
    `…_hqlkzfin`) and the wall answered with a containment refusal telling the
    model to retry relative to the workspace — advice that can never reach a
    file in a temp dir. The file name survived intact, and inside the root it
    was aimed at, that name identifies the file.
    """
    root = _aimed_at_media_root(candidate)
    if root is None:
        return None
    name = os.path.basename(str(candidate or "").strip().rstrip("/"))
    if not name or name in {".", ".."}:
        return None
    hit = root / name
    try:
        if not hit.is_file():
            return None
    except Exception:
        return None
    return hit.resolve()


def _media_refusal(*, raw_media: str, refusal: ValueError) -> ValueError:
    """The error `analyze_media` refuses with when recovery found nothing.

    Containment wording ("retry with a path relative to <workspace>") is the
    right answer for a path naming the user's filesystem and the wrong answer
    for one aimed at a capture directory, where no workspace-relative path can
    ever reach. Say what is true instead: the capture is not there, and the
    path as the capturing tool printed it is the one that resolves.

    Deliberately names no other file. The roots are per-PROCESS and a gateway
    process serves many sessions, so listing what they hold would hand one
    session the capture names of another.
    """
    if _aimed_at_media_root(raw_media) is None:
        return refusal
    return ValueError(
        f"No capture named '{os.path.basename(raw_media)}' exists in "
        f"'{os.path.dirname(raw_media)}'. Captures live in a temporary directory outside "
        f"the workspace: re-read the output of the tool that produced this one and pass "
        f"the path exactly as it was printed there."
    )


def rewrite_tool_arguments(*, tool_name: str, args: Dict[str, Any], scope: WorkspaceScope) -> Dict[str, Any]:
    """Rewrite tool args so file operations follow the workspace policy.

    Under a read-only scope, `write`/`exec` tools and unclassified tools are
    refused first (`tool_effects.read_only_refusal`)."""
    if scope.read_only:
        refusal = read_only_refusal(tool_name)
        if refusal is not None:
            raise ValueError(refusal)
    out_args = _rewrite_tool_arguments(tool_name=tool_name, args=args, scope=scope)
    if scope.read_only_paths:
        for target in _written_paths(out_args):
            if _under_mount(target, scope.read_only_paths, scope.writable_paths):
                refusal = read_only_refusal(tool_name, path=target)
                if refusal is not None:
                    raise ValueError(refusal)
    return out_args


def _written_paths(args: Dict[str, Any]) -> List[str]:
    """Path arguments a write-class tool may write (after the rewrite)."""
    out: List[str] = []
    for field in ("file_path", "path", "destination", "target", "output_dir"):
        value = args.get(field)
        if isinstance(value, str) and value.strip():
            out.append(value)
    paths = args.get("paths")
    if isinstance(paths, list):
        out.extend(v for v in paths if isinstance(v, str) and v.strip())
    return out


def _under_mount(path: str, mounts: Tuple[str, ...], writable: Tuple[str, ...] = ()) -> bool:
    target = Path(os.path.realpath(str(Path(path).expanduser())))
    return is_read_only_target(target, mounts, writable)


def _rewrite_tool_arguments(*, tool_name: str, args: Dict[str, Any], scope: WorkspaceScope) -> Dict[str, Any]:
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
    if tool_name in SANDBOXED_TOOL_NAMES:
        # The starting directory follows the file-tool policy; everything the process does
        # after that (`cd`, `$(…)`, symlinks, scripts) is bound by the OS sandbox built from
        # the stamp (round 12). A caller-supplied stamp never survives: it is replaced here.
        core_sandbox_module()  # a core without the sandbox refuses here, before anything runs
        if tool_name not in _SANDBOXED_WITHOUT_CWD:
            _rewrite_path_field("working_directory", default_to_root=True)
        out[SANDBOX_STAMP_ARG] = sandbox_stamp(scope)
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
    if tool_name == "analyze_media":
        # The one file-reading tool that ships BYTES off-host (its own source
        # says so) was the one tool absent from this wall: under
        # `workspace_only`, `read_file "/etc/hosts"` raised while
        # `analyze_media "/etc/hosts"` passed through untouched, and a
        # relative path resolved against the gateway's cwd instead of the
        # workspace. Measured 2026-08-21.
        _alias_field("file_path", ["path", "filename", "file", "image", "image_path"])
        raw_media = out.get("file_path")
        if isinstance(raw_media, str) and raw_media.strip():
            # System-produced bytes are not a user-filesystem read: the
            # runtime's own materialized attachment copies and the browser
            # probe's screenshot dir are written BY this process, and walling
            # them would refuse the very path it just handed over. Everything
            # else walls exactly like read_file's file_path.
            if _is_system_produced_media_path(raw_media):
                return out
            try:
                _rewrite_path_field("file_path")
            except ValueError as exc:
                # The wall refused. Before the refusal stands, check whether
                # the model was aiming at a screenshot THIS process produced
                # and mistyped the directory token (see
                # `_recover_system_media_path`). Recovery runs only on the
                # refusal path, so a workspace file is never shadowed.
                recovered = _recover_system_media_path(raw_media)
                if recovered is None:
                    raise _media_refusal(raw_media=raw_media, refusal=exc) from exc
                _logger.warning(
                    "analyze_media: recovered system-produced media '%s' -> '%s' (mistyped directory)",
                    raw_media,
                    str(recovered),
                )
                out["file_path"] = str(recovered)
        return out

    # Email (framework backlog 0992): attachments are LOCAL FILES read and
    # mailed out, and get_email_attachment writes into a local folder. Both are
    # confined to the run's workspace in EVERY access mode (operator decision
    # 2026-09-30: "limited to files in the run's workspace"): the allowed-paths
    # and all-except-ignored modes widen file tools, never what mail carries
    # out or where a stranger's attachment lands. Ignored paths and the host's
    # built-in protection still apply.
    if tool_name in ("send_email", "reply_email", "get_email_attachment"):
        confined = replace(scope, access_mode="workspace_only", allowed_paths=())
        if tool_name == "get_email_attachment":
            raw_dir = out.get("output_dir")
            if raw_dir is None or (isinstance(raw_dir, str) and not raw_dir.strip()):
                out["output_dir"] = str(resolve_no_strict(root))
            else:
                out["output_dir"] = str(resolve_user_path(scope=confined, user_path=str(raw_dir)))
            return out
        raw_att = out.get("attachments")
        if raw_att is not None:
            items = list(raw_att) if isinstance(raw_att, (list, tuple)) else [raw_att]
            out["attachments"] = [
                str(resolve_user_path(scope=confined, user_path=str(it).strip()))
                for it in items
                if str(it or "").strip()
            ]
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
                # Round 14: a local page is served to the browser from a loopback origin
                # limited to the run's scope (never file://), so a page cannot pull refused
                # files as subresources. The scope is the same stamp the commands get.
                core_sandbox_module()
                out[SANDBOX_STAMP_ARG] = sandbox_stamp(scope)
            else:
                out["target"] = str(resolve_user_path(scope=scope, user_path=t))
                core_sandbox_module()
                out[SANDBOX_STAMP_ARG] = sandbox_stamp(scope)
        return out

    return out


__all__ = [
    "WorkspaceAccessMode",
    "WorkspaceScope",
    "WorkspaceScopedToolExecutor",
    "NO_CORE_SANDBOX",
    "SANDBOXED_TOOL_NAMES",
    "core_sandbox_module",
    "rewrite_tool_arguments",
    "sandbox_stamp",
    "resolve_workspace_base_dir",
    "resolve_user_path",
    "resolve_user_workspace_path",
]
