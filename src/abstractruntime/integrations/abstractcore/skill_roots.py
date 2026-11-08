"""Files of a run's activated skills are always readable (operator ruling 2026-11-08).

A skill the run was GIVEN must be reachable by the model that was told about
it. AbstractCore's ``read_file`` applies the workspace ``.abstractignore`` and
the workspace scope, which prune ``untracked/`` and the gateway data folder —
exactly where a skills shelf lives — so an activated skill's SKILL.md came
back as ``Error: File '…' is ignored by .abstractignore policy``. The runtime
therefore answers a ``read_file`` whose path lies inside one of the run's
SKILL FOLDERS itself: the whole file, UTF-8, in the line-numbered shape core
renders, ``start_line``/``end_line`` honoured — before core's tool (and its
policies) is ever asked. Any other path reaches core unchanged.

Skill folders derive from the run's own record, never from the model's words:

- ``vars["skills"]`` as a mapping (the workforce payload shape,
  ``abstractgateway.skills_union.resolve_backlog_skills``): roots ``shelf``,
  ``imported_shelf``, ``roots`` (a list or a ``os.pathsep``-joined string) and
  ``ABSTRACTCODE_SKILLS_ROOTS`` (joined the same way; also under ``env``),
  names ``active``;
- ``vars["skills"]`` as a list of names (the run-start selection a gateway
  resolves): roots and the ``active`` names come from
  ``vars["_runtime"]["skills_resolution"]`` (``shelf``, ``imported_shelf``,
  ``roots``) when the host recorded them; without a resolution the listed
  names stand;
- ``vars["_runtime"]["skills_block"]``: the ``<path>`` each ``<skill>`` carries
  (AbstractSkill 0.3.11 renders the SKILL.md the loader read) names its
  folder directly.

A folder qualifies as ``<root>/<name>`` or ``<root>/skills/<name>`` only when
it holds a ``SKILL.md``. A requested path qualifies when it lies lexically
under a skill folder (as spelled, or as its real path); it is then resolved
with ``os.path.realpath`` and must still lie under that folder's real path —
a symlink leading out is REFUSED with a typed error, never served and never
handed to core.
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

SKILL_FILENAME = "SKILL.md"
ROOTS_ENV_KEY = "ABSTRACTCODE_SKILLS_ROOTS"
_PATH_RE = re.compile(r"<path>(.*?)</path>", re.DOTALL)

# (served?, output, error): the shape of one tool result's payload half.
SkillRead = Tuple[bool, Optional[str], Optional[str]]


def _as_names(raw: Any) -> List[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        # A pin ``name@tree_hash`` names the same folder.
        name = item.strip().split("@", 1)[0].strip()
        if not name or name in {".", ".."} or "/" in name or "\\" in name or os.sep in name:
            continue
        if name not in out:
            out.append(name)
    return out


def _as_roots(raw: Any) -> List[str]:
    """A root value: one path, a ``os.pathsep``-joined string of paths, or a list of either."""
    if raw is None or isinstance(raw, bool):
        return []
    items: List[Any] = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    out: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        for piece in item.split(os.pathsep):
            p = piece.strip()
            if p:
                out.append(p)
    return out


def _roots_and_active(block: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    roots: List[str] = []
    for key in ("shelf", "imported_shelf", "roots", ROOTS_ENV_KEY):
        roots.extend(_as_roots(block.get(key)))
    env = block.get("env")
    if isinstance(env, Mapping):
        roots.extend(_as_roots(env.get(ROOTS_ENV_KEY)))
    return roots, _as_names(block.get("active"))


def _folders_from_skills_block(block: Any) -> List[Path]:
    if not isinstance(block, str) or "<path>" not in block:
        return []
    out: List[Path] = []
    for match in _PATH_RE.finditer(block):
        raw = html.unescape(match.group(1)).strip()
        if not raw:
            continue
        p = Path(os.path.expanduser(raw))
        out.append(p.parent if p.name == SKILL_FILENAME else p)
    return out


def skill_folders_for_run_vars(vars: Optional[Mapping[str, Any]]) -> List[Path]:
    """The skill folders this run was given, as absolute normalized paths that hold a SKILL.md."""
    if not isinstance(vars, Mapping):
        return []
    roots: List[str] = []
    names: List[str] = []
    folders: List[Path] = []

    skills = vars.get("skills")
    rt = vars.get("_runtime")
    resolution = rt.get("skills_resolution") if isinstance(rt, Mapping) else None

    if isinstance(skills, Mapping):
        r, n = _roots_and_active(skills)
        roots.extend(r)
        names.extend(n)
    elif isinstance(skills, (list, tuple, str)):
        if isinstance(resolution, Mapping) and "active" in resolution:
            names.extend(_as_names(resolution.get("active")))
        else:
            names.extend(_as_names(skills))
    if isinstance(resolution, Mapping):
        r, n = _roots_and_active(resolution)
        roots.extend(r)
        names.extend(x for x in n if x not in names)
    if isinstance(rt, Mapping):
        folders.extend(_folders_from_skills_block(rt.get("skills_block")))

    for root in roots:
        base = Path(os.path.expanduser(root))
        if not base.is_absolute():
            continue
        for name in names:
            folders.append(base / name)
            folders.append(base / "skills" / name)

    out: List[Path] = []
    seen: set = set()
    for folder in folders:
        try:
            if not folder.is_absolute() or not (folder / SKILL_FILENAME).is_file():
                continue
        except OSError:
            continue
        key = os.path.normpath(str(folder))
        if key in seen:
            continue
        seen.add(key)
        out.append(Path(key))
    return out


def _under(path: str, base: str) -> bool:
    base = base.rstrip(os.sep) or os.sep
    return path.startswith(base + os.sep)


def _int_or_none(raw: Any) -> Optional[int]:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _numbered(lines: List[str], first: int) -> str:
    last = first + len(lines) - 1 if lines else first
    width = max(1, len(str(max(1, last))))
    return "\n".join(f"{first + i:>{width}}: {text}" for i, text in enumerate(lines))


def skill_file_read(vars: Optional[Mapping[str, Any]], arguments: Mapping[str, Any]) -> Optional[SkillRead]:
    """Answer a ``read_file`` on a path inside one of the run's skill folders.

    Returns ``None`` when the path is not under a skill folder (core answers as
    before); otherwise ``(True, output, None)`` with the file in core's
    line-numbered shape, or ``(False, None, error)`` for a path that resolves
    out of its folder, a missing file, a directory, a binary file or an
    invalid line range.
    """
    raw = (
        arguments.get("file_path")
        or arguments.get("path")
        or arguments.get("filename")
        or arguments.get("file")
    )
    if not isinstance(raw, str) or not raw.strip():
        return None
    folders = skill_folders_for_run_vars(vars)
    if not folders:
        return None
    requested = raw.strip()
    expanded = os.path.expanduser(requested)
    if not os.path.isabs(expanded):
        return None
    lexical = os.path.normpath(expanded)

    owner: Optional[str] = None
    for folder in folders:
        spelled = os.path.normpath(str(folder))
        real_folder = os.path.realpath(spelled)
        if _under(lexical, spelled) or _under(lexical, real_folder):
            owner = real_folder
            break
    if owner is None:
        return None

    real = os.path.realpath(lexical)
    if not _under(real, owner):
        return (
            False,
            None,
            f"Error: File '{requested}' resolves outside its skill folder ({owner}) and was refused",
        )
    if os.path.isdir(real):
        return (False, None, f"Error: '{requested}' is a directory, not a file")
    if not os.path.isfile(real):
        return (False, None, f"Error: File '{requested}' does not exist")
    try:
        with open(real, "r", encoding="utf-8", newline="") as fh:
            text = fh.read()
    except UnicodeDecodeError:
        return (False, None, f"Error: Cannot read '{requested}' - file appears to be binary")
    except OSError as exc:
        return (False, None, f"Error: Cannot read '{requested}': {exc}")

    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines = [ln[:-1] if ln.endswith("\r") else ln for ln in lines]
    total = len(lines)

    start_raw = arguments.get("start_line") or arguments.get("startLine") or arguments.get("start_line_one_indexed")
    end_raw = (
        arguments.get("end_line")
        or arguments.get("endLine")
        or arguments.get("end_line_one_indexed_inclusive")
        or arguments.get("end_line_one_indexed")
    )
    start = _int_or_none(start_raw)
    end = _int_or_none(end_raw)
    if start_raw is not None and start is None:
        return (False, None, f"Error: start_line must be an integer (got {start_raw})")
    if end_raw is not None and end is None:
        return (False, None, f"Error: end_line must be an integer (got {end_raw})")
    if start is None:
        start = 1
    if start < 1:
        return (False, None, f"Error: start_line must be >= 1 (got {start})")
    if end is not None and end < 1:
        return (False, None, f"Error: end_line must be >= 1 (got {end})")
    if end is not None and start > end:
        return (False, None, f"Error: start_line ({start}) cannot be greater than end_line ({end})")

    if start == 1 and end is None:
        return (True, f"File: {requested} ({total} lines)\n\n{_numbered(lines, 1)}", None)
    if start > total:
        return (False, None, f"Error: Start line {start} exceeds file length ({total} lines)")
    stop = total if end is None else min(end, total)
    selected = lines[start - 1 : stop]
    return (True, f"File: {requested} ({len(selected)} lines)\n\n{_numbered(selected, start)}", None)
