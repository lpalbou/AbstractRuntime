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

WHERE a skill folder may be is the HOST's word, never the run's (review
2026-11-09 finding 1): run vars are client-writable, so folders taken from a
``skills`` mapping's roots, ``_runtime.skills_resolution`` or the ``<path>``
of ``_runtime.skills_block`` let any client read every file beside any
``SKILL.md`` on the machine. The host registers its skill roots once per
process (``register_skill_roots``; AbstractGateway registers its curated shelf
and its imported folder when it builds a service). With NO root registered,
nothing is served here and every path goes to core (fail closed).

WHICH skills under those roots is the run's selection, by NAME only:

- ``vars["_runtime"]["skills_resolution"]["active"]`` when the host recorded a
  resolution (the trust-gated outcome; held and blocked names are not in it);
- otherwise ``vars["skills"]``: a list of names, or the ``active`` names of a
  mapping;
- plus the ``<name>`` of each ``<skill>`` in ``vars["_runtime"]["skills_block"]``
  (the block rides alone into Agent-node and subflow child runs).

A folder qualifies as ``<root>/<name>`` or ``<root>/skills/<name>`` of a
REGISTERED root, only when it holds a ``SKILL.md``; a name is one path segment
(never ``/``, ``.`` or ``..``). A requested path qualifies when it lies
lexically under such a folder (as spelled, or as its real path); it is then
resolved with ``os.path.realpath`` and must still lie under that folder's real
path — a symlink leading out is REFUSED with a typed error, never served and
never handed to core.
"""

from __future__ import annotations

import html
import os
import re
import threading
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Tuple, Union

SKILL_FILENAME = "SKILL.md"
_NAME_RE = re.compile(r"<name>(.*?)</name>", re.DOTALL)

# The host's skill roots (absolute, normalized), registered once per process.
_ROOTS_LOCK = threading.Lock()
_REGISTERED_ROOTS: List[str] = []

# (served?, output, error): the shape of one tool result's payload half.
SkillRead = Tuple[bool, Optional[str], Optional[str]]


def register_skill_roots(roots: Iterable[Union[str, "os.PathLike[str]"]]) -> List[str]:
    """Add the HOST's skill roots for this process: folders whose ``<name>/SKILL.md`` or
    ``skills/<name>/SKILL.md`` are skills. Only absolute paths count; a root need not exist yet
    (a shelf seeded later is served once it holds the skill). Returns every registered root.

    Only a host calls this, from its own configuration — never from run vars."""
    added: List[str] = []
    for raw in roots or ():
        try:
            text = os.fspath(raw)
        except TypeError:
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        expanded = os.path.expanduser(text.strip())
        if not os.path.isabs(expanded):
            continue
        added.append(os.path.normpath(expanded))
    with _ROOTS_LOCK:
        for root in added:
            if root not in _REGISTERED_ROOTS:
                _REGISTERED_ROOTS.append(root)
        return list(_REGISTERED_ROOTS)


def registered_skill_roots() -> List[str]:
    """The skill roots the host registered in this process (empty = nothing is served)."""
    with _ROOTS_LOCK:
        return list(_REGISTERED_ROOTS)


def clear_skill_roots() -> None:
    """Forget every registered root (tests; a host rebuilding its configuration)."""
    with _ROOTS_LOCK:
        _REGISTERED_ROOTS.clear()


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


def _names_in_block(block: Any) -> List[str]:
    if not isinstance(block, str) or "<name>" not in block:
        return []
    return _as_names([html.unescape(m.group(1)).strip() for m in _NAME_RE.finditer(block)])


def selected_skill_names(vars: Optional[Mapping[str, Any]]) -> List[str]:
    """The skill NAMES this run selected (see the module doc); never a path."""
    if not isinstance(vars, Mapping):
        return []
    names: List[str] = []
    skills = vars.get("skills")
    rt = vars.get("_runtime")
    resolution = rt.get("skills_resolution") if isinstance(rt, Mapping) else None
    if isinstance(resolution, Mapping) and "active" in resolution:
        names.extend(_as_names(resolution.get("active")))
    elif isinstance(skills, Mapping):
        names.extend(_as_names(skills.get("active")))
    elif isinstance(skills, (list, tuple, str)):
        names.extend(_as_names(skills))
    if isinstance(rt, Mapping):
        names.extend(n for n in _names_in_block(rt.get("skills_block")) if n not in names)
    return names


def skill_folders_for_run_vars(vars: Optional[Mapping[str, Any]]) -> List[Path]:
    """The skill folders this run may read: ``<root>/<name>`` or ``<root>/skills/<name>`` of a
    REGISTERED root, holding a SKILL.md, for each name the run selected."""
    roots = registered_skill_roots()
    if not roots:
        return []
    names = selected_skill_names(vars)
    out: List[Path] = []
    seen: set = set()
    for root in roots:
        base = Path(root)
        for name in names:
            for folder in (base / name, base / "skills" / name):
                try:
                    if not (folder / SKILL_FILENAME).is_file():
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
