"""A tool that names a path must ride the workspace wall — or say so here.

`rewrite_tool_arguments` dispatches by tool NAME through an if-chain whose
fall-through is `return out` unchanged. That miss case is silent and
success-shaped, so a tool joins the wall only if someone remembers. It has
already been forgotten twice:

  * `browser_probe` — fixed after local-file probes were found bypassing the
    wall entirely ("a new spelling this rewriter did not cover", the comment
    still sits in the branch);
  * `analyze_media` — found unwalled on 2026-08-21, on the one file-reading
    tool whose own source says it "sends image bytes to the configured vision
    route (possibly a remote provider)".

This test is the mechanism those two fixes lacked. It derives path-shaped
parameters from the LIVE builtin inventory, so a new tool with a `file_path`
(or `path`, `directory_path`, `target`, …) fails here until someone decides,
in writing, which list it belongs to. Deciding is cheap; the default was the
expensive part.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from abstractcore.tools.inventory import list_builtin_tool_inventory
from abstractruntime.integrations.abstractcore.workspace_scoped_tools import (
    WorkspaceScope,
    rewrite_tool_arguments,
)

#: (tool, field) pairs that name something on the filesystem and MUST wall.
WALLED: set[tuple[str, str]] = {
    ("analyze_code", "file_path"),
    ("analyze_media", "file_path"),
    ("browser_probe", "target"),
    ("edit_file", "file_path"),
    ("execute_command", "working_directory"),
    ("list_files", "directory_path"),
    ("read_file", "file_path"),
    ("search_files", "path"),
    ("shell_exec", "working_directory"),
    ("skim_files", "paths"),
    ("skim_folders", "paths"),
    ("write_file", "file_path"),
}

#: (tool, field) pairs whose NAME looks path-shaped but which address nothing
#: on this filesystem. Each needs a reason, because "not a path" is a claim.
NOT_A_PATH: dict[tuple[str, str], str] = {
    ("list_whatsapp_messages", "direction"): "message direction (in/out), not a directory",
    ("send_telegram_artifact", "filename"): "display name for the upload; the bytes come from the artifact id",
    ("send_telegram_artifact", "artifact_base_dir_env_var"): "the NAME of an env var, and the id it joins is regex-gated to [A-Za-z0-9_-]",
    ("search_files", "file_pattern"): "a glob matched against names, never opened",
    ("search_files", "ignore_dirs"): "directory NAMES to skip, matched not opened",
    ("skim_folders", "file_pattern"): "a glob matched against names, never opened",
    ("skim_files", "target_percent"): "a percentage; the substring 'target' is a coincidence",
}

_PATH_SHAPED = ("path", "file", "dir", "folder", "target")


def _candidates() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for tool in list_builtin_tool_inventory():
        for field in (tool.parameters or {}):
            if any(token in field.lower() for token in _PATH_SHAPED):
                found.add((tool.name, field))
    return found


def test_every_path_shaped_parameter_is_classified():
    """The loud miss: a new tool cannot arrive without a decision."""
    undeclared = sorted(_candidates() - set(WALLED) - set(NOT_A_PATH))
    assert not undeclared, (
        "path-shaped parameter(s) with no decision: "
        + ", ".join(f"{t}.{f}" for t, f in undeclared)
        + " — add each to WALLED (and give it a branch in rewrite_tool_arguments) "
        "or to NOT_A_PATH with the reason it addresses nothing on disk."
    )


def test_no_stale_classifications():
    """An entry naming no live parameter would silently pre-classify a future one."""
    stale = sorted((set(WALLED) | set(NOT_A_PATH)) - _candidates())
    assert not stale, "stale entries: " + ", ".join(f"{t}.{f}" for t, f in stale)


@pytest.mark.parametrize("tool,field", sorted(WALLED))
def test_declared_path_arguments_refuse_to_escape(tool: str, field: str, tmp_path: Path):
    """The check that would have caught analyze_media."""
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    scope = WorkspaceScope.from_input_data({"workspace_root": str(ws)})
    assert scope is not None

    outside = "/etc/hosts"
    args = {field: [outside] if field == "paths" else outside}
    with pytest.raises(ValueError):
        rewrite_tool_arguments(tool_name=tool, args=args, scope=scope)
