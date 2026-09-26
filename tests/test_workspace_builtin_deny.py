"""The host's built-in workspace protection is enforced, never described.

The gateway sends its own rule as `workspace_builtin_deny_prefixes` (its data
folder, credential folders) plus `workspace_builtin_allow` (the run's own
workspace folder inside the data folder), next to the operator's
`workspace_ignored_paths`. Pinned here (REVIEW/16, 2026-09-26: enumerating the
data folder into the system prompt grew it ~950 tokens per turn and busted the
prompt cache every turn):

* every file tool, listing and shell cwd is refused under a denied prefix,
  except under the allow entry, which wins inside the denied prefix;
* the operator's own ignored paths still refuse even inside the allow entry;
* the system prompt is byte-identical with and without the built-in entries,
  however many entries (or files in the data folder) there are;
* child runs inherit the built-in entries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from abstractruntime.integrations.abstractcore.workspace_scoped_tools import (
    WorkspaceScope,
    describe_workspace_scope,
    rewrite_tool_arguments,
)


@pytest.fixture()
def layout(tmp_path: Path) -> Dict[str, Path]:
    data = tmp_path / "gateway-data"
    own = data / "workspaces" / "session-mine"
    other = data / "workspaces" / "session-other"
    creds = tmp_path / "home" / ".ssh"
    for d in (own, other, creds):
        d.mkdir(parents=True)
    (data / "run_123.json").write_text("{}")
    (other / "secret.txt").write_text("theirs")
    (own / "notes.txt").write_text("mine")
    (creds / "id_ed25519").write_text("key")
    return {"data": data, "own": own, "other": other, "creds": creds, "tmp": tmp_path}


def _scope(layout: Dict[str, Path], *, builtin: bool = True, mode: str = "all_except_ignored", **extra: Any) -> WorkspaceScope:
    vars_: Dict[str, Any] = {"workspace_root": str(layout["own"]), "workspace_access_mode": mode}
    if builtin:
        vars_["workspace_builtin_deny_prefixes"] = [str(layout["data"]), str(layout["creds"])]
        vars_["workspace_builtin_allow"] = [str(layout["own"])]
    vars_.update(extra)
    scope = WorkspaceScope.from_input_data(vars_)
    assert scope is not None
    return scope


@pytest.mark.parametrize(
    "tool,field",
    [
        ("read_file", "file_path"),
        ("write_file", "file_path"),
        ("edit_file", "file_path"),
        ("list_files", "directory_path"),
        ("search_files", "path"),
        ("execute_command", "working_directory"),
        ("shell_exec", "working_directory"),
    ],
)
@pytest.mark.parametrize("target", ["data_file", "other_session", "data_root", "creds"])
def test_denied_prefixes_are_refused_for_every_tool(layout, tool, field, target) -> None:
    path = {
        "data_file": layout["data"] / "run_123.json",
        "other_session": layout["other"] / "secret.txt",
        "data_root": layout["data"],
        "creds": layout["creds"] / "id_ed25519",
    }[target]
    scope = _scope(layout)
    with pytest.raises(ValueError, match="protected by the host"):
        rewrite_tool_arguments(tool_name=tool, args={field: str(path)}, scope=scope)


def test_the_runs_own_folder_is_allowed_inside_the_denied_data_dir(layout) -> None:
    scope = _scope(layout)
    out = rewrite_tool_arguments(tool_name="read_file", args={"file_path": "notes.txt"}, scope=scope)
    assert Path(out["file_path"]) == (layout["own"] / "notes.txt").resolve()
    out = rewrite_tool_arguments(
        tool_name="read_file", args={"file_path": str(layout["own"] / "notes.txt")}, scope=scope
    )
    assert Path(out["file_path"]) == (layout["own"] / "notes.txt").resolve()
    # Default cwd is the own folder: allowed.
    out = rewrite_tool_arguments(tool_name="execute_command", args={"command": "ls"}, scope=scope)
    assert Path(out["working_directory"]) == layout["own"].resolve()


def test_relative_escape_into_the_data_dir_is_refused(layout) -> None:
    scope = _scope(layout)
    with pytest.raises(ValueError):
        rewrite_tool_arguments(tool_name="read_file", args={"file_path": "../session-other/secret.txt"}, scope=scope)


def test_without_the_builtin_entries_the_data_dir_is_reachable(layout) -> None:
    """Control arm: the refusal above comes from the built-in entries."""
    scope = _scope(layout, builtin=False)
    out = rewrite_tool_arguments(
        tool_name="read_file", args={"file_path": str(layout["other"] / "secret.txt")}, scope=scope
    )
    assert out["file_path"].endswith("secret.txt")


def test_operator_ignored_paths_still_win_inside_the_allow_entry(layout) -> None:
    scope = _scope(layout, workspace_ignored_paths=[str(layout["own"] / "notes.txt")])
    with pytest.raises(ValueError, match="workspace_ignored_paths"):
        rewrite_tool_arguments(tool_name="read_file", args={"file_path": "notes.txt"}, scope=scope)


def test_workspace_or_allowed_grants_do_not_override_the_builtin_deny(layout) -> None:
    scope = _scope(layout, mode="workspace_or_allowed", workspace_allowed_paths=[str(layout["data"])])
    with pytest.raises(ValueError, match="protected by the host"):
        rewrite_tool_arguments(
            tool_name="read_file", args={"file_path": str(layout["other"] / "secret.txt")}, scope=scope
        )


@pytest.mark.parametrize("mode", ["workspace_only", "all_except_ignored", "workspace_or_allowed"])
def test_system_prompt_is_byte_identical_with_and_without_builtin_entries(layout, mode) -> None:
    operator = {"workspace_ignored_paths": [str(layout["tmp"] / "operator-excluded")]}
    without = describe_workspace_scope(_scope(layout, builtin=False, mode=mode, **operator))
    with_builtin = describe_workspace_scope(_scope(layout, builtin=True, mode=mode, **operator))
    assert with_builtin == without
    # The operator's own exclusion is still rendered, the host's never.
    assert "operator-excluded" in with_builtin
    excluded = [ln for ln in with_builtin.splitlines() if ln.startswith("Excluded paths")]
    assert len(excluded) == 1 and str(layout["data"]) not in excluded[0]
    assert str(layout["creds"]) not in with_builtin

    # A data dir that grows (one more run file per turn) changes nothing.
    for i in range(50):
        (layout["data"] / f"ledger_{i}.jsonl").write_text("")
    grown = describe_workspace_scope(
        _scope(
            layout,
            builtin=True,
            mode=mode,
            workspace_builtin_deny_prefixes=[str(layout["data"]), str(layout["creds"])]
            + [str(p) for p in layout["data"].iterdir()],
            **operator,
        )
    )
    assert grown == without


def test_children_inherit_the_builtin_entries(layout) -> None:
    from abstractruntime import (
        Effect, EffectType, InMemoryLedgerStore, InMemoryRunStore, Runtime, StepPlan, WorkflowRegistry, WorkflowSpec,
    )

    child = WorkflowSpec(workflow_id="child", entry_node="end",
                         nodes={"end": lambda r, c: StepPlan(node_id="end", complete_output={})})
    parent = WorkflowSpec(workflow_id="parent", entry_node="spawn", nodes={
        "spawn": lambda r, c: StepPlan(node_id="spawn", effect=Effect(
            type=EffectType.START_SUBWORKFLOW,
            payload={"workflow_id": "child", "vars": {}, "async": True, "wait": True},
            result_key="child",
        )),
    })
    registry = WorkflowRegistry()
    registry.register(child)
    registry.register(parent)
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=registry)
    deny = [str(layout["data"])]
    allow = [str(layout["own"])]
    run_id = rt.start(workflow=parent, vars={
        "workspace_root": str(layout["own"]),
        "workspace_builtin_deny_prefixes": deny,
        "workspace_builtin_allow": allow,
    })
    state = rt.tick(workflow=parent, run_id=run_id, max_steps=1)
    child_id = state.waiting.wait_key.split(":", 1)[1]
    child_vars = rt.get_state(child_id).vars
    assert child_vars["workspace_builtin_deny_prefixes"] == deny
    assert child_vars["workspace_builtin_allow"] == allow
