"""Bounded execute_command for the entity walled set (operator-confirmed,
laurent dm#66 2026-07-19: "we have tiers of execution, the one i don't
allow are rm (unless in his workspace) and any mutable command").

Containment layers pinned here: workspace cwd, denial BY PROGRAM NAME
(param-independent), rm walled to the workspace, git read-only, no shell
operators, parameter-explicit child env (no ambient keys), default OFF
(no default grant carries it)."""
from __future__ import annotations

import tempfile
from pathlib import Path


def _ws(tmp: Path):
    from abstractruntime.identity.tools import WorkspaceRoot

    return WorkspaceRoot(tmp)


def test_execute_runs_in_workspace_cwd_and_returns_output() -> None:
    from abstractruntime.identity.tools import _run_execute_command

    tmp = Path(tempfile.mkdtemp())
    ws = _ws(tmp)
    (ws.root / "hello.txt").write_text("tide line\n", encoding="utf-8")
    out = _run_execute_command("cat hello.txt", ws)
    assert "tide line" in out and "(exit code 0)" in out


def test_denied_programs_by_name_param_independent() -> None:
    from abstractruntime.identity.tools import _run_execute_command

    ws = _ws(Path(tempfile.mkdtemp()))
    for cmd in ("sudo ls", "kill -9 1", "chmod 777 x", "dd if=/dev/zero of=x"):
        out = _run_execute_command(cmd, ws)
        assert "refused" in out and "denied program" in out, (cmd, out)


def test_rm_walled_to_workspace() -> None:
    from abstractruntime.identity.tools import _run_execute_command

    ws = _ws(Path(tempfile.mkdtemp()))
    (ws.root / "scratch.txt").write_text("x", encoding="utf-8")
    ok = _run_execute_command("rm scratch.txt", ws)
    assert "refused" not in ok and not (ws.root / "scratch.txt").exists()
    bad = _run_execute_command("rm /etc/hosts", ws)
    assert "refused" in bad and "inside your workspace" in bad
    bad2 = _run_execute_command("rm ../outside.txt", ws)
    assert "refused" in bad2


def test_git_read_only_by_verb() -> None:
    from abstractruntime.identity.tools import _run_execute_command

    ws = _ws(Path(tempfile.mkdtemp()))
    out = _run_execute_command("git commit -m x", ws)
    assert "refused" in out and "read-only" in out
    out2 = _run_execute_command("git reset --hard", ws)
    assert "refused" in out2
    # A read verb reaches execution (fails as not-a-repo, which is fine —
    # the point is the gate let it through to run).
    out3 = _run_execute_command("git status", ws)
    assert "refused: git" not in out3


def test_shell_operators_refused_one_program_per_call() -> None:
    from abstractruntime.identity.tools import _run_execute_command

    ws = _ws(Path(tempfile.mkdtemp()))
    out = _run_execute_command("echo a && rm -rf /", ws)
    assert "refused" in out and "shell operator" in out
    out2 = _run_execute_command("echo `whoami`", ws)
    assert "refused" in out2


def test_child_env_is_parameter_explicit_no_ambient_keys() -> None:
    import os

    from abstractruntime.identity.tools import _run_execute_command

    ws = _ws(Path(tempfile.mkdtemp()))
    os.environ["AGORA_API_KEY_TEST_SENTINEL"] = "leak-me"
    try:
        out = _run_execute_command("env", ws)
        assert "leak-me" not in out, "ambient env must not ride into the child"
        assert f"HOME={ws.root}" in out
    finally:
        os.environ.pop("AGORA_API_KEY_TEST_SENTINEL", None)


def test_execute_is_in_no_default_grant() -> None:
    """Default OFF everywhere: the operator flips phases in
    tool_policy.yaml; no ruled default carries execute_command."""
    from abstractruntime.identity.tool_policy import resolve_tool_grant

    tmp = Path(tempfile.mkdtemp())
    for phase in ("visit", "work", "personal", "sleep"):
        grant = resolve_tool_grant(tmp, phase)
        assert "execute_command" not in grant.tools, phase


def test_grant_gated_teaching_paragraph() -> None:
    """The execute teaching composes ONLY when the grant carries the tool
    (teach-what-is-wired, per phase)."""
    from abstractruntime.identity.chat import compose_system_base
    from abstractruntime.identity.tools import TIER1_TOOL_NAMES

    with_grant = compose_system_base(
        prelude_text="P", phase="personal", enable_tools=True,
        allowed_tools=TIER1_TOOL_NAMES + ("execute_command",),
        workspace_enabled=True, overlay={},
    )
    assert "execute_command - run ONE program" in with_grant or "grants you execute_command" in with_grant
    without = compose_system_base(
        prelude_text="P", phase="personal", enable_tools=True,
        allowed_tools=TIER1_TOOL_NAMES, workspace_enabled=True, overlay={},
    )
    assert "execute_command" not in without


def test_git_positional_verbs_and_write_flags_refused() -> None:
    """The abstractcode adversarial P0 corpus applied: write sub-verbs as
    POSITIONALS refuse via the read-verb allowlist; write/exec FLAGS on
    allowed verbs (--output writes a file, --ext-diff executes) refuse
    via the flag screen."""
    from abstractruntime.identity.tools import _run_execute_command

    ws = _ws(Path(tempfile.mkdtemp()))
    for cmd in ("git remote set-url origin evil", "git reflog expire --all",
                "git branch -v newname", "git stash drop"):
        out = _run_execute_command(cmd, ws)
        assert "refused" in out and "read-only" in out, (cmd, out)
    # (-O/orderfile READS a path - not screened; the write/exec flags are.)
    for cmd in ("git log --output=/tmp/x", "git diff --ext-diff"):
        out = _run_execute_command(cmd, ws)
        assert "refused" in out, (cmd, out)
