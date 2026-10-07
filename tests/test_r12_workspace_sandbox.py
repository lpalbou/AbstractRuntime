"""Round 12: the runtime half of the command sandbox and the nesting rule.

- R12 NESTING RULE in the file-tool scope checks (longest real-path prefix; a refusal wins a
  tie; built-in refusals absolute).
- `sandbox_stamp(scope)`: the run's effective workspace set, from the SAME keys the file tools
  read, stamped as the hidden `_sandbox` arg of every process-spawning tool (a model value is
  never honoured).
- End to end inside the host's OS sandbox (macOS sandbox-exec; Linux bubblewrap in CI): gateway-shaped run vars -> TOOL_CALLS handler -> core execute_command /
  shell_exec / runtime local_helper_start / the entity exec tool.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractcore.tools import sandbox as core_sandbox
from abstractruntime.core.models import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor
from abstractruntime.integrations.abstractcore.workspace_scoped_tools import (
    SANDBOXED_TOOL_NAMES,
    WorkspaceScope,
    resolve_user_path,
    rewrite_tool_arguments,
    sandbox_stamp,
)

MARK = "R12-RUNTIME-REFUSED-MARKER"
CHILD = "R12-RUNTIME-CHILD-OK"

# The end-to-end tests run inside the host's real OS sandbox: macOS sandbox-exec, or Linux
# bubblewrap (the CI job `linux-sandbox` installs bwrap and fails if these tests skip).
if sys.platform == "darwin":
    HOST_KIND = core_sandbox.KIND_MACOS if os.access(core_sandbox.SANDBOX_EXEC, os.X_OK) else None
elif sys.platform.startswith("linux"):
    HOST_KIND = core_sandbox.KIND_BWRAP if core_sandbox._bwrap_path() else None
else:
    HOST_KIND = None
HOST_LABEL = core_sandbox.KIND_LABELS.get(HOST_KIND or "", "")

real_sandbox = pytest.mark.skipif(
    HOST_KIND is None,
    reason="real sandbox end-to-end: this host has neither /usr/bin/sandbox-exec (macOS) nor bubblewrap (Linux)",
)


def _disable_host_sandbox(monkeypatch, t: Path) -> None:
    """Make this host's sandbox unavailable (the fail-closed path)."""
    monkeypatch.setattr(core_sandbox, "SANDBOX_EXEC", str(t / "missing-sandbox-exec"))
    monkeypatch.setattr(core_sandbox, "_bwrap_path", lambda: None)
    monkeypatch.setattr(core_sandbox, "_landlock_abi", lambda: 0)


@pytest.fixture(autouse=True)
def _fresh_host():
    core_sandbox._reset_host_for_tests()
    yield
    core_sandbox._reset_host_for_tests()


@pytest.fixture
def t(tmp_path, monkeypatch):
    return _tree(Path(os.path.realpath(tmp_path)), monkeypatch)


def _tree(root: Path, monkeypatch) -> Path:
    for d in ("data/workspaces/session-1", "data/other", "home/.ssh", "home/parent/child", "home/parent/child/deny", "ro", "rw", "refused"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "refused/secret.txt").write_text(MARK)
    (root / "home/parent/secret.txt").write_text(MARK)
    (root / "home/parent/child/ok.txt").write_text(CHILD)
    (root / "home/parent/child/deny/x.txt").write_text(MARK)
    (root / "home/.ssh/id_marker").write_text(MARK)
    (root / "data/other/x.txt").write_text(MARK)
    monkeypatch.setenv("HOME", str(root / "home"))
    return root


def _vars(t: Path, *, posture: str = "any_except_denied", default_ro: bool = False, **extra) -> Dict[str, Any]:
    """Run vars exactly as the gateway's apply_workspace_policy flattens them."""
    root = str(t / "data/workspaces/session-1")
    v: Dict[str, Any] = {
        "workspace_root": root,
        "workspace_access_mode": "all_except_ignored" if posture == "any_except_denied" else "workspace_or_allowed",
        "workspace_allowed_paths": [str(t / "ro"), str(t / "rw"), str(t / "home/parent/child")],
        "workspace_ignored_paths": "\n".join([str(t / "refused"), str(t / "home/parent"), str(t / "home/parent/child/deny")]),
        "workspace_read_only_paths": (["/"] if default_ro else []) + [str(t / "ro")],
        "workspace_writable_paths": [root, str(t / "rw"), str(t / "home/parent/child")],
        "workspace_builtin_deny_prefixes": [str(t / "data"), str(t / "home/.ssh")],
        "workspace_builtin_allow": [root],
    }
    v.update(extra)
    return v


# --- R12 NESTING RULE in the file-tool scope ---------------------------------------------


@pytest.mark.parametrize("posture", ["any_except_denied", "allowed_only"])
def test_nesting_most_specific_row_wins(t, posture):
    scope = WorkspaceScope.from_input_data(_vars(t, posture=posture))
    # Refused parent, allowed child: the child is reachable, the rest of the parent is not.
    assert resolve_user_path(scope=scope, user_path=str(t / "home/parent/child/ok.txt"))
    with pytest.raises(ValueError, match="blocked" if posture == "any_except_denied" else "outside workspace roots"):
        resolve_user_path(scope=scope, user_path=str(t / "home/parent/secret.txt"))
    # A refused row inside an allowed one refuses that subtree.
    with pytest.raises(ValueError, match="blocked"):
        resolve_user_path(scope=scope, user_path=str(t / "home/parent/child/deny/x.txt"))
    with pytest.raises(ValueError):
        resolve_user_path(scope=scope, user_path=str(t / "refused/secret.txt"))


def test_nesting_tie_refusal_wins(t):
    v = _vars(t, workspace_ignored_paths=str(t / "rw"))
    scope = WorkspaceScope.from_input_data(v)
    with pytest.raises(ValueError, match="blocked"):
        resolve_user_path(scope=scope, user_path=str(t / "rw/f.txt"))


def test_nesting_builtin_refusal_is_absolute(t):
    v = _vars(t, workspace_allowed_paths=[str(t / "data/other"), str(t / "home")])
    scope = WorkspaceScope.from_input_data(v)
    for p in (t / "data/other/x.txt", t / "home/.ssh/id_marker"):
        with pytest.raises(ValueError, match="protected by the host"):
            resolve_user_path(scope=scope, user_path=str(p))
    assert resolve_user_path(scope=scope, user_path="note.txt")  # the run's own folder


def test_nesting_symlink_resolves_before_matching(t):
    (t / "rw/link").symlink_to(t / "refused")
    scope = WorkspaceScope.from_input_data(_vars(t))
    with pytest.raises(ValueError):
        resolve_user_path(scope=scope, user_path=str(t / "rw/link/secret.txt"))


# --- the stamp ---------------------------------------------------------------------------


def test_stamp_is_the_file_tool_scope(t):
    stamp = sandbox_stamp(WorkspaceScope.from_input_data(_vars(t)))
    assert stamp["posture"] == "any_except_denied" and stamp["default_mode"] == "rw"
    assert stamp["private_workspace"] == str(t / "data/workspaces/session-1")
    modes = {r["path"]: r["mode"] for r in stamp["allowed"]}
    assert modes == {str(t / "ro"): "ro", str(t / "rw"): "rw", str(t / "home/parent/child"): "rw"}
    assert set(stamp["refused"]) == {str(t / "refused"), str(t / "home/parent"), str(t / "home/parent/child/deny")}
    assert stamp["builtin_refused"] == [str(t / "data"), str(t / "home/.ssh")]
    assert stamp["builtin_allowed"] == [str(t / "data/workspaces/session-1")]
    assert "env" not in stamp and "unsandboxed_commands_allowed" not in stamp


def test_stamp_postures_and_ro_default(t):
    s1 = sandbox_stamp(WorkspaceScope.from_input_data(_vars(t, posture="allowed_only")))
    assert s1["posture"] == "allowed_only"
    s2 = sandbox_stamp(WorkspaceScope.from_input_data(_vars(t, default_ro=True)))
    assert s2["default_mode"] == "ro"
    assert {r["path"]: r["mode"] for r in s2["allowed"]}[str(t / "rw")] == "rw"  # writable exception
    s3 = sandbox_stamp(WorkspaceScope.from_input_data(_vars(t, workspace_access_mode="workspace_only")))
    assert s3["posture"] == "allowed_only" and s3["allowed"] == []


@pytest.mark.parametrize("tool", sorted(SANDBOXED_TOOL_NAMES))
def test_every_spawning_tool_is_stamped_and_a_forged_stamp_is_replaced(t, tool):
    scope = WorkspaceScope.from_input_data(_vars(t))
    forged = {"private_workspace": "/", "posture": "any_except_denied", "allowed": [], "refused": []}
    out = rewrite_tool_arguments(tool_name=tool, args={"command": "ls", "_sandbox": forged}, scope=scope)
    assert out["_sandbox"] == sandbox_stamp(scope)
    if tool == "execute_python":
        assert "working_directory" not in out  # starts in the stamp's private workspace
    else:
        assert out["working_directory"] == str(t / "data/workspaces/session-1")


class _Capture:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def execute(self, *, tool_calls, **kw):
        self.calls.extend(dict(tc) for tc in tool_calls)
        return {"mode": "executed", "results": [{"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok"} for tc in tool_calls]}


def _effect(name: str, arguments: Dict[str, Any]) -> Effect:
    return Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": [{"call_id": "c1", "name": name, "arguments": arguments}]})


def test_handler_pops_a_model_stamp_without_scope():
    ex = _Capture()
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s", vars={})
    make_tool_calls_handler(tools=ex)(run, _effect("execute_command", {"command": "ls", "_sandbox": {"posture": "any_except_denied"}}), None)
    assert "_sandbox" not in ex.calls[0]["arguments"]


def test_handler_stamps_from_run_vars(t):
    ex = _Capture()
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s", vars=_vars(t))
    make_tool_calls_handler(tools=ex)(run, _effect("execute_command", {"command": "ls", "_sandbox": {"private_workspace": "/"}}), None)
    assert ex.calls[0]["arguments"]["_sandbox"] == sandbox_stamp(WorkspaceScope.from_input_data(_vars(t)))


# --- end to end on macOS -----------------------------------------------------------------


def _real_handler():
    from abstractcore.tools.common_tools import execute_command
    from abstractcore.tools.shell_tools import shell_exec
    from abstractruntime.integrations.abstractcore.local_helper_tools import local_helper_start, local_helper_stop

    return make_tool_calls_handler(tools=MappingToolExecutor.from_tools([execute_command, shell_exec, local_helper_start, local_helper_stop]))


def _result(outcome) -> Dict[str, Any]:
    res = outcome.result["results"][0]
    return res


@real_sandbox
@pytest.mark.parametrize("posture,default_ro", [("any_except_denied", False), ("any_except_denied", True), ("allowed_only", False)])
def test_e2e_execute_command_through_the_runtime(t, posture, default_ro):
    handler = _real_handler()
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s", vars=_vars(t, posture=posture, default_ro=default_ro))
    for cmd in (
        f"cat {t}/refused/secret.txt",
        f"ls -la {t}/refused",
        f"cat {t}/home/parent/secret.txt",
        f"cd {t}/home/parent/child/deny && cat x.txt",
        "cat ~/.ssh/id_marker",
        f"cat {t}/data/other/x.txt",
    ):
        res = _result(handler(run, _effect("execute_command", {"command": cmd}), None))
        text = str(res.get("output"))
        listing = [ln.strip() for ln in str((res.get("output") or {}).get("stdout") or "").splitlines()]
        assert MARK not in text and "secret.txt" not in listing, cmd
        assert HOST_KIND in text
    res = _result(handler(run, _effect("execute_command", {"command": f"cat {t}/home/parent/child/ok.txt && echo hi > note.txt && cat note.txt"}), None))
    text = str(res.get("output"))
    assert CHILD in text and "hi" in text, text  # allowed child + private workspace work
    assert (t / "data/workspaces/session-1/note.txt").exists()
    if posture == "any_except_denied":
        # Positive control: the same command prints the marker once the row is not refused.
        open_run = RunState.new(workflow_id="wf", entry_node="n", session_id="s", vars=_vars(t, posture=posture, default_ro=default_ro, workspace_ignored_paths=""))
        res = _result(handler(open_run, _effect("execute_command", {"command": f"cat {t}/refused/secret.txt"}), None))
        assert MARK in str(res.get("output"))


@real_sandbox
def test_e2e_shell_session_through_the_runtime(t):
    handler = _real_handler()
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s", vars=_vars(t))
    try:
        res = _result(handler(run, _effect("shell_exec", {"command": f"cd {t}/refused; cat secret.txt; cat {t}/home/parent/child/ok.txt"}), None))
        text = str(res.get("output"))
        assert MARK not in text and CHILD in text and f"Sandbox: {HOST_LABEL}" in text
    finally:
        from abstractcore.tools.shell_session import get_shell_session_registry

        get_shell_session_registry().close_namespace(str(run.run_id))


@real_sandbox
def test_e2e_local_helper_is_sandboxed(t):
    handler = _real_handler()
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s", vars=_vars(t))
    leak = t / "data/workspaces/session-1/leak.txt"
    res = _result(handler(run, _effect("local_helper_start", {"command": f"cp {t}/refused/secret.txt {leak}", "ready_timeout": 1}), None))
    time.sleep(0.5)
    handler(run, _effect("local_helper_stop", {}), None)
    assert not leak.exists() or MARK not in leak.read_text()
    assert HOST_KIND in str(res.get("output"))


@real_sandbox
def test_e2e_fail_closed_keeps_the_run_going(t, monkeypatch):
    _disable_host_sandbox(monkeypatch, t)
    handler = _real_handler()
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s", vars=_vars(t))
    outcome = handler(run, _effect("execute_command", {"command": f"cat {t}/refused/secret.txt"}), None)
    assert str(outcome.status) == "completed"  # the run continues
    res = _result(outcome)
    assert res["success"] is False and "not sandboxed on this gateway host" in str(res)
    assert MARK not in str(res)


@pytest.fixture(params=["temp_root", "user_data_root"])
def t_anywhere(request, tmp_path, monkeypatch):
    """The same tree under pytest's temp root (macOS: /private/var/folders or /private/tmp,
    Linux: /tmp) AND under a user-data root (the real home), so the entity test cannot flip
    on where TMPDIR points: an unlisted folder must be unreadable in both."""
    if request.param == "temp_root":
        yield _tree(Path(os.path.realpath(tmp_path)), monkeypatch)
        return
    import pwd
    import shutil
    import tempfile

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    base = real_home / ".cache" / "abstractruntime-tests"
    base.mkdir(parents=True, exist_ok=True)
    root = Path(os.path.realpath(tempfile.mkdtemp(prefix="r12-", dir=str(base))))
    try:
        yield _tree(root, monkeypatch)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_macos_allow_list_denies_the_shared_temp_roots():
    """macOS "Deny everything, allow listed workspaces" denies the shared temp roots too
    (where pytest's tmp tree and other processes' files live), not only /Users."""
    roots = core_sandbox._USER_DATA_ROOTS_DARWIN
    for r in ("/Users", "/Volumes", "/private/var/root", "/private/tmp", "/private/var/folders"):
        assert r in roots, r


@real_sandbox
def test_e2e_entity_exec_is_sandboxed(t_anywhere):
    t = t_anywhere
    from abstractruntime.identity.tools import WorkspaceRoot, _run_execute_command

    home = t / "entity"
    ws = WorkspaceRoot(home)
    out = _run_execute_command(f"cat {t}/refused/secret.txt", ws)
    assert MARK not in out
    out = _run_execute_command(f"python3 -c \"print(open('{t}/home/parent/secret.txt').read())\"", ws)
    assert MARK not in out
    (ws.root / "mine.txt").write_text("ENTITY-OK")
    assert "ENTITY-OK" in _run_execute_command("cat mine.txt", ws)


# --- a core without the sandbox fails closed ---------------------------------------------


@pytest.fixture
def no_core_sandbox(monkeypatch):
    import abstractcore.tools as core_tools

    monkeypatch.setitem(sys.modules, "abstractcore.tools.sandbox", None)
    monkeypatch.delattr(core_tools, "sandbox", raising=False)


def test_core_without_sandbox_refuses_before_anything_runs(t, no_core_sandbox):
    from abstractruntime.integrations.abstractcore.workspace_scoped_tools import NO_CORE_SANDBOX

    proof = t / "data/workspaces/session-1/ran.txt"
    ex = _Capture()
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s", vars=_vars(t))
    for tool in sorted(SANDBOXED_TOOL_NAMES):
        outcome = make_tool_calls_handler(tools=ex)(run, _effect(tool, {"command": f"touch {proof}"}), None)
        assert str(outcome.status) == "completed"  # the run continues
        res = outcome.result["results"][0]
        assert res["success"] is False and res["error"] == NO_CORE_SANDBOX
    assert ex.calls == [] and not proof.exists()  # nothing reached the executor, nothing ran


def test_local_helper_and_entity_without_core_sandbox_refuse(t, no_core_sandbox):
    from abstractruntime.identity.tools import WorkspaceRoot, _run_execute_command
    from abstractruntime.integrations.abstractcore.local_helper_tools import local_helper_start
    from abstractruntime.integrations.abstractcore.workspace_scoped_tools import NO_CORE_SANDBOX

    res = local_helper_start(command="touch should-not-exist", working_directory=str(t / "data"), _registry_namespace="r12")
    assert res == {"success": False, "error": NO_CORE_SANDBOX}
    ws = WorkspaceRoot(t / "entity")
    assert _run_execute_command("touch should-not-exist", ws) == f"refused: {NO_CORE_SANDBOX}"
    assert not (ws.root / "should-not-exist").exists() and not (t / "data/should-not-exist").exists()


def test_stale_execute_python_without_the_stamp_kwarg_fails_closed(t):
    """An AbstractAgent older than 0.3.18 has an execute_python without `_sandbox`: the stamped
    call fails on the unknown argument (TypeError surfaced as a failed tool result) and the
    snippet never runs — it is never executed unsandboxed."""
    from abstractcore.tools import tool

    ran = t / "data/workspaces/session-1/stale-ran.txt"

    @tool(name="execute_python", description="Stale execute_python without the sandbox stamp.")
    def stale_execute_python(code: str, timeout_s: float = 0.0, max_output_chars: int = 0) -> dict:
        ran.write_text("ran")
        return {"stdout": "ran", "stderr": "", "exit_code": 0}

    handler = make_tool_calls_handler(tools=MappingToolExecutor.from_tools([stale_execute_python]))
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s", vars=_vars(t))
    outcome = handler(run, _effect("execute_python", {"code": "print(1)"}), None)
    assert str(outcome.status) == "completed"
    res = _result(outcome)
    assert res["success"] is False and "_sandbox" in str(res.get("error"))
    assert not ran.exists()


def test_model_key_never_maps_onto_the_hidden_stamp(t, monkeypatch):
    """The executor's key normalization (`filePath` -> `file_path`) must not turn a model's
    `sandbox` / `Sandbox` key into the host-only `_sandbox` parameter."""
    from abstractcore.tools import tool

    seen = {}

    @tool(name="probe_tool", description="Records the stamp it receives.")
    def probe_tool(command: str = "", _sandbox: dict = None) -> dict:
        seen["stamp"] = _sandbox
        return {"ok": True}

    ex = MappingToolExecutor.from_tools([probe_tool])
    ex.execute(tool_calls=[{"call_id": "c1", "name": "probe_tool", "arguments": {"command": "x", "sandbox": {"posture": "any_except_denied"}, "Sandbox": {}}}])
    assert seen["stamp"] is None
