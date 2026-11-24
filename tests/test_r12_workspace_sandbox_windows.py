"""The command sandbox through the runtime on Windows, end to end (real processes, real files).

Same tree and the same run vars as ``test_r12_workspace_sandbox.py``; the commands are cmd.exe's.
The Windows sandbox expresses "Allow everything, refuse listed workspaces" with read & write by
default; the two shapes it cannot express are refused with their reason and the run continues.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from abstractcore.tools import sandbox as core_sandbox

from abstractruntime.core.models import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor

KIND = getattr(core_sandbox, "KIND_WINDOWS", None)
pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or KIND is None,
    reason="the Windows command sandbox: Windows, with an AbstractCore that has it",
)

MARK = "R12-RUNTIME-REFUSED-MARKER"
CHILD = "R12-RUNTIME-CHILD-OK"


@pytest.fixture(autouse=True)
def _fresh_host():
    core_sandbox._reset_host_for_tests()
    yield
    core_sandbox._reset_host_for_tests()


@pytest.fixture
def t(tmp_path) -> Path:
    root = Path(os.path.realpath(tmp_path))
    for d in ("data/workspaces/session-1", "data/other", "home/.ssh", "home/parent/child/deny", "ro", "rw", "refused"):
        (root / d).mkdir(parents=True)
    for rel, text in (
        ("refused/secret.txt", MARK),
        ("home/parent/secret.txt", MARK),
        ("home/parent/child/ok.txt", CHILD),
        ("home/parent/child/deny/x.txt", MARK),
        ("home/.ssh/id_marker", MARK),
        ("data/other/x.txt", MARK),
        ("ro/doc.txt", CHILD),
    ):
        (root / rel).write_text(text)
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


def _handler():
    from abstractcore.tools.common_tools import execute_command
    from abstractruntime.integrations.abstractcore.local_helper_tools import local_helper_start, local_helper_stop

    return make_tool_calls_handler(tools=MappingToolExecutor.from_tools([execute_command, local_helper_start, local_helper_stop]))


def _run(t: Path, **kw) -> RunState:
    return RunState.new(workflow_id="wf", entry_node="n", session_id="s", vars=_vars(t, **kw))


def _call(handler, run: RunState, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    effect = Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": [{"name": name, "arguments": args, "call_id": "c1"}]})
    return handler(run, effect, None).result["results"][0]


def test_refused_rows_are_closed_to_commands_and_the_rest_works(t):
    handler, run = _handler(), _run(t)
    for cmd in (
        f'type "{t}\\refused\\secret.txt"',
        f'dir "{t}\\refused"',
        f'type "{t}\\home\\parent\\secret.txt"',
        f'cd /d "{t}\\home\\parent\\child\\deny" && type x.txt',
        f'type "{t}\\home\\.ssh\\id_marker"',
        f'type "{t}\\data\\other\\x.txt"',
        f'echo x> "{t}\\ro\\doc.txt"',
        f'echo x> "{t}\\refused\\new.txt"',
    ):
        res = _call(handler, run, "execute_command", {"command": cmd})
        text = str(res.get("output"))
        assert MARK not in text and "secret.txt" not in str((res.get("output") or {}).get("stdout") or ""), cmd
        assert KIND in text, cmd
    assert (t / "ro/doc.txt").read_text() == CHILD and not (t / "refused/new.txt").exists()
    # An allowed row inside a refused one, a read-only row, the run's own folder, the rest of the world.
    res = _call(handler, run, "execute_command", {"command": f'type "{t}\\home\\parent\\child\\ok.txt" && type "{t}\\ro\\doc.txt" && echo hi> note.txt && type note.txt && echo w> "{t}\\rw\\w.txt"'})
    text = str(res.get("output"))
    assert text.count(CHILD) >= 2 and "hi" in text, text
    assert (t / "data/workspaces/session-1/note.txt").exists() and (t / "rw/w.txt").exists()
    # Positive control: the same command prints the marker once the row is not refused.
    res = _call(handler, _run(t, workspace_ignored_paths=""), "execute_command", {"command": f'type "{t}\\refused\\secret.txt"'})
    assert MARK in str(res.get("output"))


@pytest.mark.parametrize("posture,default_ro,reason", [
    ("any_except_denied", True, core_sandbox.__dict__.get("WINDOWS_READ_ONLY_UNEXPRESSIBLE")),
    ("allowed_only", False, core_sandbox.__dict__.get("WINDOWS_ALLOW_LIST_UNEXPRESSIBLE")),
])
def test_what_windows_cannot_express_is_refused_and_the_run_continues(t, posture, default_ro, reason):
    res = _call(_handler(), _run(t, posture=posture, default_ro=default_ro), "execute_command", {"command": f'type "{t}\\rw\\nothing.txt" & echo ran> ran.txt'})
    assert res["success"] is False and reason and reason in str(res)
    assert not (t / "data/workspaces/session-1/ran.txt").exists()  # the command did not run


def test_a_local_helper_runs_inside_the_sandbox(t):
    handler, run = _handler(), _run(t)
    leak = t / "data/workspaces/session-1/leak.txt"
    command = f'cmd /c "type {t}\\refused\\secret.txt > {leak} & echo done > {leak}.done"'
    try:
        res = _call(handler, run, "local_helper_start", {"command": command, "ready_timeout": 1})
        deadline = time.time() + 20
        while time.time() < deadline and not Path(str(leak) + ".done").exists():
            time.sleep(0.2)
        assert Path(str(leak) + ".done").exists(), res  # the helper ran, in the run's folder
        assert not leak.exists() or MARK not in leak.read_text()
        assert KIND in str(res.get("output"))
    finally:
        _call(handler, run, "local_helper_stop", {})


def test_a_local_helper_command_is_split_and_found_by_windows_rules(tmp_path, monkeypatch):
    from abstractruntime.integrations.abstractcore.local_helper_tools import _parse_command

    (tmp_path / "devserver.cmd").write_text("@echo off")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    argv, error = _parse_command("devserver run C:" + chr(92) + "proj" + chr(92) + 'app.py "two words"')
    assert error is None
    assert os.path.normcase(argv[0]) == os.path.normcase(str(tmp_path / "devserver.cmd"))  # found as a shell finds it
    assert argv[1:] == ["run", "C:" + chr(92) + "proj" + chr(92) + "app.py", "two words"]  # the path keeps its separators
    assert _parse_command("a | b")[1] and _parse_command("echo $(whoami)")[1]  # still no shell operators
