"""The command sandbox through the runtime on Windows, end to end (real processes, real files).

Same tree and the same run vars as ``test_r12_workspace_sandbox.py``; the commands are cmd.exe's.
Every workspace shape is enforced: "Allow everything, refuse listed workspaces" with read & write
or read-only by default, and "Deny everything, allow listed workspaces". An AbstractCore whose
Windows sandbox only has the first one (2.38.0) refuses the two others, with their reason, and the
run continues: the tests of those two shapes then skip, saying which AbstractCore they need, and
the refusal is checked instead. Support is read from AbstractCore's public API
(``host_sandbox_kind("allowed_only")``), never from its version.
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
OUTSIDE = "R12-RUNTIME-OUTSIDE-OK"

# The Windows sandbox of backlog 1341 enforces "Deny everything, allow listed workspaces" and the
# read-only default together; 2.38.0 enforces neither.
EVERY_SHAPE = sys.platform == "win32" and KIND is not None and core_sandbox.host_sandbox_kind("allowed_only") == KIND
NEEDS_EVERY_SHAPE = pytest.mark.skipif(
    not EVERY_SHAPE,
    reason='needs an AbstractCore whose Windows sandbox enforces "Deny everything, allow listed workspaces" and the '
    "read-only default (the release after abstractcore 2.38.0, backlog 1341)",
)
SHAPES = [
    pytest.param("any_except_denied", False, id="any-rw"),
    pytest.param("any_except_denied", True, id="any-ro", marks=NEEDS_EVERY_SHAPE),
    pytest.param("allowed_only", False, id="allow-list", marks=NEEDS_EVERY_SHAPE),
]


@pytest.fixture(autouse=True)
def _fresh_host():
    core_sandbox._reset_host_for_tests()
    yield
    core_sandbox._reset_host_for_tests()


@pytest.fixture
def t(tmp_path) -> Path:
    root = Path(os.path.realpath(tmp_path))
    for d in ("data/workspaces/session-1", "data/other", "home/.ssh", "home/parent/child/deny", "home/outside", "ro", "rw", "refused"):
        (root / d).mkdir(parents=True)
    for rel, text in (
        ("home/outside/f.txt", OUTSIDE),
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


@pytest.fixture(autouse=True)
def _machine_is_the_tree(t, monkeypatch):
    """No rule, integrity label or state file of these tests may land outside the test tree.
    AbstractCore's Windows sandbox reads three views of the real machine (the user data roots
    refused under "Deny everything", the folders Windows already labels Low, its own state
    folder): each is replaced by a folder of the tree, and every rule set is checked before
    the launcher runs."""
    if hasattr(core_sandbox, "_windows_user_data_roots"):
        monkeypatch.setattr(core_sandbox, "_windows_user_data_roots", lambda *a, **k: (str(t / "home"),))
    if hasattr(core_sandbox, "_windows_low_places"):
        (t / "home/AppData/LocalLow").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(core_sandbox, "_windows_low_places", lambda *a, **k: (str(t / "home/AppData/LocalLow"),))
    if hasattr(core_sandbox, "_windows_state_folder"):
        monkeypatch.setattr(core_sandbox, "_windows_state_folder", lambda *a, **k: str(t / "machine/state"))
    real_rules = getattr(core_sandbox, "windows_rules", None)
    if real_rules is None:
        return

    def inside(p: str) -> bool:
        real = os.path.normcase(os.path.realpath(p))
        return real.startswith(os.path.normcase(str(t)).rstrip(os.sep) + os.sep)

    def confined(spec):
        rules = real_rules(spec)
        outside = [r for r in rules for p in r[1:] if not inside(p)]
        assert not outside, f"SAFETY: a sandbox rule outside the test tree, nothing was launched: {outside}"
        if hasattr(core_sandbox, "_windows_state_folder"):
            assert inside(core_sandbox._windows_state_folder())
        return rules

    monkeypatch.setattr(core_sandbox, "windows_rules", confined)


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


def _command(handler, run: RunState, cmd: str):
    """(tool result, its text, stdout lines) of a command that RAN (a refusal fails the test)."""
    res = _call(handler, run, "execute_command", {"command": cmd})
    output = res.get("output")
    assert isinstance(output, dict) and "return_code" in output, f"{cmd!r} never ran: {res}"
    return res, str(output), [line.strip() for line in str(output.get("stdout") or "").splitlines()]


def _in_the_profile(path: Path) -> bool:
    home = os.path.normcase(os.path.realpath(os.path.expanduser("~")))
    return os.path.normcase(str(path)).startswith(home.rstrip(os.sep) + os.sep)


@pytest.mark.parametrize("posture,default_ro", SHAPES)
def test_refused_rows_are_closed_to_commands_and_the_rest_works(t, posture, default_ro):
    handler, run = _handler(), _run(t, posture=posture, default_ro=default_ro)
    confined = posture == "allowed_only" or default_ro
    for cmd in (
        f'type "{t}\\refused\\secret.txt"',
        f'dir "{t}\\refused"',
        f'type "{t}\\home\\parent\\secret.txt"',
        f'cd /d "{t}\\home\\parent\\child\\deny" && type x.txt',
        f'type "{t}\\home\\.ssh\\id_marker"',
        f'type "{t}\\data\\other\\x.txt"',
        f'"{sys.executable}" -c "print(open(r\'{t}\\refused\\secret.txt\').read())"',
        f'echo x> "{t}\\ro\\doc.txt"',
        f'echo x> "{t}\\refused\\new.txt"',
        f'echo x> "{t}\\home\\parent\\new.txt"',
        f'echo x> "{t}\\data\\other\\new.txt"',
    ):
        res, text, stdout = _command(handler, run, cmd)
        assert MARK not in text and "secret.txt" not in stdout, f"{posture}/{'ro' if default_ro else 'rw'}: {cmd}"
        assert KIND in text, cmd
    assert (t / "ro/doc.txt").read_text() == CHILD
    assert not any((t / p).exists() for p in ("refused/new.txt", "home/parent/new.txt", "data/other/new.txt"))
    # An allowed row inside a refused one, a read-only row, the run's own folder and its TEMP, a read & write row.
    res, text, stdout = _command(handler, run, f'type "{t}\\home\\parent\\child\\ok.txt" && type "{t}\\ro\\doc.txt" && echo hi> note.txt && type note.txt && '
                                               f'echo w> "{t}\\rw\\w.txt" && echo c> "{t}\\home\\parent\\child\\c.txt" && echo t> "%TEMP%\\t.txt" && echo TEMP=%TEMP%')  # fmt: skip
    assert text.count(CHILD) >= 2 and "hi" in text, text
    assert (t / "data/workspaces/session-1/note.txt").exists() and (t / "rw/w.txt").exists() and (t / "home/parent/child/c.txt").exists()
    assert f"TEMP={t / 'data/workspaces/session-1/.tmp'}" in stdout and (t / "data/workspaces/session-1/.tmp/t.txt").exists()
    # The rest of the world: written only under read & write by default; read under "Allow everything".
    res, text, stdout = _command(handler, run, f'type "{t}\\home\\outside\\f.txt" & echo w> "{t}\\home\\outside\\w.txt"')
    assert (t / "home/outside/w.txt").exists() != confined, f"an unlisted folder written={not confined}"
    if posture == "any_except_denied":
        assert OUTSIDE in text, text
    elif _in_the_profile(t):
        assert OUTSIDE not in text, "unlisted user data was read under the allow-list"
    # Positive control: the same command prints the marker once the row is not refused (and,
    # under the allow-list, is listed).
    extra: Dict[str, Any] = {"workspace_ignored_paths": ""}
    if posture == "allowed_only":
        extra["workspace_allowed_paths"] = _vars(t)["workspace_allowed_paths"] + [str(t / "refused")]
    res, text, _ = _command(handler, _run(t, posture=posture, default_ro=default_ro, **extra), f'type "{t}\\refused\\secret.txt"')
    assert MARK in text


@pytest.mark.skipif(EVERY_SHAPE, reason="this AbstractCore enforces every shape on Windows (test_refused_rows_are_closed_to_commands_and_the_rest_works)")
@pytest.mark.parametrize("posture,default_ro,reason", [
    ("any_except_denied", True, core_sandbox.__dict__.get("WINDOWS_READ_ONLY_UNEXPRESSIBLE")),
    ("allowed_only", False, core_sandbox.__dict__.get("WINDOWS_ALLOW_LIST_UNEXPRESSIBLE")),
])  # fmt: skip
def test_an_older_abstractcore_refuses_the_two_shapes_and_the_run_continues(t, posture, default_ro, reason):
    res = _call(_handler(), _run(t, posture=posture, default_ro=default_ro), "execute_command", {"command": f'type "{t}\\rw\\nothing.txt" & echo ran> ran.txt'})
    assert res["success"] is False and reason and reason in str(res)
    assert not (t / "data/workspaces/session-1/ran.txt").exists()  # the command did not run


@pytest.mark.parametrize("posture,default_ro", SHAPES)
def test_a_local_helper_runs_inside_the_sandbox(t, posture, default_ro):
    handler, run = _handler(), _run(t, posture=posture, default_ro=default_ro)
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
