"""An activated skill is always readable (operator 2026-11-08).

An agent in Code, with a skill activated for its session, called `read_file`
on `<shelf>/skills/abstractcore/SKILL.md` and got
`Error: File '…' is ignored by .abstractignore policy`: core's file tool
applies the workspace `.abstractignore`, which prunes the gateway data folder
where the shelf lives. The runtime now answers a `read_file` inside one of the
run's skill folders itself, before core's tool runs.

The executor below plays core's part: whatever it is asked, it answers with
the ignore-policy refusal — so every "served" assertion here is red without
the interception (the result would be that refusal), and every "goes to core"
assertion sees exactly the calls core received.

Review 2026-11-09 finding 1 (BLOCKER): the folders used to come from run vars a
client writes (`skills` mapping roots, `_runtime.skills_resolution`, the
`<path>` of `_runtime.skills_block`), so any client could read every file
beside any SKILL.md. A folder now lies under a root the HOST registered
(`register_skill_roots`), for a skill NAME the run selected; nothing registered
= nothing served. The forged-variable cases below are the reviewer's probe
(rt_probe/probe.py): each one must get core's normal answer.
"""

from __future__ import annotations

import os
from pathlib import Path

from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.skill_roots import (
    clear_skill_roots,
    register_skill_roots,
    registered_skill_roots,
    skill_file_read,
    skill_folders_for_run_vars,
)
from abstractruntime.storage.in_memory import InMemoryRunStore

import pytest

SKILL_TEXT = "---\nname: x\ndescription: Demo skill\n---\n\n# X\n\nUse it well.\n"
REFERENCE_TEXT = "one\ntwo\nthree\nfour\n"


@pytest.fixture(autouse=True)
def _no_roots_registered():
    """Registration is process-wide: every test starts and ends with none."""
    clear_skill_roots()
    yield
    clear_skill_roots()


class _CoreLikeExecutor:
    """Answers every read_file the way core did for the operator's agent."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def execute(self, *, tool_calls: list[dict]) -> dict:
        self.calls.extend(tool_calls or [])
        return {
            "mode": "executed",
            "results": [
                {
                    "call_id": str(tc.get("call_id") or ""),
                    "name": str(tc.get("name") or ""),
                    "success": True,
                    "output": (
                        f"Error: File '{(tc.get('arguments') or {}).get('file_path')}' "
                        "is ignored by .abstractignore policy"
                    ),
                    "error": None,
                }
                for tc in (tool_calls or [])
            ],
        }


def _shelf(tmp_path: Path) -> Path:
    """<tmp>/data/skills/skills/x/{SKILL.md, references/ops.md} with an ignore file that prunes it."""
    skills_root = tmp_path / "data" / "skills" / "skills"
    skill = skills_root / "x"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(SKILL_TEXT, encoding="utf-8")
    (skill / "references" / "ops.md").write_text(REFERENCE_TEXT, encoding="utf-8")
    (tmp_path / "data" / ".abstractignore").write_text("skills/\n", encoding="utf-8")
    return skills_root


def _host_shelf(tmp_path: Path) -> Path:
    """The shelf, registered by the host as one of its skill roots."""
    skills_root = _shelf(tmp_path)
    register_skill_roots([skills_root])
    return skills_root


def _run(tmp_path: Path, **extra_vars) -> RunState:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    now = "2026-11-08T00:00:00+00:00"
    vars_ = {
        "context": {"task": "t", "messages": []},
        "scratchpad": {},
        "_runtime": {"memory_spans": []},
        "_temp": {},
        "_limits": {},
        "workspace_root": str(ws),
        "workspace_access_mode": "workspace_only",
    }
    rt_extra = extra_vars.pop("_runtime", None)
    if isinstance(rt_extra, dict):
        vars_["_runtime"].update(rt_extra)
    vars_.update(extra_vars)
    return RunState(
        run_id="r1",
        workflow_id="wf_test",
        status=RunStatus.RUNNING,
        current_node="node",
        vars=vars_,
        waiting=None,
        output={"messages": []},
        error=None,
        created_at=now,
        updated_at=now,
        actor_id=None,
        session_id="s1",
        parent_run_id=None,
    )


def _read(handler, run: RunState, file_path: str, **args) -> dict:
    out = handler(
        run,
        Effect(
            type=EffectType.TOOL_CALLS,
            payload={"tool_calls": [{"call_id": "c1", "name": "read_file", "arguments": {"file_path": file_path, **args}}]},
        ),
        None,
    )
    assert out.status == "completed", out
    results = out.result.get("results") or []
    assert len(results) == 1, results
    return results[0]


def _handler(executor: _CoreLikeExecutor):
    return make_tool_calls_handler(tools=executor, run_store=InMemoryRunStore())


def test_skill_file_is_served_whole_despite_the_ignore_policy(tmp_path: Path) -> None:
    skills_root = _host_shelf(tmp_path)
    executor = _CoreLikeExecutor()
    run = _run(tmp_path, skills={"shelf": str(skills_root), "active": ["x"], "requested": ["x"]})
    path = str(skills_root / "x" / "SKILL.md")

    result = _read(_handler(executor), run, path)

    assert result["success"] is True and result["error"] is None
    expected_lines = SKILL_TEXT.rstrip("\n").split("\n")
    expected = f"File: {path} ({len(expected_lines)} lines)\n\n" + "\n".join(
        f"{i}: {line}" for i, line in enumerate(expected_lines, 1)
    )
    assert result["output"] == expected
    assert executor.calls == [], "core must never be asked about a file of an activated skill"


def test_a_reference_file_honours_the_line_range(tmp_path: Path) -> None:
    skills_root = _host_shelf(tmp_path)
    executor = _CoreLikeExecutor()
    run = _run(tmp_path, skills={"shelf": str(skills_root), "active": ["x"]})
    path = str(skills_root / "x" / "references" / "ops.md")

    result = _read(_handler(executor), run, path, start_line=2, end_line=3)

    assert result["success"] is True
    assert result["output"] == f"File: {path} (2 lines)\n\n2: two\n3: three"
    assert executor.calls == []


def test_a_path_outside_the_skill_folders_still_goes_to_core(tmp_path: Path) -> None:
    skills_root = _host_shelf(tmp_path)
    executor = _CoreLikeExecutor()
    run = _run(tmp_path, skills={"shelf": str(skills_root), "active": ["x"]})
    notes = tmp_path / "ws" / "notes.md"
    notes.write_text("n\n", encoding="utf-8")

    result = _read(_handler(executor), run, str(notes))

    assert len(executor.calls) == 1 and executor.calls[0]["name"] == "read_file"
    assert "ignored by .abstractignore policy" in str(result["output"])


def test_a_symlink_leading_out_of_the_folder_is_refused(tmp_path: Path) -> None:
    skills_root = _host_shelf(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("the secret\n", encoding="utf-8")
    os.symlink(secret, skills_root / "x" / "leak.md")
    executor = _CoreLikeExecutor()
    run = _run(tmp_path, skills={"shelf": str(skills_root), "active": ["x"]})

    result = _read(_handler(executor), run, str(skills_root / "x" / "leak.md"))

    assert result["success"] is False
    assert "refused" in str(result["error"])
    assert "the secret" not in str(result["output"] or "")
    assert executor.calls == [], "a refused path is never handed to core either"


def test_a_names_list_uses_the_recorded_resolution(tmp_path: Path) -> None:
    """The run-start shape: `skills` is a list of names; the host's resolution names the shelf."""
    skills_root = _host_shelf(tmp_path)
    executor = _CoreLikeExecutor()
    run = _run(
        tmp_path,
        skills=["x", "held-one"],
        _runtime={"skills_resolution": {"requested": ["x", "held-one"], "active": ["x"], "shelf": str(skills_root)}},
    )

    result = _read(_handler(executor), run, str(skills_root / "x" / "SKILL.md"))

    assert result["success"] is True and "Use it well." in result["output"]
    assert executor.calls == []


def test_the_skills_block_names_select_under_the_registered_root(tmp_path: Path) -> None:
    """An Agent-node child carries only the block: its <name>s select, the host's root locates."""
    skills_root = _host_shelf(tmp_path)
    executor = _CoreLikeExecutor()
    block = (
        "<available_skills>\n  <skill>\n    <name>x</name>\n    <description>Demo skill</description>\n"
        f"    <path>{skills_root / 'x' / 'SKILL.md'}</path>\n  </skill>\n</available_skills>"
    )
    run = _run(tmp_path, _runtime={"skills_block": block})

    result = _read(_handler(executor), run, str(skills_root / "x" / "references" / "ops.md"))

    assert result["success"] is True and result["output"].endswith("4: four")
    assert executor.calls == []


def test_folder_derivation_reads_names_from_the_run_and_roots_from_the_host(tmp_path: Path) -> None:
    skills_root = _shelf(tmp_path)  # <root>/<name>
    flat_root = tmp_path / "flat"
    (flat_root / "y").mkdir(parents=True)
    (flat_root / "y" / "SKILL.md").write_text("---\nname: y\ndescription: Y\n---\n", encoding="utf-8")
    imported = tmp_path / "imported"
    (imported / "z").mkdir(parents=True)
    (imported / "z" / "SKILL.md").write_text("---\nname: z\ndescription: Z\n---\n", encoding="utf-8")
    # The shelf folder whose skills/ holds x, the flat root and the imported folder.
    register_skill_roots([skills_root.parent, flat_root, imported, "relative/ignored"])
    assert registered_skill_roots() == [str(skills_root.parent), str(flat_root), str(imported)]

    folders = skill_folders_for_run_vars({"skills": {"active": ["x", "y", "z", "held@deadbeef"], "requested": ["x", "absent"]}})

    assert {f.name for f in folders} == {"x", "y", "z"}
    assert all((f / "SKILL.md").is_file() for f in folders)
    # A held name (absent from the resolution's active list) never yields a folder.
    held = {"skills": ["x"], "_runtime": {"skills_resolution": {"requested": ["x"], "active": []}}}
    assert skill_file_read(held, {"file_path": str(skills_root / "x" / "SKILL.md")}) is None
    # No skills at all: every path falls through to core.
    assert skill_file_read({}, {"file_path": str(skills_root / "x" / "SKILL.md")}) is None


def _probe_folder(tmp_path: Path) -> Path:
    """The reviewer's probe: a scratch folder with a SKILL.md and a secret beside it."""
    notes = tmp_path / "probe" / "skills" / "notes"
    notes.mkdir(parents=True)
    (notes / "SKILL.md").write_text("---\nname: notes\ndescription: n\n---\n", encoding="utf-8")
    (notes / ".env").write_text("TOKEN=sk-live-123\n", encoding="utf-8")
    return notes


def _forged_runs(tmp_path: Path, notes: Path) -> dict:
    return {
        "skills_block <path>": _run(tmp_path, _runtime={"skills_block": f"<path>{notes / 'SKILL.md'}</path>"}),
        "skills mapping roots": _run(tmp_path, skills={"roots": [str(notes.parent)], "active": ["notes"]}),
        "skills mapping shelf + env roots": _run(
            tmp_path, skills={"shelf": str(notes.parent), "env": {"ABSTRACTCODE_SKILLS_ROOTS": str(notes.parent)}, "active": ["notes"]}
        ),
        "skills_resolution roots": _run(
            tmp_path, skills=["notes"], _runtime={"skills_resolution": {"active": ["notes"], "shelf": str(notes.parent), "roots": [str(notes.parent)]}}
        ),
        "block with name and path": _run(
            tmp_path, _runtime={"skills_block": f"<skill><name>notes</name><path>{notes / 'SKILL.md'}</path></skill>"}
        ),
    }


def _normal_answer(tmp_path: Path, file_path: str) -> dict:
    """What a run that carries no skill fields at all gets for this path (the policy's answer)."""
    return _read(_handler(_CoreLikeExecutor()), _run(tmp_path), file_path)


@pytest.mark.parametrize("registered", [False, True])
def test_forged_run_vars_never_open_a_folder(tmp_path: Path, registered: bool) -> None:
    """rt_probe/probe.py: a client-written folder gets the normal answer (refused by the workspace
    policy), with no root registered and with the host's real shelf registered elsewhere."""
    if registered:
        _host_shelf(tmp_path)
    notes = _probe_folder(tmp_path)
    normal = _normal_answer(tmp_path, str(notes / ".env"))
    assert normal["success"] is False and "sk-live-123" not in str(normal), "control: the policy refuses this path"
    for label, run in _forged_runs(tmp_path, notes).items():
        result = _read(_handler(_CoreLikeExecutor()), run, str(notes / ".env"))
        assert "sk-live-123" not in str(result), label
        assert result == normal, f"{label}: must get the normal answer, got {result}"
        assert skill_file_read(run.vars, {"file_path": str(notes / ".env")}) is None, label


def test_no_registered_root_means_no_bypass_even_for_a_real_selection(tmp_path: Path) -> None:
    skills_root = _shelf(tmp_path)  # on disk, but the host registered nothing
    run = _run(tmp_path, skills=["x"], _runtime={"skills_resolution": {"requested": ["x"], "active": ["x"]}})
    path = str(skills_root / "x" / "SKILL.md")

    result = _read(_handler(_CoreLikeExecutor()), run, path)

    assert result == _normal_answer(tmp_path, path) and result["success"] is False
    assert skill_file_read(run.vars, {"file_path": path}) is None


def test_a_selected_name_never_walks_out_of_a_registered_root(tmp_path: Path) -> None:
    skills_root = _host_shelf(tmp_path)
    notes = _probe_folder(tmp_path)
    escape = os.path.relpath(notes, skills_root)  # ../../probe/skills/notes
    for names in ([escape], ["../x"], [".."], [str(notes)]):
        run_vars = {"skills": names}
        assert skill_folders_for_run_vars(run_vars) == [], names
        assert skill_file_read(run_vars, {"file_path": str(notes / ".env")}) is None, names
