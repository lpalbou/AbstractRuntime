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
"""

from __future__ import annotations

import os
from pathlib import Path

from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.skill_roots import (
    skill_file_read,
    skill_folders_for_run_vars,
)
from abstractruntime.storage.in_memory import InMemoryRunStore

SKILL_TEXT = "---\nname: x\ndescription: Demo skill\n---\n\n# X\n\nUse it well.\n"
REFERENCE_TEXT = "one\ntwo\nthree\nfour\n"


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
    skills_root = _shelf(tmp_path)
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
    skills_root = _shelf(tmp_path)
    executor = _CoreLikeExecutor()
    run = _run(tmp_path, skills={"shelf": str(skills_root), "active": ["x"]})
    path = str(skills_root / "x" / "references" / "ops.md")

    result = _read(_handler(executor), run, path, start_line=2, end_line=3)

    assert result["success"] is True
    assert result["output"] == f"File: {path} (2 lines)\n\n2: two\n3: three"
    assert executor.calls == []


def test_a_path_outside_the_skill_folders_still_goes_to_core(tmp_path: Path) -> None:
    skills_root = _shelf(tmp_path)
    executor = _CoreLikeExecutor()
    run = _run(tmp_path, skills={"shelf": str(skills_root), "active": ["x"]})
    notes = tmp_path / "ws" / "notes.md"
    notes.write_text("n\n", encoding="utf-8")

    result = _read(_handler(executor), run, str(notes))

    assert len(executor.calls) == 1 and executor.calls[0]["name"] == "read_file"
    assert "ignored by .abstractignore policy" in str(result["output"])


def test_a_symlink_leading_out_of_the_folder_is_refused(tmp_path: Path) -> None:
    skills_root = _shelf(tmp_path)
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
    skills_root = _shelf(tmp_path)
    executor = _CoreLikeExecutor()
    run = _run(
        tmp_path,
        skills=["x", "held-one"],
        _runtime={"skills_resolution": {"requested": ["x", "held-one"], "active": ["x"], "shelf": str(skills_root)}},
    )

    result = _read(_handler(executor), run, str(skills_root / "x" / "SKILL.md"))

    assert result["success"] is True and "Use it well." in result["output"]
    assert executor.calls == []


def test_the_skills_block_path_names_the_folder(tmp_path: Path) -> None:
    """No shelf anywhere in vars: the <path> AbstractSkill renders into the block is the record."""
    skills_root = _shelf(tmp_path)
    executor = _CoreLikeExecutor()
    block = (
        "<available_skills>\n  <skill>\n    <name>x</name>\n    <description>Demo skill</description>\n"
        f"    <path>{skills_root / 'x' / 'SKILL.md'}</path>\n  </skill>\n</available_skills>"
    )
    run = _run(tmp_path, skills=["x"], _runtime={"skills_block": block, "skills_resolution": {"active": ["x"]}})

    result = _read(_handler(executor), run, str(skills_root / "x" / "references" / "ops.md"))

    assert result["success"] is True and result["output"].endswith("4: four")
    assert executor.calls == []


def test_folder_derivation_follows_the_gateway_spellings(tmp_path: Path) -> None:
    skills_root = _shelf(tmp_path)  # <root>/skills/<name>
    flat_root = tmp_path / "flat"  # <root>/<name>
    (flat_root / "y").mkdir(parents=True)
    (flat_root / "y" / "SKILL.md").write_text("---\nname: y\ndescription: Y\n---\n", encoding="utf-8")
    imported = tmp_path / "imported"
    (imported / "z").mkdir(parents=True)
    (imported / "z" / "SKILL.md").write_text("---\nname: z\ndescription: Z\n---\n", encoding="utf-8")

    folders = skill_folders_for_run_vars(
        {
            "skills": {
                "shelf": str(skills_root.parent),  # the shelf folder whose skills/ holds x
                "imported_shelf": str(imported),
                "ABSTRACTCODE_SKILLS_ROOTS": f"{flat_root}{os.pathsep}{tmp_path / 'nowhere'}",
                "active": ["x", "y", "z", "held@deadbeef"],
                "requested": ["x", "y", "z", "held", "absent"],
            }
        }
    )

    assert {f.name for f in folders} == {"x", "y", "z"}
    assert all((f / "SKILL.md").is_file() for f in folders)
    # A held (not active) or absent name never yields a folder; nothing is served for it.
    assert skill_file_read({"skills": {"shelf": str(skills_root), "active": []}}, {"file_path": str(skills_root / "x" / "SKILL.md")}) is None
    # No skills at all: every path falls through to core.
    assert skill_file_read({}, {"file_path": str(skills_root / "x" / "SKILL.md")}) is None
