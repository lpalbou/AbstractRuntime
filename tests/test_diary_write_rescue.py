"""A failed book write must not throw the entity's reply away.

Record-everything ruling (laurent, 2026-07-26): everything is kept; privacy
is access control, never erasure. Gateway's strip-review found the one gap
(V1): when the diary write itself fails - disk or store error - the turn
failed loudly but the whole reply, including the words the entity elected
to keep, was gone. Now the raw reply is saved to <home>/rescue/ first,
labeled with run and turn ids so a repair can replay the write.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from abstractruntime.core.models import Effect, EffectType, RunState
from abstractruntime.core.runtime import EffectOutcome
from abstractruntime.identity.act_only import (
    ActOnlyResolutionError,
    capture_diary_elections,
    rescue_reply_to_home,
    wrap_llm_handler_with_act_only,
)

REPLY = "Before the fence.\n```diary\nkind: note\nwords I chose to keep\n```\nAfter."


def _run() -> RunState:
    return RunState.new(workflow_id="wf", entry_node="n", session_id="s1", vars={})


def _failing_diary_write(run: Any, effect: Effect, nxt: Any) -> EffectOutcome:
    return EffectOutcome.failed("disk full: cannot append to the book")


def test_failed_book_write_rescues_the_reply(tmp_path: Path) -> None:
    with pytest.raises(ActOnlyResolutionError) as exc:
        capture_diary_elections(
            REPLY,
            run=_run(),
            turn_id="t-0001",
            diary_write_handler=_failing_diary_write,
            rescue_dir=tmp_path,
        )
    # The failure stays loud and names the rescue file.
    assert "disk full" in str(exc.value)
    assert "rescued to" in str(exc.value)

    files = list((tmp_path / "rescue").glob("reply_*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["raw_reply"] == REPLY  # words verbatim, nothing lost
    assert saved["turn_id"] == "t-0001"
    assert "disk full" in saved["error"]


def test_missing_turn_id_also_rescues(tmp_path: Path) -> None:
    with pytest.raises(ActOnlyResolutionError):
        capture_diary_elections(
            REPLY,
            run=_run(),
            turn_id="",
            diary_write_handler=_failing_diary_write,
            rescue_dir=tmp_path,
        )
    files = list((tmp_path / "rescue").glob("reply_*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8"))["raw_reply"] == REPLY


def test_rescue_failure_is_named_not_hidden(tmp_path: Path) -> None:
    """If the rescue itself fails (same disk trouble), the loud failure
    says so instead of pretending the words were saved."""
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the rescue dir should go", encoding="utf-8")
    with pytest.raises(ActOnlyResolutionError) as exc:
        capture_diary_elections(
            REPLY,
            run=_run(),
            turn_id="t-0001",
            diary_write_handler=_failing_diary_write,
            rescue_dir=blocked,  # mkdir under a FILE fails -> rescue fails
        )
    assert "rescue also failed" in str(exc.value)


def test_no_rescue_dir_keeps_the_old_loud_failure(tmp_path: Path) -> None:
    """Callers that pass no home (plain workflow lanes) keep the exact
    pre-ruling behavior - loud failure, no rescue claim in the message."""
    with pytest.raises(ActOnlyResolutionError) as exc:
        capture_diary_elections(
            REPLY,
            run=_run(),
            turn_id="t-0001",
            diary_write_handler=_failing_diary_write,
        )
    assert "rescued" not in str(exc.value)


def test_wrapped_handler_threads_the_rescue_dir(tmp_path: Path) -> None:
    def llm(run: Any, effect: Effect, nxt: Any) -> EffectOutcome:
        return EffectOutcome.completed({"content": REPLY})

    wrapped = wrap_llm_handler_with_act_only(
        llm, diary_write_handler=_failing_diary_write, rescue_dir=tmp_path
    )
    out = wrapped(_run(), Effect(type=EffectType.LLM_CALL, payload={"turn_id": "t-0002"}), None)
    assert out.status == "failed"
    files = list((tmp_path / "rescue").glob("reply_*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8"))["raw_reply"] == REPLY


def test_rescue_helper_never_raises(tmp_path: Path) -> None:
    assert rescue_reply_to_home(None, run_id="r", turn_id="t", raw_reply="x", error="e") is None
    p = rescue_reply_to_home(tmp_path, run_id="r/../weird id", turn_id="t?", raw_reply="x", error="e")
    assert p is not None and Path(p).exists()  # hostile ids sanitized into the filename
