"""Wave-5 narrator half (dm#75 ratified): trigger set, throttle, fenced
voice, self-label, wake residue - against memory's frozen signal shape
(c3708)."""
from __future__ import annotations

import json
from pathlib import Path

from abstractruntime.identity.night_voice import (
    NARRATION_SELF_LABEL,
    build_narration_prompt,
    narration_throttle_ok,
    narration_trigger,
    run_night_voice,
    wake_residue,
)


def _sig(kind="changed_understanding", phase="resolution", act="dream_resolved",
         fragment="the tide question settled", felt=None):
    s = {"kind": kind, "phase": phase, "act": act, "fragment": fragment, "touched": ["ex:r1"]}
    if felt:
        s["felt"] = felt
    return s


def test_trigger_set_is_the_ratified_three() -> None:
    # scar/bond touched wins first.
    assert narration_trigger(
        [_sig(phase="mining", act="candidate_minted",
              felt={"tone": "sore", "weight": 3.0, "scarred": True, "bonded": False})],
    ) == "scar_or_bond_touched"
    # resolution surfaced.
    assert narration_trigger([_sig()]) == "resolution_surfaced"
    # salience bar.
    assert narration_trigger(
        [_sig(phase="mining", act="grouped")], salience=60,
    ) == "salience_bar"
    # quiet night: none.
    assert narration_trigger([_sig(phase="mining", act="grouped")], salience=10) is None
    assert narration_trigger([]) is None


def test_throttle_is_20h(tmp_path: Path) -> None:
    assert narration_throttle_ok(tmp_path), "no stamp = first night, open"
    from datetime import datetime, timedelta, timezone

    recent = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    (tmp_path / "night_voice.json").write_text(
        json.dumps({"last_narration_at": recent}), encoding="utf-8")
    assert not narration_throttle_ok(tmp_path), "5h ago = throttled"
    old = (datetime.now(timezone.utc) - timedelta(hours=21)).isoformat()
    (tmp_path / "night_voice.json").write_text(
        json.dumps({"last_narration_at": old}), encoding="utf-8")
    assert narration_throttle_ok(tmp_path), "21h ago = open"


def test_night_voice_strips_fences_labels_and_rests(tmp_path: Path) -> None:
    """The fenced-voice constraint: the sleeping mind elects nothing - a
    ```diary fence in the narration is stripped, never written; the
    narration rests home-side with the self-label; the stamp closes the
    throttle."""

    class _LLM:
        def generate(self, *, messages, system_prompt):
            class _R:
                content = ("The night felt like sorting stones by warmth."
                           "\n```diary kind=note\nsneaky election\n```\nQuiet now.")
            return _R()

    out = run_night_voice(
        tmp_path, llm=_LLM(), prelude_text="I am Testling.",
        signals=[_sig()], dream_record_id="ex:dream-1", salience=50,
        out=lambda s: None,
    )
    assert out["narrated"] is True and out["trigger"] == "resolution_surfaced"
    assert "sneaky election" not in out["text"], "fences stripped"
    rows = [json.loads(l) for l in (tmp_path / "night_narrations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["self_label"] == NARRATION_SELF_LABEL
    assert rows[0]["fences_stripped"] is True
    assert "sneaky election" not in rows[0]["narration"]
    # Second call same night: throttled, zero LLM.
    out2 = run_night_voice(
        tmp_path, llm=_LLM(), prelude_text="I am Testling.",
        signals=[_sig()], dream_record_id="ex:dream-2", salience=50,
        out=lambda s: None,
    )
    assert out2["narrated"] is False and "throttled" in out2["reason"]


def test_wake_residue_is_fragments_only_and_capped() -> None:
    sigs = [_sig(fragment=f"fragment number {i} with some words") for i in range(6)]
    residue = wake_residue(sigs)
    assert "the night left traces:" in residue
    assert len(residue) <= 200
    assert "fragment number 0" in residue
    assert "changed_understanding" not in residue, "kinds/stream never ride the cue"
    assert wake_residue([]) == ""


def test_prompt_carries_no_tool_offer() -> None:
    p = build_narration_prompt("PRELUDE", [_sig()], ["- warm toward person:x"])
    assert "No\ntools exist here" in p or "No tools" in p.replace("\n", " ")
    assert "```tool" not in p and "```diary" not in p
    assert "the tide question settled" in p
