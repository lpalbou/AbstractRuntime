"""Diary election capture at the LLM result boundary (the write half).

The read half (refs/dereference) was DELETED per laurent's A ruling
(2026-07-20); these pins cover what remains: the case-insensitive capture
gate and word-free metadata for private entries.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("abstractmemory")

from abstractruntime.core.models import Effect, EffectType  # noqa: E402
from abstractruntime.core.runtime import EffectOutcome  # noqa: E402
from abstractruntime.identity.act_only import (  # noqa: E402
    ActOnlyResolutionError,
    capture_diary_elections,
    wrap_llm_handler_with_act_only,
)

import json  # noqa: E402

PRIVATE_WORDS = "the secret only my book holds"


class _Run:
    run_id = "run-capture"
    vars: Dict[str, Any] = {}


def test_capture_gate_is_case_insensitive() -> None:
    """Adversary find 3 (2026-07-11): the fence parser is IGNORECASE but
    the wrapper's capture gate was a lowercase substring check — a model
    emitting ```Diary would skip capture and rest the raw private words in
    run vars + ledger while silently losing the book write."""
    calls: List[Dict[str, Any]] = []

    class _Out:
        status = "completed"
        result = {"entry_id": "diary_ab12cd34", "projected_record_id": None, "warnings": []}

    def fake_diary_write(run: Any, effect: Effect, dnn: Any) -> Any:
        calls.append(effect.payload)
        return _Out()

    def fake_llm(run: Any, effect: Effect, dnn: Any) -> EffectOutcome:
        return EffectOutcome.completed({
            "content": "kept.\n```Diary kind=note visibility=private\n" + PRIVATE_WORDS + "\n```",
        })

    wrapped = wrap_llm_handler_with_act_only(
        fake_llm,
        diary_read_handler=lambda *a: None,
        diary_write_handler=fake_diary_write,
    )
    out = wrapped(_Run(), Effect(type=EffectType.LLM_CALL, payload={
        "messages": [{"role": "user", "content": "keep a note"}], "turn_id": "t-case",
    }), None)
    assert out.status == "completed"
    assert calls, "the uppercase fence must still reach the book"
    assert PRIVATE_WORDS not in str(out.result.get("content") or "")


def test_private_diary_meta_carries_no_words() -> None:
    """The proximity pin (memory's e-s 233 rider): capture metadata for a
    PRIVATE entry must never grow gist/text keys — the visit sheet trusts
    this omission (it rests in run vars + pending_reflection.json and
    rides the reflection prompt graph-ward)."""
    calls: List[Dict[str, Any]] = []

    class _Out:
        status = "completed"
        result = {"entry_id": "diary_ab12cd34", "projected_record_id": "ex:p1", "warnings": []}

    def fake_diary_write(run: Any, effect: Effect, dnn: Any) -> Any:
        calls.append(effect.payload)
        return _Out()

    from abstractruntime.identity.act_only import capture_diary_elections

    marked, entries, _ = capture_diary_elections(
        "kept.\n```diary kind=note visibility=private\ngist: secret gist\n" + PRIVATE_WORDS + "\n```",
        run=_Run(), turn_id="t-1", diary_write_handler=fake_diary_write,
    )
    assert PRIVATE_WORDS not in marked
    assert len(entries) == 1
    meta_str = json.dumps(entries[0])
    assert "gist" not in entries[0] and "text" not in entries[0]
    assert "secret gist" not in meta_str and PRIVATE_WORDS not in meta_str
    assert entries[0]["visibility"] == "private"


# ---------------------------------------------------------------------------
# Stamp-conditional capture for SHARED runtimes (flow c5342 R1 — the Mira
# diary leak: the flow-brain lane's base-runtime LLM_CALL had no capture
# wrap, so elected fences passed through unparsed and private words rested
# in ledgers). Runtime exports the composition helper; gateway wires its
# stamp resolver as the predicate.
# ---------------------------------------------------------------------------


def test_conditional_capture_passthrough_for_unstamped_runs() -> None:
    """No stamp = byte-identical passthrough: the base handler's outcome
    object comes back untouched (raw fences INCLUDED — a plain workflow run
    is not an entity turn and must not be rewritten)."""
    from abstractruntime.identity.act_only import wrap_llm_handler_with_conditional_capture

    raw = EffectOutcome.completed({
        "content": "plain run.\n```diary kind=note\nnot an entity\n```",
    })

    def fake_llm(run: Any, effect: Effect, dnn: Any) -> EffectOutcome:
        return raw

    wrapped = wrap_llm_handler_with_conditional_capture(
        fake_llm,
        should_capture=lambda run: False,
        diary_write_for_run=lambda run: None,
    )
    out = wrapped(_Run(), Effect(type=EffectType.LLM_CALL, payload={"turn_id": "t-plain"}), None)
    assert out is raw, "unstamped runs get the SAME outcome object - byte-identical"


def test_conditional_capture_captures_for_stamped_runs() -> None:
    """Stamped run = the full G1 write boundary: fences fly to the resolved
    home book, the durable result keeps the MARKED reply."""
    from abstractruntime.identity.act_only import wrap_llm_handler_with_conditional_capture

    calls: List[Dict[str, Any]] = []

    class _Out:
        status = "completed"
        result = {"entry_id": "diary_ff00aa11", "projected_record_id": None, "warnings": []}

    def fake_diary_write(run: Any, effect: Effect, dnn: Any) -> Any:
        calls.append(effect.payload)
        return _Out()

    def fake_llm(run: Any, effect: Effect, dnn: Any) -> EffectOutcome:
        return EffectOutcome.completed({
            "content": "kept.\n```diary kind=note visibility=private\n" + PRIVATE_WORDS + "\n```",
        })

    wrapped = wrap_llm_handler_with_conditional_capture(
        fake_llm,
        should_capture=lambda run: True,
        diary_write_for_run=lambda run: fake_diary_write,
    )
    out = wrapped(_Run(), Effect(type=EffectType.LLM_CALL, payload={
        "messages": [{"role": "user", "content": "note it"}], "turn_id": "t-stamped",
    }), None)
    assert calls, "the fence flew to the book"
    assert PRIVATE_WORDS not in json.dumps(out.result), "the durable result is MARKED, never raw"


def test_conditional_capture_refuses_stamped_run_without_a_book() -> None:
    """should_capture=True with no resolvable DIARY_WRITE must RAISE: a
    stamped run without its book would leak elected words - failing the
    effect is strictly better than either leak direction."""
    from abstractruntime.identity.act_only import wrap_llm_handler_with_conditional_capture

    wrapped = wrap_llm_handler_with_conditional_capture(
        lambda run, effect, dnn: EffectOutcome.completed({"content": "x"}),
        should_capture=lambda run: True,
        diary_write_for_run=lambda run: None,
    )
    with pytest.raises(RuntimeError) as exc_info:
        wrapped(_Run(), Effect(type=EffectType.LLM_CALL, payload={"turn_id": "t-nobook"}), None)
    assert "refusing loudly" in str(exc_info.value)
