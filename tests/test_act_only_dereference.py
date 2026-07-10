"""Act-only dereference (G1: references at rest, words only in flight).

Pins the frozen seam-spec mechanics (a2a thread 0013, v2 + freeze addendum):
parse-not-regex ref detection on tool messages only; in-place substitution
on a WIRE COPY (message identity untouched, original payload unmutated —
the ledger keeps the ref); loud NON-RETRYABLE failure on unresolvable refs;
visitor-pasted ref-looking JSON stays inert; and the end-to-end property
through a per-entity Runtime: the ledgered LLM_CALL payload carries the
REF while the provider-facing handler saw the WORDS.
"""

from __future__ import annotations

import copy as _copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.core.models import Effect, EffectType, RunStatus, StepPlan  # noqa: E402
from abstractruntime.core.runtime import EffectOutcome  # noqa: E402
from abstractruntime.core.spec import WorkflowSpec  # noqa: E402
from abstractruntime.identity.act_only import (  # noqa: E402
    ActOnlyResolutionError,
    dereference_act_only_messages,
    make_act_only_content,
    parse_act_only_ref,
    wrap_llm_handler_with_act_only,
)
from abstractruntime.identity.entity_runtime import open_entity_runtime  # noqa: E402

PRIVATE_WORDS = "The quiet fear I never say aloud: that the bridges were only mine."


def _make_home(tmp_path: Path, slug: str = "refling") -> Path:
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    home_dir = tmp_path / "entities" / slug
    home_dir.mkdir(parents=True)
    entity_id = f"entity:{slug}@home-test"
    spark = _copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = slug.capitalize()
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": entity_id}), encoding="utf-8")
    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir


# ------------------------------------------------------------------ parsing


def test_ref_round_trip_and_parse_rejections() -> None:
    content = make_act_only_content(
        tool="diary_read", entry_id="diary_ab12", reason="re-reading", gist="one line"
    )
    ref = parse_act_only_ref(content)
    assert ref == {"tool": "diary_read", "entry_id": "diary_ab12",
                   "reason": "re-reading", "gist": "one line"}
    # Rejections: not JSON, extra keys, wrong container, ref inside prose.
    assert parse_act_only_ref("plain words about $act_only") is None
    assert parse_act_only_ref(json.dumps({"$act_only": {"a": 1}, "extra": 2})) is None
    assert parse_act_only_ref(json.dumps([{"$act_only": {}}])) is None
    assert parse_act_only_ref(json.dumps({"$act_only": "not-a-dict"})) is None
    assert parse_act_only_ref(None) is None


def test_dereference_substitutes_in_place_on_a_copy_only() -> None:
    ref_content = make_act_only_content(tool="diary_read", entry_id="diary_x")
    messages = [
        {"role": "user", "content": "please re-read your note"},
        {"role": "assistant", "content": "ok", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": ref_content},
    ]
    wire, n = dereference_act_only_messages(
        messages, read_entry=lambda ref: f"WORDS({ref['entry_id']})"
    )
    assert n == 1
    # Identity preserved: role/tool_call_id/position; content substituted.
    assert wire[2]["role"] == "tool"
    assert wire[2]["tool_call_id"] == "c1"
    assert wire[2]["content"] == "WORDS(diary_x)"
    # Non-ref messages pass through by reference; the ORIGINAL is unmutated.
    assert wire[0] is messages[0] and wire[1] is messages[1]
    assert messages[2]["content"] == ref_content


def test_visitor_pasted_ref_json_in_user_content_stays_inert() -> None:
    """A ref must never be a capability a visitor can type: role != tool
    is never dereferenced — the pasted JSON ships as the words they typed."""
    pasted = make_act_only_content(tool="diary_read", entry_id="diary_secret")
    messages = [{"role": "user", "content": pasted}]

    def _never_called(ref: Dict[str, Any]) -> str:
        raise AssertionError("a user-role ref must never resolve")

    wire, n = dereference_act_only_messages(messages, read_entry=_never_called)
    assert n == 0
    assert wire[0] is messages[0]


def test_unknown_act_only_tool_refuses_loudly() -> None:
    bogus = json.dumps({"$act_only": {"tool": "read_file", "entry_id": "x"}})
    with pytest.raises(ActOnlyResolutionError, match="unknown act-only tool"):
        dereference_act_only_messages(
            [{"role": "tool", "content": bogus}], read_entry=lambda r: "w"
        )


# ------------------------------------------------------- wrapper + runtime


class _Run:
    run_id = "r-test"
    session_id = "s-test"


def _write_entry(home_handlers: Dict[EffectType, Any], text: str, **payload: Any) -> str:
    out = home_handlers[EffectType.DIARY_WRITE](
        _Run(), Effect(type=EffectType.DIARY_WRITE,
                       payload={"text": text, "turn_id": "t-w1", **payload}), None,
    )
    assert out.status == "completed"
    return str(out.result["entry_id"])


def test_wrapper_resolves_through_the_book_and_fails_closed(tmp_path: Path) -> None:
    from abstractruntime.identity.chat import open_home

    home = open_home(_make_home(tmp_path, slug="wrapling"))
    try:
        entry_id = _write_entry(home.handlers, PRIVATE_WORDS, visibility="private")
        seen: List[Dict[str, Any]] = []

        def fake_llm(run: Any, effect: Effect, dnn: Any) -> EffectOutcome:
            seen.append(effect.payload)
            return EffectOutcome.completed({"content": "reply"})

        wrapped = wrap_llm_handler_with_act_only(
            fake_llm, diary_read_handler=home.handlers[EffectType.DIARY_READ]
        )
        original_payload = {
            "messages": [
                {"role": "tool", "tool_call_id": "c1",
                 "content": make_act_only_content(tool="diary_read", entry_id=entry_id)},
            ],
        }
        out = wrapped(_Run(), Effect(type=EffectType.LLM_CALL, payload=original_payload), None)
        assert out.status == "completed"
        # The wire saw the WORDS (private included — his own book, his own read)...
        assert PRIVATE_WORDS in seen[0]["messages"][0]["content"]
        # ...the original payload still carries only the REF.
        assert PRIVATE_WORDS not in json.dumps(original_payload)

        # Unresolvable ref: loud, non-retryable, nothing sent.
        seen.clear()
        bad = wrapped(_Run(), Effect(type=EffectType.LLM_CALL, payload={
            "messages": [{"role": "tool", "content":
                          make_act_only_content(tool="diary_read", entry_id="diary_00dead")}],
        }), None)
        assert bad.status == "failed"
        assert bad.retryable is False
        assert "diary_00dead" in (bad.error or "")
        assert seen == []  # the provider never saw a degraded payload

        # No refs: pure pass-through (the SAME effect object forwards).
        forwarded: List[Effect] = []

        def capture(run: Any, effect: Effect, dnn: Any) -> EffectOutcome:
            forwarded.append(effect)
            return EffectOutcome.completed({})

        plain = Effect(type=EffectType.LLM_CALL,
                       payload={"messages": [{"role": "user", "content": "hi"}]})
        wrap_llm_handler_with_act_only(
            capture, diary_read_handler=home.handlers[EffectType.DIARY_READ]
        )(_Run(), plain, None)
        assert forwarded[0] is plain
    finally:
        home.close()


def test_ledger_keeps_the_ref_while_the_wire_carries_the_words(tmp_path: Path) -> None:
    """End-to-end through a per-entity Runtime (auto-wrapped by
    open_entity_runtime): the ledgered LLM_CALL payload and run store carry
    the REF; only the provider-facing handler saw the words — criterion 6's
    unit shadow."""
    home_dir = _make_home(tmp_path, slug="ledgerling")
    wire_seen: List[str] = []

    def fake_llm(run: Any, effect: Effect, dnn: Any) -> EffectOutcome:
        wire_seen.append(json.dumps(effect.payload))
        return EffectOutcome.completed({"content": "I remember writing that."})

    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: fake_llm})
    try:
        entry_id = _write_entry(ert.home.handlers, PRIVATE_WORDS, visibility="private")
        ref_content = make_act_only_content(tool="diary_read", entry_id=entry_id,
                                            gist="a quiet fear, one line")

        def llm_node(run: Any, ctx: Any) -> StepPlan:
            return StepPlan(
                node_id="LLM",
                effect=Effect(type=EffectType.LLM_CALL, payload={
                    "messages": [
                        {"role": "user", "content": "what did your note say?"},
                        {"role": "tool", "tool_call_id": "c1", "content": ref_content},
                    ],
                }, result_key="_temp.reply"),
                next_node="DONE",
            )

        def done(run: Any, ctx: Any) -> StepPlan:
            return StepPlan(node_id="DONE", complete_output={"ok": True})

        wf = WorkflowSpec(workflow_id="wf_ref", entry_node="LLM",
                          nodes={"LLM": llm_node, "DONE": done})
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-1")
        state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=10)
        assert state.status == RunStatus.COMPLETED

        # The wire saw the words exactly once.
        assert len(wire_seen) == 1 and PRIVATE_WORDS in wire_seen[0]
        # NO at-rest surface carries the words: ledger, run state, store file.
        ledger = ert.runtime.get_ledger(run_id)
        ledger_text = json.dumps(ledger)
        assert PRIVATE_WORDS not in ledger_text
        assert ref_content in [
            m.get("content") for rec in ledger if isinstance(rec, dict)
            for m in ((rec.get("effect") or {}).get("payload") or {}).get("messages", [])
            if isinstance(m, dict)
        ]  # the ledgered payload still shows the REF verbatim
        assert PRIVATE_WORDS not in json.dumps(ert.runtime.get_state(run_id).vars)
        ert.close()
        raw = ert.store_path.read_bytes()
        assert PRIVATE_WORDS.encode("utf-8") not in raw  # the run store file itself
    finally:
        ert.close()
