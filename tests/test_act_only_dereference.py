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


def test_unknown_act_only_tool_tombstones_with_loud_warning() -> None:
    """The wedge amendment (agent's 0013/080636Z finding): refs are durable,
    so a fatal failure would re-fail every later call in the run — one bad
    historical ref degrades ONE message (labeled tombstone + #FALLBACK),
    never kills the visit. Never silent, never raw ref JSON on the wire."""
    bogus = json.dumps({"$act_only": {"tool": "read_file", "entry_id": "x"}})
    warnings: list = []
    wire, n = dereference_act_only_messages(
        [{"role": "tool", "content": bogus}], read_entry=lambda r: "w", warnings=warnings
    )
    assert n == 1
    content = wire[0]["content"]
    assert content.startswith("[act-only content unavailable:")
    assert "read_file" in content and "x" in content
    assert "$act_only" not in content  # never the raw ref JSON
    assert len(warnings) == 1 and warnings[0].startswith("#FALLBACK")


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

        # Unresolvable ref: SURVIVABLE labeled tombstone (the wedge
        # amendment) — the call proceeds, the degradation is loud in the
        # wire text AND the durable result; the raw ref JSON never ships.
        seen.clear()
        bad = wrapped(_Run(), Effect(type=EffectType.LLM_CALL, payload={
            "messages": [{"role": "tool", "content":
                          make_act_only_content(tool="diary_read", entry_id="diary_00dead")}],
        }), None)
        assert bad.status == "completed"
        assert any(w.startswith("#FALLBACK") and "diary_00dead" in w
                   for w in bad.result.get("act_only_warnings", []))
        wire_content = seen[0]["messages"][0]["content"]
        assert wire_content.startswith("[act-only content unavailable:")
        assert "diary_00dead" in wire_content
        assert "$act_only" not in wire_content  # never raw ref JSON on the wire
        assert PRIVATE_WORDS not in json.dumps(seen)  # and never the words

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


def test_diary_list_ref_reruns_the_listing_at_send_time(tmp_path: Path) -> None:
    """e-s 233 R3 (adversary-broken claim, now fixed): diary_list is
    act-only on the durable lane — its ref carries tool+args and the
    LISTING is re-run fresh at send time. Private gists appear on the
    WIRE only (the entity's own book, its own eyes); the durable payload
    keeps the word-free ref."""
    from abstractruntime.identity.chat import open_home
    from abstractruntime.identity.tools import _run_diary_list

    home = open_home(_make_home(tmp_path, slug="listling"))
    try:
        _write_entry(home.handlers, PRIVATE_WORDS, visibility="private", gist="the bridges fear")
        _write_entry(home.handlers, "A public thought about rivers.", gist="rivers thought")
        seen: List[Dict[str, Any]] = []

        def fake_llm(run: Any, effect: Effect, dnn: Any) -> EffectOutcome:
            seen.append(effect.payload)
            return EffectOutcome.completed({"content": "reply"})

        wrapped = wrap_llm_handler_with_act_only(
            fake_llm,
            diary_read_handler=home.handlers[EffectType.DIARY_READ],
            diary_list_resolver=lambda ref: _run_diary_list(
                home.diary, str((ref.get("args") or {}).get("body") or "")
            ),
        )
        original_payload = {
            "messages": [
                {"role": "tool", "tool_call_id": "c1",
                 "content": make_act_only_content(tool="diary_list", args={"body": "5"})},
            ],
        }
        out = wrapped(_Run(), Effect(type=EffectType.LLM_CALL, payload=original_payload), None)
        assert out.status == "completed"
        wire = seen[0]["messages"][0]["content"]
        # The wire carries the fresh listing — both entries, gists included
        # (in-flight to the entity itself is legal, R2).
        assert "the bridges fear" in wire and "rivers thought" in wire
        assert wire.startswith("[diary_list - resolved from the book at send time]")
        # The durable payload keeps ONLY the ref — no gist words at rest.
        durable = json.dumps(original_payload)
        assert "bridges" not in durable and "rivers" not in durable
        assert parse_act_only_ref(original_payload["messages"][0]["content"])["tool"] == "diary_list"

        # No resolver wired -> loud survivable tombstone, never a leak.
        unwired = wrap_llm_handler_with_act_only(
            fake_llm, diary_read_handler=home.handlers[EffectType.DIARY_READ]
        )
        seen.clear()
        out2 = unwired(_Run(), Effect(type=EffectType.LLM_CALL, payload={
            "messages": [{"role": "tool", "content":
                          make_act_only_content(tool="diary_list", args={"body": "5"})}],
        }), None)
        assert out2.status == "completed"
        assert any("#FALLBACK" in w and "diary_list" in w
                   for w in out2.result.get("act_only_warnings", []))
        assert seen[0]["messages"][0]["content"].startswith("[act-only content unavailable:")
    finally:
        home.close()


def test_raising_resolver_tombstones_never_kills_the_call(tmp_path: Path) -> None:
    """Adversary find 2 (2026-07-11): a RAISED resolver/store error (sqlite
    failure, resolver bug) gets the same survivability as a structured
    refusal — tombstone + #FALLBACK, never a failed effect (which would
    terminal-FAIL the visit, the mechanic-4 wedge class)."""
    from abstractruntime.identity.chat import open_home

    home = open_home(_make_home(tmp_path, slug="raisling"))
    try:
        seen: List[Dict[str, Any]] = []

        def fake_llm(run: Any, effect: Effect, dnn: Any) -> EffectOutcome:
            seen.append(effect.payload)
            return EffectOutcome.completed({"content": "reply"})

        def exploding_resolver(ref: Dict[str, Any]) -> str:
            raise RuntimeError("sqlite disk I/O error")

        wrapped = wrap_llm_handler_with_act_only(
            fake_llm,
            diary_read_handler=home.handlers[EffectType.DIARY_READ],
            diary_list_resolver=exploding_resolver,
        )
        out = wrapped(_Run(), Effect(type=EffectType.LLM_CALL, payload={
            "messages": [{"role": "tool", "content":
                          make_act_only_content(tool="diary_list", args={"body": "5"})}],
        }), None)
        assert out.status == "completed"  # the visit survives
        assert any("#FALLBACK" in w and "disk I/O" in w
                   for w in out.result.get("act_only_warnings", []))
        assert seen[0]["messages"][0]["content"].startswith("[act-only content unavailable:")
    finally:
        home.close()


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
