"""Durable session conversation replay — read-side contract tests.

Contract (agora `durable-sessions` v1): the run store is the single durable
source of a session's conversation; `session_chat_messages` reconstructs it
as chat messages for host-side seeding of new runs in the same session.
"""

from __future__ import annotations

import inspect

import pytest

from abstractruntime import (
    HISTORY_REPLAY_MAX_TOKENS,
    InMemoryLedgerStore,
    InMemoryRunStore,
    RunState,
    RunStatus,
    SESSION_TURN_KIND,
    fold_history_window,
    session_chat_messages,
)
from abstractruntime.memory.token_budget import estimate_message_tokens


def _turn_run(
    *,
    run_id: str,
    session_id: str = "sess-1",
    prompt: str = "",
    answer: str = "",
    status: RunStatus = RunStatus.COMPLETED,
    created_at: str = "2026-01-01T00:00:00+00:00",
    workflow_id: str = "wf_chat",
    parent_run_id: str | None = None,
) -> RunState:
    return RunState(
        run_id=run_id,
        workflow_id=workflow_id,
        status=status,
        current_node="done",
        vars={
            "prompt": prompt,
            "context": {"task": prompt, "messages": []},
        },
        output={"response": answer} if answer else {},
        error=None,
        created_at=created_at,
        updated_at=created_at,
        actor_id="tester",
        session_id=session_id,
        parent_run_id=parent_run_id,
        waiting=None,
    )


def test_session_chat_messages_reconstructs_turns_chronologically() -> None:
    run_store = InMemoryRunStore()
    run_store.save(
        _turn_run(
            run_id="run-1",
            prompt="analyze the PDF",
            answer="The document concludes X.",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    run_store.save(
        _turn_run(
            run_id="run-2",
            prompt="explain simply",
            answer="In short: X.",
            created_at="2026-01-01T00:05:00+00:00",
        )
    )

    messages = session_chat_messages(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        session_id="sess-1",
    )

    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in messages] == [
        "analyze the PDF",
        "The document concludes X.",
        "explain simply",
        "In short: X.",
    ]
    for m in messages:
        assert m["metadata"]["kind"] == SESSION_TURN_KIND
        assert m["metadata"]["run_id"] in {"run-1", "run-2"}
        assert m["metadata"]["ts"]


def test_session_chat_messages_strips_runtime_metadata_envelope() -> None:
    run_store = InMemoryRunStore()
    run_store.save(
        _turn_run(
            run_id="run-env",
            prompt='<runtime_metadata>{"display":"[t]"}</runtime_metadata>\nreal question',
            answer="real answer",
        )
    )

    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1"
    )

    assert messages[0]["content"] == "real question"


def test_session_chat_messages_skips_incomplete_internal_and_child_runs() -> None:
    run_store = InMemoryRunStore()
    # Contributing turn.
    run_store.save(_turn_run(run_id="run-ok", prompt="q1", answer="a1"))
    # Failed run: half-turn stays invisible (contract v1, question (a)).
    run_store.save(
        _turn_run(
            run_id="run-failed",
            prompt="q-failed",
            answer="",
            status=RunStatus.FAILED,
            created_at="2026-01-01T00:01:00+00:00",
        )
    )
    # Completed but silent (no answer): invisible.
    run_store.save(
        _turn_run(
            run_id="run-silent",
            prompt="q-silent",
            answer="",
            created_at="2026-01-01T00:02:00+00:00",
        )
    )
    # Still RUNNING (mid-turn): invisible — it has not answered yet.
    run_store.save(
        _turn_run(
            run_id="run-running",
            prompt="q-running",
            answer="",
            status=RunStatus.RUNNING,
            created_at="2026-01-01T00:02:30+00:00",
        )
    )
    # Internal bookkeeping run: invisible.
    run_store.save(
        _turn_run(
            run_id="session_memory_sess-1",
            prompt="",
            answer="",
            workflow_id="__session_memory__",
            created_at="2026-01-01T00:03:00+00:00",
        )
    )
    # Child run of a turn: only ROOT runs are turns.
    run_store.save(
        _turn_run(
            run_id="run-child",
            prompt="child prompt",
            answer="child answer",
            parent_run_id="run-ok",
            created_at="2026-01-01T00:04:00+00:00",
        )
    )

    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1"
    )

    assert [m["metadata"]["run_id"] for m in messages] == ["run-ok", "run-ok"]
    # Pairs-only invariant (runtime review A3): strictly alternating
    # user/assistant, never a dangling half-turn.
    assert [m["role"] for m in messages] == ["user", "assistant"]


class _CountingLedgerStore(InMemoryLedgerStore):
    def __init__(self) -> None:
        super().__init__()
        self.list_calls = 0

    def list(self, run_id: str):  # type: ignore[override]
        self.list_calls += 1
        return super().list(run_id)


def test_session_chat_messages_reads_no_ledgers_when_outputs_carry_answers() -> None:
    """Runtime review A1 (cost contract): the seed read is O(turns) run
    loads — the ledger is touched only as a per-run fallback when a
    completed run's output carries no answer."""
    run_store = InMemoryRunStore()
    for i in range(3):
        run_store.save(
            _turn_run(
                run_id=f"run-{i}",
                prompt=f"q{i}",
                answer=f"a{i}",
                created_at=f"2026-01-01T00:{i:02d}:00+00:00",
            )
        )
    ledger_store = _CountingLedgerStore()

    messages = session_chat_messages(
        run_store=run_store, ledger_store=ledger_store, session_id="sess-1"
    )

    assert len(messages) == 6
    assert ledger_store.list_calls == 0


def test_session_chat_messages_works_on_json_file_store(tmp_path) -> None:
    """Same contract on the production file store (the live gateway's)."""
    from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore

    run_store = JsonFileRunStore(str(tmp_path))
    run_store.save(_turn_run(run_id="run-1", prompt="q1", answer="a1"))
    run_store.save(
        _turn_run(
            run_id="run-2",
            prompt="q2",
            answer="a2",
            created_at="2026-01-01T00:05:00+00:00",
        )
    )
    run_store.save(
        _turn_run(
            run_id="run-failed",
            prompt="qf",
            answer="",
            status=RunStatus.FAILED,
            created_at="2026-01-01T00:06:00+00:00",
        )
    )

    messages = session_chat_messages(
        run_store=run_store,
        ledger_store=JsonlLedgerStore(str(tmp_path)),
        session_id="sess-1",
    )

    assert [m["content"] for m in messages] == ["q1", "a1", "q2", "a2"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]


def _pair_tokens(prompt: str, answer: str) -> int:
    return estimate_message_tokens({"role": "user", "content": prompt}) + estimate_message_tokens(
        {"role": "assistant", "content": answer}
    )


def _text_with_tokens(role: str, tokens: int) -> str:
    """A prose string whose message estimate is exactly `tokens` (the one estimator)."""
    unit = "the history window keeps whole turns "
    text = unit * (tokens // 2 + 4)
    lo, hi = 1, len(text)
    while lo < hi:  # smallest prefix whose estimate reaches `tokens`
        mid = (lo + hi) // 2
        if estimate_message_tokens({"role": role, "content": text[:mid]}) >= tokens:
            hi = mid
        else:
            lo = mid + 1
    out = text[:lo]
    assert estimate_message_tokens({"role": role, "content": out}) == tokens
    return out


def test_gateway_060_call_shape_replays_the_whole_window_and_records_the_ignored_caps(caplog) -> None:
    """AbstractGateway 0.6.0 seeds a session with EXACTLY this call (bundle_host
    `_seed_session_history`, defaults max_messages=40, max_total_chars=24000)
    and swallows any exception into `seeded=0`. Runtime 0.7.0 first refused
    the retired kwargs with a TypeError, so a 0.6.0 gateway on 0.7.0 replayed
    no history at all (tag gate B1, attack/gw060_probe.py). The retired caps
    are accepted, IGNORED — no count or char cap returns — logged, and named
    in the window's report."""
    run_store = InMemoryRunStore()
    # 30 turns = 60 messages (past the retired 40) with ~36,000 chars in total
    # (past the retired 24,000) and one 9,000-char answer (past the retired
    # 8,000 per message): all of it is inside 50k tokens, so all of it replays.
    for i in range(30):
        run_store.save(
            _turn_run(
                run_id=f"run-{i:02d}",
                prompt=f"q{i}",
                answer=("x" * 9000) if i == 3 else f"a{i} " + "y" * 900,
                created_at=f"2026-01-01T00:{i:02d}:00+00:00",
            )
        )
    limit, max_chars = 40, 24000
    with caplog.at_level("WARNING", logger="abstractruntime.session_history"):
        messages = session_chat_messages(
            run_store=run_store,
            ledger_store=InMemoryLedgerStore(),
            session_id="sess-1",
            max_messages=limit,
            max_total_chars=max_chars,
        )
    assert len(messages) == 60
    assert messages[7]["content"] == "x" * 9000
    assert all("#TRUNCATION" not in m["content"] for m in messages)
    assert messages.report["ignored_inputs"] == {"max_messages": 40, "max_total_chars": 24000}
    assert messages.report["dropped_messages"] == 0 and messages.report["replayed_messages"] == 60
    warned = [r for r in caplog.records if "retired replay cap" in r.getMessage()]
    assert len(warned) == 1 and "max_messages=40" in warned[0].getMessage()

    # The third retired cap never cuts a message either.
    one = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1", max_chars_per_message=10
    )
    assert one[7]["content"] == "x" * 9000 and one.report["ignored_inputs"] == {"max_chars_per_message": 10}
    # A current caller passes none of them: no key, no warning.
    assert "ignored_inputs" not in session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1"
    ).report


def test_history_window_default_is_50k_tokens_and_no_message_or_char_cap() -> None:
    """Operator ruling 2026-09-28: no message-count cap, no char cap, no per-
    message cut — one window of the most recent 50,000 tokens."""
    assert HISTORY_REPLAY_MAX_TOKENS == 50_000
    params = inspect.signature(session_chat_messages).parameters
    assert params["max_tokens"].default == 50_000
    # The retired caps are still ACCEPTED (hosts built against 0.6 pass them)
    # but carry no default that could bound anything: see the 0.6.0 test below.
    for retired in ("max_messages", "max_chars_per_message", "max_total_chars"):
        assert params[retired].kind is inspect.Parameter.KEYWORD_ONLY and params[retired].default is None

    run_store = InMemoryRunStore()
    # 60 short turns (120 messages, far past the old 40-message cap) and one
    # 30,000-char answer (past the old 8,000-char cut and 24,000-char budget):
    # all of it fits 50k tokens, so all of it replays, byte for byte.
    long_answer = "word " * 6000
    for i in range(60):
        run_store.save(
            _turn_run(
                run_id=f"run-{i:02d}",
                prompt=f"q{i}",
                answer=long_answer.strip() if i == 30 else f"a{i}",
                created_at=f"2026-01-01T{i // 60:02d}:{i % 60:02d}:00+00:00",
            )
        )
    messages = session_chat_messages(run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1")
    assert len(messages) == 120
    assert messages[61]["content"] == long_answer.strip()
    assert all("#TRUNCATION" not in m["content"] for m in messages)
    report = messages.report
    assert report["max_tokens"] == 50_000
    assert report["replayed_messages"] == 120
    assert report["dropped_messages"] == 0 and report["dropped_tokens"] == 0
    assert report["replayed_tokens"] == sum(estimate_message_tokens(m) for m in messages)
    assert report["token_estimator"] == "abstractruntime.memory.token_budget.estimate_message_tokens"


def test_history_window_exactly_50k_fits_and_one_more_token_drops_the_oldest_turn() -> None:
    """Boundary: a window whose turns total EXACTLY 50,000 tokens is replayed
    whole; one token more and the oldest turn falls out, whole, announced."""
    older = [(_text_with_tokens("user", 10_000), _text_with_tokens("assistant", 10_000))]
    newest = [(_text_with_tokens("user", 15_000), _text_with_tokens("assistant", 15_000))]
    pairs = [[{"role": "user", "content": u}, {"role": "assistant", "content": a}] for u, a in older + newest]

    kept, report = fold_history_window(pairs)
    assert report["replayed_tokens"] == 50_000
    assert len(kept) == 2 and report["dropped_messages"] == 0

    one_more = [[{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]] + pairs
    kept2, report2 = fold_history_window(one_more)
    assert kept2 == pairs  # the tiny oldest turn cannot join a full window
    assert report2["dropped_messages"] == 2 and report2["replayed_tokens"] == 50_000

    over = [[{"role": "user", "content": _text_with_tokens("user", 10_001)}, pairs[0][1]], pairs[1]]
    kept3, report3 = fold_history_window(over)
    assert kept3 == [pairs[1]]
    assert report3["dropped_messages"] == 2
    assert report3["dropped_tokens"] == 20_001 and report3["replayed_tokens"] == 30_000
    # Never a cut: every kept message is byte-identical to its source.
    assert kept3[0][0]["content"] == pairs[1][0]["content"]


def test_history_window_drops_oldest_whole_turns_and_announces_it() -> None:
    run_store = InMemoryRunStore()
    for i in range(10):
        run_store.save(
            _turn_run(
                run_id=f"run-{i}",
                prompt=f"question number {i}",
                answer=f"answer number {i}",
                created_at=f"2026-01-01T00:{i:02d}:00+00:00",
            )
        )
    newest3 = sum(_pair_tokens(f"question number {i}", f"answer number {i}") for i in (7, 8, 9))

    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1", max_tokens=newest3
    )
    assert [m["metadata"]["run_id"] for m in messages] == ["run-7", "run-7", "run-8", "run-8", "run-9", "run-9"]
    assert messages[0]["role"] == "user"
    # Loud (ADR-0026 §1): the oldest surviving message names the window.
    head = messages[0]["content"]
    assert head.startswith("[#TRUNCATION: 14 earlier message(s)")
    assert f"the most recent {newest3} tokens" in head
    assert head.endswith("question number 7")
    assert messages[0]["metadata"]["replay_truncated"] is True
    assert messages.report == {
        **messages.report,
        "max_tokens": newest3,
        "replayed_messages": 6,
        "replayed_tokens": newest3,
        "dropped_messages": 14,
        "dropped_counts_complete": True,
        "oversize_turn_kept": False,
    }
    # One token short: the oldest of the three falls out.
    tighter = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1", max_tokens=newest3 - 1
    )
    assert [m["metadata"]["run_id"] for m in tighter][::2] == ["run-8", "run-9"]


def test_history_window_keeps_an_oversize_newest_turn_whole_and_says_so() -> None:
    """A newest turn larger than the whole window is kept WHOLE (never cut —
    ADR-0026 §2/§3; dropping it would replay nothing), alone, and the report
    records the exception."""
    run_store = InMemoryRunStore()
    huge = "token " * 60_000
    for i, answer in enumerate(["a0", huge.strip()]):
        run_store.save(
            _turn_run(run_id=f"run-{i}", prompt=f"q{i}", answer=answer, created_at=f"2026-01-01T00:0{i}:00+00:00")
        )
    messages = session_chat_messages(run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1")
    assert [m["metadata"]["run_id"] for m in messages] == ["run-1", "run-1"]
    assert messages[1]["content"] == huge.strip()
    assert messages.report["oversize_turn_kept"] is True
    assert messages.report["replayed_tokens"] > 50_000
    assert messages.report["dropped_messages"] == 2
    assert "#TRUNCATION" in messages[0]["content"]


def test_history_window_reads_past_the_first_batch_and_flags_uncounted_older_turns() -> None:
    """The read is newest-first in doubling batches: a window that holds more
    than one batch keeps reading; one that fills early stops and says the
    dropped counts cover only what was read."""
    run_store = InMemoryRunStore()
    for i in range(100):
        run_store.save(
            _turn_run(run_id=f"run-{i:03d}", prompt=f"q{i}", answer=f"a{i}",
                      created_at=f"2026-01-01T{i // 60:02d}:{i % 60:02d}:00+00:00")
        )
    everything = session_chat_messages(run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1")
    assert len(everything) == 200 and everything.report["dropped_counts_complete"] is True

    small = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1",
        max_tokens=_pair_tokens("q99", "a99") + _pair_tokens("q98", "a98"),
    )
    assert [m["metadata"]["run_id"] for m in small][::2] == ["run-098", "run-099"]
    assert small.report["dropped_counts_complete"] is False
    assert "(and older turns not counted)" in small[0]["content"]


def test_history_window_rejects_a_non_positive_budget() -> None:
    for bad in (0, -1, None, True, 1.5):
        with pytest.raises(ValueError):
            session_chat_messages(run_store=InMemoryRunStore(), session_id="sess-1", max_tokens=bad)


def test_session_chat_messages_skips_scheduled_turns() -> None:
    """Audit #4: scheduled wrapper runs are not conversation."""
    run_store = InMemoryRunStore()
    scheduled = _turn_run(
        run_id="run-sched",
        prompt="scheduled prompt",
        answer="scheduled answer",
        workflow_id="scheduled:abc123",
    )
    # No context.messages: without a chat-classified sibling the scheduled
    # root would otherwise survive the reconstruction's chat preference.
    scheduled.vars = {"prompt": "scheduled prompt"}
    run_store.save(scheduled)

    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1"
    )
    assert messages == []


def test_session_chat_messages_orders_same_millisecond_turns_correctly() -> None:
    """Audit #7: sub-millisecond creation times must not reverse turn order
    (the stable sort used to keep the index's newest-first order on ties)."""
    run_store = InMemoryRunStore()
    run_store.save(
        _turn_run(
            run_id="run-first",
            prompt="q-first",
            answer="a-first",
            created_at="2026-01-01T00:00:00.001000+00:00",
        )
    )
    run_store.save(
        _turn_run(
            run_id="run-second",
            prompt="q-second",
            answer="a-second",
            created_at="2026-01-01T00:00:00.001500+00:00",
        )
    )

    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1"
    )
    assert [m["content"] for m in messages] == [
        "q-first",
        "a-first",
        "q-second",
        "a-second",
    ]


def test_session_chat_messages_includes_tenant_catalog_workflow_turns() -> None:
    """Live-proof regression (2026-07-16): gateway catalog workflow ids are
    namespaced `__catalog__v2__...` — a bare startswith('__') internal check
    classified EVERY catalog turn internal, hiding all thin-client turns
    from session views and the durable replay. Internal = dunder BOTH ends
    (`__session_memory__`), never a mere prefix."""
    run_store = InMemoryRunStore()
    run_store.save(
        _turn_run(
            run_id="run-catalog",
            prompt="remember the codename",
            answer="noted.",
            workflow_id=(
                "__catalog__v2__tenant_catalog__ZGVmYXVsdA__"
                "YWJzdHJhY3Rhc3Npc3RhbnQtb3JjaGVzdHJhdG9y@0.0.1:d5f9fdd0"
            ),
        )
    )
    run_store.save(
        _turn_run(
            run_id="session_memory_sess-1",
            prompt="x",
            answer="y",
            workflow_id="__session_memory__",
            created_at="2026-01-01T00:01:00+00:00",
        )
    )

    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1"
    )

    assert [m["metadata"]["run_id"] for m in messages] == ["run-catalog", "run-catalog"]
    assert messages[0]["content"] == "remember the codename"


def test_session_chat_messages_excludes_named_runs_and_empty_session() -> None:
    run_store = InMemoryRunStore()
    run_store.save(_turn_run(run_id="run-a", prompt="qa", answer="aa"))
    run_store.save(
        _turn_run(
            run_id="run-b",
            prompt="qb",
            answer="ab",
            created_at="2026-01-01T00:01:00+00:00",
        )
    )

    messages = session_chat_messages(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        session_id="sess-1",
        exclude_run_ids=["run-b"],
    )
    assert [m["metadata"]["run_id"] for m in messages] == ["run-a", "run-a"]

    assert (
        session_chat_messages(
            run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id=""
        )
        == []
    )


def test_announce_dropped_is_public_and_the_private_name_is_an_alias() -> None:
    """Hosts that fold history themselves (the gateway's client-sent history)
    write the same notice through the public name; the pre-0.7.0 private name
    stays importable."""
    import abstractruntime
    from abstractruntime.session_history import _announce_dropped, announce_dropped

    assert abstractruntime.announce_dropped is announce_dropped and _announce_dropped is announce_dropped
    pairs = [[{"role": "user", "content": "old " * 400}, {"role": "assistant", "content": "a"}],
             [{"role": "user", "content": "new"}, {"role": "assistant", "content": "b"}]]
    kept, report = fold_history_window(pairs, max_tokens=sum(estimate_message_tokens(m) for m in pairs[1]))
    messages = [m for pair in kept for m in pair]
    announce_dropped(messages, report)
    assert messages[0]["content"].startswith("[#TRUNCATION: 2 earlier message(s)")
    assert messages[0]["metadata"]["replay_truncated"] is True
