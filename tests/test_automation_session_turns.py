"""select_session_turns: the one definition of a session's turns (automations C3/E).

Turns = parent-less runs + automation occurrences; never descendants,
controllers, internal runs, legacy scheduled wrappers or drafts. History
bundles and session replay (growing-mode seeding) both read it, so an
automation's occurrences show up as conversation turns everywhere.
"""

from __future__ import annotations

import pytest

from abstractruntime import RunState, RunStatus, session_chat_messages
from abstractruntime.history_bundle import _best_effort_session_turns
from abstractruntime.session_turns import select_session_turns
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore

AUTO = "auto-1"
SID = "s-auto"


def make_store(kind, tmp_path):
    if kind == "memory":
        return InMemoryRunStore()
    if kind == "json":
        return JsonFileRunStore(tmp_path / "runs")
    return SqliteRunStore(SqliteDatabase(tmp_path / "runs.sqlite"))


def _save(store, run_id, minute, *, prompt="", answer="", parent=None, meta=None, workflow_id="wf",
          status=RunStatus.COMPLETED, lifecycle=None, session_id=SID):
    vars_obj = {"prompt": prompt, "context": {"messages": []}, "_meta": dict(meta or {})}
    if lifecycle:
        vars_obj["_run_lifecycle"] = lifecycle
    ts = f"2026-09-27T10:{minute:02d}:00+00:00"
    store.save(RunState(run_id=run_id, workflow_id=workflow_id, status=status, current_node="n", vars=vars_obj,
                        output={"answer": answer} if answer else {}, session_id=session_id, parent_run_id=parent,
                        created_at=ts, updated_at=ts))


def _occ(index, *, attempt=1, role="occurrence"):
    return {"occurrence": {"automation_id": AUTO, "occurrence_index": index, "attempt": attempt,
                           "role": role, "session_kind": "automation"}}


def populate(store):
    _save(store, AUTO, 0, meta={"automation": {"title": "t"}})                         # controller: never a turn
    _save(store, "chat-1", 1, prompt="hello", answer="hi")                             # a chat turn
    _save(store, "occ-1", 2, prompt="tick 1", answer="mem 41%", parent=AUTO, meta=_occ(1))
    _save(store, "occ-1-desc", 3, prompt="sub", answer="x", parent="occ-1", meta=_occ(1, role="descendant"))
    _save(store, "occ-2a1", 4, prompt="tick 2", parent=AUTO, meta=_occ(2), status=RunStatus.FAILED)
    _save(store, "occ-2a2", 5, prompt="tick 2", answer="mem 43%", parent=AUTO, meta=_occ(2, attempt=2))
    _save(store, "chat-2", 6, prompt="why up?", answer="because")
    _save(store, "occ-3", 7, prompt="tick 3", answer="mem 40%", parent=AUTO, meta=_occ(3))
    _save(store, "legacy", 8, prompt="old", answer="old", workflow_id="scheduled:x", meta={"schedule": {"kind": "scheduled_run"}})
    _save(store, "internal", 9, prompt="m", answer="m", workflow_id="__session_memory__")
    _save(store, "draft", 10, prompt="d", answer="d", lifecycle={"purpose": "draft_test"})


def ids(runs):
    return [r.run_id for r in runs]


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite"])
def test_turns_are_roots_plus_occurrences(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    populate(store)
    assert ids(select_session_turns(store, SID)) == ["chat-1", "occ-1", "occ-2a2", "chat-2", "occ-3"]
    assert ids(select_session_turns(store, SID, include_occurrences=False)) == ["chat-1", "chat-2"]
    assert ids(select_session_turns(store, SID, include_drafts=True))[-1] == "draft"
    assert ids(select_session_turns(store, SID, through_occurrence=2)) == ["chat-1", "occ-1", "occ-2a2"]
    assert ids(select_session_turns(store, SID, automation_id="other")) == ["chat-1", "chat-2"]
    assert ids(select_session_turns(store, SID, limit=2)) == ["chat-2", "occ-3"]
    until = 1790503560000  # 2026-09-27T10:06:00Z
    assert ids(select_session_turns(store, SID, until_ms=until)) == ["chat-1", "occ-1", "occ-2a2", "chat-2"]


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite"])
def test_history_bundle_turns_include_occurrences(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    populate(store)
    turns = _best_effort_session_turns(run_store=store, ledger_store=InMemoryLedgerStore(), artifact_store=None,
                                       session_id=SID, limit=50, include_stats=False, include_artifacts=False)
    assert [t["run_id"] for t in turns] == ["chat-1", "occ-1", "occ-2a2", "chat-2", "occ-3"]
    occ = next(t for t in turns if t["run_id"] == "occ-2a2")
    assert (occ["kind"], occ["automation_id"], occ["occurrence_index"]) == ("occurrence", AUTO, 2)


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite"])
def test_growing_mode_seed_replays_prior_occurrences(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    populate(store)
    messages = session_chat_messages(run_store=store, ledger_store=InMemoryLedgerStore(), session_id=SID)
    assert [m["content"] for m in messages] == [
        "hello", "hi", "tick 1", "mem 41%", "tick 2", "mem 43%", "why up?", "because", "tick 3", "mem 40%",
    ]


def test_legacy_scheduled_prefix_alone_is_not_a_turn(tmp_path) -> None:
    store = make_store("sqlite", tmp_path)
    _save(store, "chat", 1, prompt="p", answer="a")
    _save(store, "wrapper", 2, prompt="p", answer="a", workflow_id="scheduled:abc")  # no _meta.schedule
    assert ids(select_session_turns(store, SID)) == ["chat"]


# --------------------------------------------------------------------------
# Strict history (contract C3 amendment): automation preparation and
# discussion seeding fail instead of starting without context.
# --------------------------------------------------------------------------

from abstractruntime import SessionHistoryError  # noqa: E402
from abstractruntime.storage.artifacts import InMemoryArtifactStore, artifact_ref  # noqa: E402

SEED = [
    {"role": "user", "content": "tick 1", "metadata": {"run_id": "occ-1"}},
    {"role": "assistant", "content": "mem 41%", "metadata": {"run_id": "occ-1"}},
]


def _discussion(store, *, seed=SEED, root_has_seed=True):
    disc = {"automation_id": AUTO, "occurrence_index": 1, "discussion_root_run_id": "disc-root"}
    root_meta = {"discussion": {**disc, **({"seed_messages": seed} if root_has_seed else {})}}
    _save(store, "disc-root", 20, prompt="why 41?", answer="because", meta=root_meta, session_id="s-disc")
    _save(store, "disc-2", 21, prompt="and now?", answer="stable", meta={"discussion": dict(disc)}, session_id="s-disc")


def _contents(messages):
    return [m["content"] for m in messages]


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite"])
def test_discussion_history_prepends_the_seed(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    _discussion(store)
    messages = session_chat_messages(run_store=store, session_id="s-disc", strict=True)
    assert _contents(messages) == ["tick 1", "mem 41%", "why 41?", "because", "and now?", "stable"]
    assert messages[0]["metadata"]["discussion_seed"] is True
    # Under a tight budget the seed is dropped first (it is the oldest history).
    tight = session_chat_messages(run_store=store, session_id="s-disc", max_messages=4)
    assert _contents(tight)[1:] == ["because", "and now?", "stable"]
    assert tight[0]["content"].endswith("why 41?") and tight[0]["metadata"]["dropped_turns"] == 1


def test_missing_seed_raises_when_strict_only(tmp_path) -> None:
    store = make_store("sqlite", tmp_path)
    _discussion(store, root_has_seed=False)
    with pytest.raises(SessionHistoryError) as info:
        session_chat_messages(run_store=store, session_id="s-disc", strict=True)
    assert info.value.reason_code == "history_unavailable"
    assert _contents(session_chat_messages(run_store=store, session_id="s-disc")) == [
        "why 41?", "because", "and now?", "stable"]


def test_offloaded_seed_is_resolved_or_strictly_refused(tmp_path) -> None:
    store = make_store("sqlite", tmp_path)
    arts = InMemoryArtifactStore()
    import json as _json

    meta = arts.store(_json.dumps(SEED).encode("utf-8"), content_type="application/json", run_id="disc-root")
    _discussion(store, seed=artifact_ref(meta.artifact_id))
    resolved = session_chat_messages(run_store=store, artifact_store=arts, session_id="s-disc", strict=True)
    assert _contents(resolved)[:2] == ["tick 1", "mem 41%"]
    with pytest.raises(SessionHistoryError, match="cannot be resolved"):
        session_chat_messages(run_store=store, artifact_store=None, session_id="s-disc", strict=True)


def test_strict_refuses_a_store_without_a_run_index(tmp_path) -> None:
    class ScanOnly:
        def __init__(self, inner):
            self._inner = inner

        def list_runs(self, **kw):
            return self._inner.list_runs(**kw)

        def load(self, run_id):
            return self._inner.load(run_id)

    inner = make_store("memory", tmp_path)
    populate(inner)
    with pytest.raises(SessionHistoryError, match="no run index"):
        session_chat_messages(run_store=ScanOnly(inner), session_id=SID, strict=True)
    assert session_chat_messages(run_store=ScanOnly(inner), session_id=SID)  # best-effort path unchanged


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_strict_and_bounded_reads_of_an_automation_session(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    populate(store)
    through = session_chat_messages(run_store=store, session_id=SID, through_occurrence=2, strict=True)
    assert _contents(through) == ["hello", "hi", "tick 1", "mem 41%", "tick 2", "mem 43%"]
    other = session_chat_messages(run_store=store, session_id=SID, automation_id="other", strict=True)
    assert _contents(other) == ["hello", "hi", "why up?", "because"]
    # An ordinary chat reads the same strict or not.
    _save(store, "c1", 30, prompt="q", answer="a", session_id="plain")
    assert session_chat_messages(run_store=store, session_id="plain", strict=True) == session_chat_messages(
        run_store=store, session_id="plain")


# --------------------------------------------------------------------------
# Review 44 F1: `through_occurrence` is resolved from the index, never inside
# a newest-first window. Review 44 F2: the discussion root is validated.
# --------------------------------------------------------------------------

from abstractruntime.core.run_attribution import SessionAttributionError, session_attribution  # noqa: E402
from abstractruntime.session_history import discussion_seed_messages  # noqa: E402
from abstractruntime.session_turns import OccurrenceNotInSession  # noqa: E402


def _ts(i: int) -> str:
    return f"2026-09-{1 + i // 1440:02d}T{(i // 60) % 24:02d}:{i % 60:02d}:00+00:00"


def _save_at(store, run_id, ts, *, prompt, answer, parent=None, meta=None, session_id=SID):
    store.save(RunState(run_id=run_id, workflow_id="wf", status=RunStatus.COMPLETED, current_node="n",
                        vars={"prompt": prompt, "context": {"messages": []}, "_meta": dict(meta or {})},
                        output={"answer": answer}, session_id=session_id, parent_run_id=parent,
                        created_at=ts, updated_at=ts))


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_through_an_old_occurrence_of_a_long_session(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    _save_at(store, "chat-0", _ts(0), prompt="setup", answer="ok")
    for i in range(1, 1201):  # 1,200 occurrences: far beyond the 1,000-row window
        _save_at(store, f"occ-{i}", _ts(i), prompt=f"tick {i}", answer=f"mem {i}", parent=AUTO, meta=_occ(i))
    _save_at(store, "late-chat", _ts(1300), prompt="LATE", answer="late")

    turns = select_session_turns(store, SID, through_occurrence=3, automation_id=AUTO)
    assert [t.run_id for t in turns] == ["chat-0", "occ-1", "occ-2", "occ-3"]
    messages = session_chat_messages(run_store=store, session_id=SID, automation_id=AUTO, through_occurrence=3, strict=True)
    assert _contents(messages) == ["setup", "ok", "tick 1", "mem 1", "tick 2", "mem 2", "tick 3", "mem 3"]

    with pytest.raises(OccurrenceNotInSession):
        select_session_turns(store, SID, through_occurrence=5000)
    with pytest.raises(SessionHistoryError):
        session_chat_messages(run_store=store, session_id=SID, through_occurrence=5000, strict=True)
    assert session_chat_messages(run_store=store, session_id=SID, through_occurrence=5000) == []


def _plant_foreign(store, ws_meta):
    # Another automation's discussion, in ANOTHER session.
    foreign = {"automation_id": "auto-B", "occurrence_index": 1, "discussion_root_run_id": "foreign-root",
               "seed_messages": [{"role": "user", "content": "FOREIGN-u"}, {"role": "assistant", "content": "FOREIGN-a"}]}
    _save(store, "foreign-root", 25, prompt="f", answer="f", meta={"discussion": foreign}, session_id="s-other")
    # A rogue member of s-disc pointing at it.
    rogue = {k: v for k, v in foreign.items() if k != "seed_messages"}
    _save(store, "rogue", 26, prompt="r", answer="r", meta={"discussion": rogue}, session_id="s-disc")


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite"])
def test_a_planted_foreign_root_is_never_followed(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    _discussion(store)
    _plant_foreign(store, None)
    with pytest.raises(SessionHistoryError, match="disagree"):
        session_chat_messages(run_store=store, session_id="s-disc", strict=True)
    with pytest.raises(SessionAttributionError, match="disagree"):
        session_attribution(store, "s-disc")
    seed = discussion_seed_messages(run_store=store, session_id="s-disc")
    assert seed == []  # non-strict: no seed rather than a foreign one


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_a_root_outside_the_session_is_refused(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    disc = {"automation_id": AUTO, "occurrence_index": 1, "discussion_root_run_id": "elsewhere-root"}
    _save(store, "elsewhere-root", 20, prompt="x", answer="x",
          meta={"discussion": {**disc, "seed_messages": SEED}}, session_id="s-else")
    _save(store, "member", 21, prompt="q", answer="a", meta={"discussion": dict(disc)}, session_id="s-disc")
    with pytest.raises(SessionHistoryError, match="not a run of this session"):
        session_chat_messages(run_store=store, session_id="s-disc", strict=True)


# --------------------------------------------------------------------------
# Independent mode: each occurrence has its own session; the history through
# occurrence N is gathered across sessions by the index.
# --------------------------------------------------------------------------

def _occ_independent(i):
    return {"occurrence": {"automation_id": AUTO, "occurrence_index": i, "attempt": 1,
                           "role": "occurrence", "session_kind": "occurrence"}}


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite"])
def test_independent_automation_history_through_n_spans_sessions(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    _save(store, AUTO, 0, meta={"automation": {"title": "t"}}, session_id=f"automation:{AUTO}")
    for i in range(1, 5):
        _save(store, f"occ-{i}", i, prompt=f"tick {i}", answer=f"mem {i}", parent=AUTO,
              meta=_occ_independent(i), session_id=f"occ-session-{i}")
    _save(store, "other-auto-occ", 3, prompt="x", answer="x", parent="other",
          meta={"occurrence": {"automation_id": "other", "occurrence_index": 1, "role": "occurrence",
                               "session_kind": "occurrence"}}, session_id="occ-session-3")

    # A discussion of occurrence 3 seeds from occurrence 3's own session.
    turns = select_session_turns(store, "occ-session-3", automation_id=AUTO, through_occurrence=3)
    assert [t.run_id for t in turns] == ["occ-1", "occ-2", "occ-3"]
    seed = session_chat_messages(run_store=store, session_id="occ-session-3", automation_id=AUTO,
                                 through_occurrence=3, strict=True)
    assert _contents(seed) == ["tick 1", "mem 1", "tick 2", "mem 2", "tick 3", "mem 3"]
    with pytest.raises(SessionHistoryError):
        session_chat_messages(run_store=store, session_id="occ-session-3", automation_id=AUTO,
                              through_occurrence=9, strict=True)
    # From the automation's own session (which holds no occurrence) too.
    from_automation = select_session_turns(store, f"automation:{AUTO}", automation_id=AUTO, through_occurrence=2)
    assert [t.run_id for t in from_automation] == ["occ-1", "occ-2"]
    # Without an automation, a session is still only its own turns.
    assert [t.run_id for t in select_session_turns(store, "occ-session-3")] == ["occ-3", "other-auto-occ"]
