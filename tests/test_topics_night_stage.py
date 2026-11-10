"""The night's TOPICS stage (identity/topics.py `sleep_communities`, run by `life.build_consolidator`; operator
2026-11-10: topics run server side, during the sleep, after the repair; memory owns the graph operations).

Memory's verbs are being built by the memory owner; until they land these tests run against `_FakeCommunities`, a
fake of EXACTLY the frozen signatures (`embed_missing`, `communities_pass`), installed on `abstractmemory` for the
test only. The fake writes its version through the home's commit guard, as memory's one transaction does, so the
Stop fence is exercised for real. Each test goes red without the piece it names."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_entity_life_loop import _make_home  # noqa: E402
from test_stop_fence import _stop_here  # noqa: E402

from abstractruntime.identity.lifecycle_fence import LifecycleFence  # noqa: E402

TITLE = "Dam verification"


class _Capability:
    """The summaries capability: every authoring call answers."""

    model_info = {"provider": "fake", "model": "summarizer"}

    def __init__(self, calls: Optional[List[str]] = None) -> None:
        self.calls = calls

    def generate(self, *, prompt: str, system_prompt: str, response_model: Any) -> Any:
        if self.calls is not None:
            self.calls.append("author")
        return response_model(title=TITLE, summary="How the dam was checked, and why.")


class _FakeCommunities:
    """Memory's frozen verbs, faked. `order` is shared with the night's other stages (the test's spies)."""

    def __init__(self, order: List[str], *, stop_in_pass: Optional[Path] = None, ignore_cancelled: bool = False,
                 failures: bool = False) -> None:
        self.order = order
        self.versions: List[int] = []
        self.embed_calls: List[Dict[str, Any]] = []
        self.pass_calls: List[Dict[str, Any]] = []
        self.stop_in_pass = stop_in_pass
        self.ignore_cancelled = ignore_cancelled
        self.failures = failures

    def embed_missing(self, system, *, scopes, owner_id, record_ids, cancelled):
        self.order.append("embed_missing")
        self.embed_calls.append({"scopes": list(scopes), "owner_id": owner_id, "record_ids": record_ids,
                                 "cancelled": cancelled()})
        return {"embedded": ["ex:episode-1", "ex:episode-2"], "skipped": [],
                "failed": [{"id": "ex:episode-3", "reason": "the embedder refused the text"}]}

    def communities_pass(self, system, *, scopes, owner_id, author, cancelled, budget=None, seed_previous=True):
        from abstractmemory.commit_guard import guard_commit

        self.order.append("communities_pass")
        self.pass_calls.append({"scopes": list(scopes), "owner_id": owner_id, "budget": budget,
                                "seed_previous": seed_previous})
        authored = author("community", "Members: the dam check; the spillway test.",
                          {"pass": "communities", "record_kind": "community", "fields": ["title", "summary"]})
        assert authored["title"] == TITLE
        if self.stop_in_pass is not None:
            _stop_here(self.stop_in_pass)  # the operator's Stop while the pass computes
        receipt = {"version": None, "communities": 0, "members_placed": 0, "attached": 0, "merged": 0,
                   "unvectored": ["ex:episode-3"],
                   "authoring": {"authored": 0, "pending": 0, "failed": 0, "failures": []},
                   "cancelled_before_write": True, "timings": {"total_ms": 1.0}}
        if cancelled() and not self.ignore_cancelled:
            return receipt  # memory's one check before its transaction: nothing written
        guard_commit(system.store, "communities_pass")  # memory's one transaction: the guard before its COMMIT
        self.versions.append(len(self.versions) + 1)
        failures = [{"record_id": "ex:community-2", "record_kind": "community", "pass": "communities",
                     "fields": ["title", "summary"], "reason": "author_raised", "error_class": "TimeoutError",
                     "message": "the summaries model did not answer", "at": "2026-11-10T03:00:00+00:00"}] \
            if self.failures else []
        receipt.update({"version": self.versions[-1], "communities": 2, "members_placed": 7, "attached": 2,
                        "merged": 1, "cancelled_before_write": False,
                        "authoring": {"authored": 1, "pending": len(failures), "failed": len(failures),
                                      "failures": failures}})
        return receipt


def _install(monkeypatch, fake: _FakeCommunities, order: List[str], *, record_maintenance: bool = True,
             dream: Optional[Dict[str, Any]] = None) -> None:
    """The fake verbs on abstractmemory, a quiet `sleep_pass` that completes its night like memory's (the
    maintenance row), and a spy on the repair step."""
    import abstractmemory

    import abstractruntime.identity.authoring as authoring

    monkeypatch.setattr(abstractmemory, "embed_missing", fake.embed_missing, raising=False)
    monkeypatch.setattr(abstractmemory, "communities_pass", fake.communities_pass, raising=False)

    def sleep_pass(system, *, scopes, owner_id, should_continue=None, include_dream=True, include_identity=True,
                   author=None, cancelled=None):
        order.append("sleep_pass")
        if record_maintenance:
            system.tasks.record_maintenance(scopes, kind="night" if include_dream else "cycle",
                                            completed_at="2026-11-10T03:00:00+00:00",
                                            seq=int(system.journal.current_seq()),
                                            detail={"include_identity": bool(include_identity)})
        return {"maintenance": {}, "dream": dict(dream or {})}

    monkeypatch.setattr(abstractmemory, "sleep_pass", sleep_pass)
    real_declare = authoring.declare_missing_metadata

    def declare(*a, **k):
        order.append("repair")
        return real_declare(*a, **k)

    monkeypatch.setattr(authoring, "declare_missing_metadata", declare)


def _asleep(home_dir: Path) -> None:
    from abstractruntime.identity.life import write_entity_state

    write_entity_state(home_dir, "asleep", reason="self-elected sleep: full", mode="dreaming", written_by="self",
                       fence=LifecycleFence.admit(home_dir))


def _legacy_record(home_dir: Path) -> None:
    """A record the repair step authors (a legacy title placeholder), so the repair spends the night's budget."""
    from abstractmemory import MemoryRecordInput

    from abstractruntime.identity.chat import open_home

    home = open_home(home_dir, fence=None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            home.ms.remember_many([MemoryRecordInput(
                kind="lesson", title="untitled lesson [3fa2c1]", digest="Waiting lets the other finish.",
                attributes={"title_missing": True, "title_failure_reason": "missing"},
            )], scope="life", owner_id=home.entity_id, idempotency_key="legacy-1", turn_id="legacy-1")
    finally:
        home.close()


def _maintenance(home_dir: Path) -> Optional[Dict[str, Any]]:
    from abstractruntime.identity.chat import open_home

    home = open_home(home_dir, fence=None)
    try:
        eid = home.entity_id
        return home.ms.tasks.last_maintenance([("self", eid), ("diary", eid), ("life", eid)], kind="night")
    finally:
        home.close()


def _night(home_dir: Path, lines: List[str], **kw: Any) -> Dict[str, Any]:
    from abstractruntime.identity.life import build_consolidator

    kw.setdefault("llm_for_capability", lambda name: _Capability())
    return build_consolidator(home_dir, out=lines.append, fence=LifecycleFence.admit(home_dir), **kw)()


# ------------------------------------------------------------------------------------------------- the stage


def test_the_topics_stage_runs_after_the_repair_embeds_first_and_before_the_night_voice(tmp_path: Path,
                                                                                         monkeypatch) -> None:
    """The order (operator 2026-11-10: "redefined in sleep AFTER repair"): sleep_pass -> repair -> embed_missing ->
    communities_pass -> the night voice; the vector-less records are embedded BEFORE the pass (record_ids=None:
    every one in his scopes); the pass gets the night's own author, the fence, and what the repair left of the
    night's authoring budget."""
    import abstractruntime.identity.night_voice as night_voice
    import abstractruntime.identity.prelude as prelude

    home_dir = _make_home(tmp_path)
    _legacy_record(home_dir)
    order: List[str] = []
    fake = _FakeCommunities(order)
    _install(monkeypatch, fake, order, dream={"created": True, "dream_record_id": "ex:dream-1", "signals": ["novel"]})
    monkeypatch.setattr(prelude, "render_summon_prelude", lambda *a, **k: {"text": ""})

    def voice(*a, **k):
        order.append("night_voice")
        return {"narrated": False}

    monkeypatch.setattr(night_voice, "run_night_voice", voice)
    _asleep(home_dir)
    authored: List[str] = []
    lines: List[str] = []
    result = _night(home_dir, lines, llm_for_capability=lambda name: _Capability(authored), authoring_cap=8,
                    narrator_llm_factory=lambda: object())
    assert order == ["sleep_pass", "repair", "embed_missing", "communities_pass", "night_voice"], order
    eid = "entity:liveling@home-test"
    scopes = [("self", eid), ("diary", eid), ("life", eid)]
    assert fake.embed_calls == [{"scopes": scopes, "owner_id": eid, "record_ids": None, "cancelled": False}]
    spent = result["authoring"]["considered"]
    assert spent >= 1 and result["authoring"]["authored"] >= 1, result["authoring"]
    assert fake.pass_calls == [{"scopes": scopes, "owner_id": eid, "budget": 8 - spent, "seed_previous": True}]
    assert authored.count("author") == result["authoring"]["authored"] + 1  # the pass titled with the night's author
    assert fake.versions == [1]


def test_the_topics_receipt_its_line_and_the_maintenance_note(tmp_path: Path, monkeypatch) -> None:
    """The receipt rides the night's dict under `communities` (memory's receipt + the embedding counts + the budget);
    one own_time.log line says what formed; `communities_version` joins tonight's completed-maintenance detail (its
    completion time and seq unchanged); what could not be embedded is a night note, not a failure; a title the pass
    could not author has its authoring-failure line."""
    home_dir = _make_home(tmp_path)
    order: List[str] = []
    fake = _FakeCommunities(order, failures=True)
    _install(monkeypatch, fake, order)
    _asleep(home_dir)
    lines: List[str] = []
    result = _night(home_dir, lines)
    receipt = result["communities"]
    assert {"version", "communities", "members_placed", "attached", "merged", "unvectored", "authoring",
            "cancelled_before_write", "timings"} <= set(receipt), receipt
    assert receipt["version"] == 1 and receipt["cancelled_before_write"] is False and "cancelled" not in receipt
    assert receipt["embedding"] == {"embedded": 2, "skipped": 0, "failed": 1}
    assert receipt["budget"] == 64 - result["authoring"]["considered"]
    topic_lines = [ln for ln in lines if ln.startswith("(topics: ")]
    assert topic_lines == [
        "(topics: 1 memories could not be embedded this sleep - the embedder refused the text; the pass places "
        "them by their neighbours)",
        "(topics: version 1 - 2 topics, 7 memories placed (2 attached, 1 merged); titles 1 authored, 1 pending, "
        "1 failed; 2 embedded first, 1 still without a vector)"], lines
    failure_lines = [ln for ln in lines if ln.startswith("#AUTHORING_FAILURE") and "ex:community-2" in ln]
    assert len(failure_lines) == 1 and "memory.communities" in failure_lines[0], lines
    row = _maintenance(home_dir)
    assert row is not None
    assert row["detail"] == {"include_identity": True, "communities_version": 1}, row
    assert row["completed_at"] == "2026-11-10T03:00:00+00:00"


def test_no_maintenance_note_when_tonight_completed_no_maintenance(tmp_path: Path, monkeypatch) -> None:
    """A night whose sleep_pass ended early completed no maintenance: an older row is never re-stamped."""
    import abstractmemory

    home_dir = _make_home(tmp_path)
    order: List[str] = []
    fake = _FakeCommunities(order)
    _install(monkeypatch, fake, order, record_maintenance=False)

    def ended_early(system, **kw):
        system.tasks.record_maintenance(kw["scopes"], kind="night", completed_at="2026-11-01T03:00:00+00:00", seq=1,
                                        detail={"include_identity": True})  # last week's completed night
        return {"maintenance": {}, "dream": {}, "cancelled_after": "consolidation"}

    monkeypatch.setattr(abstractmemory, "sleep_pass", ended_early)
    _asleep(home_dir)
    result = _night(home_dir, [])
    assert result["communities"]["version"] == 1
    assert _maintenance(home_dir)["detail"] == {"include_identity": True}


# ------------------------------------------------------------------------------------------------- the Stop


def test_a_stop_before_the_topics_write_leaves_no_version(tmp_path: Path, monkeypatch) -> None:
    """The operator's Stop while the pass computes: memory's check before its one transaction sees it
    (`cancelled=fence.moved`) and writes nothing; the receipt says so; the night ends there with its one Stop line -
    no topics line, no maintenance note, no failure line."""
    home_dir = _make_home(tmp_path)
    order: List[str] = []
    fake = _FakeCommunities(order, stop_in_pass=home_dir)
    _install(monkeypatch, fake, order)
    _asleep(home_dir)
    lines: List[str] = []
    result = _night(home_dir, lines, narrator_llm_factory=lambda: (_ for _ in ()).throw(AssertionError("no voice")))
    assert fake.versions == [], "a version landed after the Stop"
    assert result["stopped_by"] == "kill_switch", result
    receipt = result["communities"]
    assert receipt["cancelled_before_write"] is True and receipt["cancelled"] == "the operator's Stop", receipt
    assert not [ln for ln in lines if ln.startswith("(topics: ")], lines
    assert lines.count("(the operator's Stop - the night ends here; nothing more is written)") == 1, lines
    assert not [ln for ln in lines if "#FALLBACK" in ln or "fail" in ln.lower()], lines
    assert "communities_version" not in _maintenance(home_dir)["detail"]


def test_the_commit_guard_refuses_a_topics_write_after_the_stop(tmp_path: Path, monkeypatch) -> None:
    """A Stop between memory's check and its COMMIT: the home's commit guard (the loop's fence) refuses the
    transaction; the stage absorbs the refusal as the Stop, never a failure."""
    home_dir = _make_home(tmp_path)
    order: List[str] = []
    fake = _FakeCommunities(order, stop_in_pass=home_dir, ignore_cancelled=True)
    _install(monkeypatch, fake, order)
    _asleep(home_dir)
    lines: List[str] = []
    result = _night(home_dir, lines)
    assert fake.versions == [], "the guard let a version land after the Stop"
    assert result["stopped_by"] == "kill_switch"
    receipt = result["communities"]
    assert receipt["cancelled_before_write"] is True and receipt["cancelled"].startswith("the operator's Stop")
    assert not [ln for ln in lines if "#FALLBACK" in ln or "fail" in ln.lower()], lines


def test_a_stop_during_the_embedding_ends_the_stage_before_the_pass(tmp_path: Path, monkeypatch) -> None:
    home_dir = _make_home(tmp_path)
    order: List[str] = []
    fake = _FakeCommunities(order)
    _install(monkeypatch, fake, order)
    real = fake.embed_missing

    def stopped_while_embedding(system, **kw):
        out = real(system, **kw)
        _stop_here(home_dir)
        return out

    import abstractmemory

    monkeypatch.setattr(abstractmemory, "embed_missing", stopped_while_embedding)
    _asleep(home_dir)
    result = _night(home_dir, [])
    assert "communities_pass" not in order and fake.versions == []
    assert result["communities"]["cancelled"] == "the operator's Stop" and result["stopped_by"] == "kill_switch"


# ------------------------------------------------------------------------------------------------- the edges


def test_a_night_ended_before_the_topics_step_skips_it(tmp_path: Path, monkeypatch) -> None:
    """A wake (or a visitor, the brake) during the pass ends the night at its next stage: no topics step."""
    import abstractmemory

    from abstractruntime.identity.life import write_entity_state

    home_dir = _make_home(tmp_path)
    order: List[str] = []
    fake = _FakeCommunities(order)
    _install(monkeypatch, fake, order)

    def woken(system, **kw):
        write_entity_state(home_dir, "awake", reason="operator wake", written_by="operator",
                           fence=LifecycleFence.admit(home_dir))
        return {"maintenance": {}, "dream": {}, "cancelled_after": "consolidation"}

    monkeypatch.setattr(abstractmemory, "sleep_pass", woken)
    _asleep(home_dir)
    result = _night(home_dir, [])
    assert result["communities"] == {"skipped": "the night ended before the topics step"}
    assert fake.embed_calls == [] and fake.pass_calls == []


def test_a_memory_without_the_verbs_is_said_loudly_and_the_night_stands(tmp_path: Path, monkeypatch) -> None:
    """No second implementation in the runtime: a memory without `embed_missing` / `communities_pass` is one loud
    #MISSING_VERB line and a receipt that names the verbs; the night's other work stands."""
    import abstractmemory

    home_dir = _make_home(tmp_path)
    order: List[str] = []
    _install(monkeypatch, _FakeCommunities(order), order)
    monkeypatch.delattr(abstractmemory, "embed_missing", raising=False)
    monkeypatch.delattr(abstractmemory, "communities_pass", raising=False)
    _asleep(home_dir)
    lines: List[str] = []
    result = _night(home_dir, lines)
    assert result["communities"] == {"missing_verb": "abstractmemory.embed_missing / communities_pass"}
    assert [ln for ln in lines if ln.startswith("#MISSING_VERB abstractmemory.embed_missing / communities_pass")]
    assert "stopped_by" not in result and order == ["sleep_pass", "repair"]


def test_a_pass_that_raises_is_said_and_the_night_stands(tmp_path: Path, monkeypatch) -> None:
    import abstractmemory

    home_dir = _make_home(tmp_path)
    order: List[str] = []
    _install(monkeypatch, _FakeCommunities(order), order)

    def broken(system, **kw):
        raise RuntimeError("the graph read failed")

    monkeypatch.setattr(abstractmemory, "communities_pass", broken, raising=False)
    _asleep(home_dir)
    lines: List[str] = []
    result = _night(home_dir, lines)
    assert result["communities"]["error"] == "RuntimeError: the graph read failed"
    assert "#FALLBACK the topics step did not run (RuntimeError: the graph read failed)" in lines
    assert "communities_version" not in _maintenance(home_dir)["detail"]


@pytest.mark.parametrize("missing", ["failures"])
def test_failed_titles_without_failure_records_are_said(tmp_path: Path, monkeypatch, missing: str) -> None:
    """The failure lines come from memory's `authoring.failures`: a failed count without them is said, never a
    silent hole."""
    import abstractmemory

    home_dir = _make_home(tmp_path)
    order: List[str] = []
    fake = _FakeCommunities(order, failures=True)
    _install(monkeypatch, fake, order)
    real = fake.communities_pass

    def countless(system, **kw):
        out = real(system, **kw)
        out["authoring"].pop(missing)
        return out

    monkeypatch.setattr(abstractmemory, "communities_pass", countless, raising=False)
    _asleep(home_dir)
    lines: List[str] = []
    _night(home_dir, lines)
    assert [ln for ln in lines if ln.startswith("#FALLBACK topics: memory counted 1 failed title(s)")], lines
