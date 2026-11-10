"""The night's TOPICS stage (identity/topics.py `sleep_communities`, run by `life.build_consolidator`; operator
2026-11-10: topics run server side, during the sleep, after the repair; memory owns the graph operations).

These tests drive memory's REAL verbs (`abstractmemory.communities.embed_missing` / `communities_pass`, AbstractMemory
3082695) on a scratch home: the night is opened with an embedding model whose client is a deterministic hash embedder
(no network), so `embed_missing` really embeds the vector-less records before the pass and the pass really forms,
authors and writes its version through the home's commit guard. Only `sleep_pass` (not the subject) is a quiet
stand-in that completes its night like memory's (the memq_maintenance row). A scripted receipt is used only for the
version-skew case. Each test goes red without the piece it names."""

from __future__ import annotations

import hashlib
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_entity_life_loop import _make_home  # noqa: E402
from test_stop_fence import _stop_here  # noqa: E402

from abstractruntime.identity.lifecycle_fence import LifecycleFence  # noqa: E402

pytest.importorskip("numpy")  # memory's pass needs abstractmemory[fast]

EID = "entity:liveling@home-test"
SCOPES = [("self", EID), ("diary", EID), ("life", EID)]

RECORDS = [  # two subjects, enough members each to stand as topics
    ("dam-1", "episode", "Checking the dam spillway", "We checked the spillway gauges of the river dam together."),
    ("dam-2", "episode", "The dam inspection report", "The inspection report of the river dam spillway came back."),
    ("dam-3", "lesson", "Spillway gauges drift", "The river dam spillway gauges drift when the winter sets in."),
    ("dam-4", "episode", "Recalibrating the gauges", "We recalibrated the river dam spillway gauges this morning."),
    ("dam-5", "episode", "Dam spillway flood test", "The river dam spillway flood test passed its gauges check."),
    ("dam-6", "episode", "The dam keeper's notes", "The keeper's notes on the river dam spillway gauges."),
    ("gar-1", "episode", "Planting the garden sensors", "Planted moisture sensors along the vegetable garden beds."),
    ("gar-2", "episode", "Watering schedule", "The vegetable garden moisture sensors drive the watering."),
    ("gar-3", "lesson", "Sensors need shade", "Garden moisture sensors read wrong in full vegetable sun."),
    ("gar-4", "episode", "Garden harvest", "The vegetable garden harvest followed the moisture sensors."),
    ("gar-5", "episode", "Replacing a sensor", "Replaced one garden moisture sensor in the vegetable beds."),
    ("gar-6", "episode", "Garden beds mulched", "Mulched the vegetable garden beds around the moisture sensors."),
]


class _HashEmbedder:
    """Stands in for OpenAICompatTextEmbedder (same constructor): bag of hashed words, no network."""

    dim = 256

    def __init__(self, *, base_url: str = "", model: str = "test-hash") -> None:
        self.model = model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in str(text).lower().split():
                tok = tok.strip(".,;:!?()[]\"'")
                if len(tok) > 3:
                    vec[int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "big") % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class _Capability:
    """The summaries capability: distinct authored words per call; `on_community(n)` runs on each community call."""

    model_info = {"provider": "fake", "model": "summarizer"}

    def __init__(self, *, raise_on_community: bool = False, on_community: Any = None) -> None:
        self.calls: List[str] = []
        self.raise_on_community = raise_on_community
        self.on_community = on_community

    def generate(self, *, prompt: str, system_prompt: str, response_model: Any) -> Any:
        n = len(self.calls)
        community = "community" in prompt.lower()
        self.calls.append("community" if community else "record")
        if community and self.on_community is not None:
            self.on_community(sum(1 for c in self.calls if c == "community"))
        if community and self.raise_on_community:
            raise TimeoutError("the summaries model did not answer")
        return response_model(title=f"Authored title {n}", summary=f"An authored summary, number {n}.")


def _home(tmp_path: Path, *, legacy: bool = False) -> Path:
    """A home whose records were formed WITHOUT vectors (no embedder at formation): the night embeds them."""
    from abstractmemory import MemoryRecordInput

    from abstractruntime.identity.chat import open_home

    home_dir = _make_home(tmp_path)
    home = open_home(home_dir, fence=None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for key, kind, title, digest in RECORDS:
                home.ms.remember_many([MemoryRecordInput(kind=kind, title=title, digest=digest)], scope="life",
                                      owner_id=home.entity_id, idempotency_key=key, turn_id=key)
            if legacy:  # a record the repair step authors, so the repair spends some of the night's budget
                home.ms.remember_many([MemoryRecordInput(
                    kind="lesson", title="untitled lesson [3fa2c1]", digest="Waiting lets the other finish.",
                    attributes={"title_missing": True, "title_failure_reason": "missing"},
                )], scope="life", owner_id=home.entity_id, idempotency_key="legacy-1", turn_id="legacy-1")
    finally:
        home.close()
    return home_dir


def _install(monkeypatch, order: List[str], *, record_maintenance: bool = True,
             dream: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Spies on memory's real verbs (they run), the hash embedder as the night's embedding client, a quiet
    `sleep_pass` that completes its night like memory's, and a spy on the repair step."""
    import abstractmemory
    import abstractmemory.communities as mc

    import abstractruntime.identity.authoring as authoring

    calls: Dict[str, List[Dict[str, Any]]] = {"embed_missing": [], "communities_pass": []}
    real_embed, real_pass = mc.embed_missing, mc.communities_pass

    def embed_missing(system, **kw):
        order.append("embed_missing")
        calls["embed_missing"].append({k: v for k, v in kw.items() if k != "cancelled"})
        return real_embed(system, **kw)

    def communities_pass(system, **kw):
        order.append("communities_pass")
        calls["communities_pass"].append({k: v for k, v in kw.items() if k not in ("author", "cancelled")})
        return real_pass(system, **kw)

    monkeypatch.setattr(mc, "embed_missing", embed_missing)
    monkeypatch.setattr(mc, "communities_pass", communities_pass)
    monkeypatch.setattr(abstractmemory, "OpenAICompatTextEmbedder", _HashEmbedder)

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
    return calls


def _asleep(home_dir: Path) -> None:
    from abstractruntime.identity.life import write_entity_state

    write_entity_state(home_dir, "asleep", reason="self-elected sleep: full", mode="dreaming", written_by="self",
                       fence=LifecycleFence.admit(home_dir))


def _night(home_dir: Path, lines: List[str], capability: Optional[_Capability] = None, **kw: Any) -> Dict[str, Any]:
    from abstractruntime.identity.life import build_consolidator

    cap = capability or _Capability()
    return build_consolidator(home_dir, out=lines.append, fence=LifecycleFence.admit(home_dir),
                              embedding_model="test-hash", llm_for_capability=lambda name: cap, **kw)()


def _read(home_dir: Path) -> Dict[str, Any]:
    """What the home holds after the night: tonight's maintenance row and the current community records."""
    from abstractmemory.communities import community_records

    from abstractruntime.identity.chat import open_home

    home = open_home(home_dir, fence=None)
    try:
        return {"maintenance": home.ms.tasks.last_maintenance(SCOPES, kind="night"),
                "communities": community_records(home.ms, scopes=SCOPES, owner_id=EID)}
    finally:
        home.close()


def _topic_lines(lines: List[str]) -> List[str]:
    return [ln for ln in lines if ln.startswith("(topics: ")]


# ------------------------------------------------------------------------------------------------- the stage


def test_the_topics_stage_runs_after_the_repair_embeds_first_and_before_the_night_voice(tmp_path: Path,
                                                                                         monkeypatch) -> None:
    """The order (operator 2026-11-10: "redefined in sleep AFTER repair"): sleep_pass -> repair -> embed_missing ->
    communities_pass -> the night voice. The vector-less records are embedded BEFORE the pass (record_ids=None:
    every one in his scopes), so the pass sees no record without a vector; the pass gets his scopes and owner and
    what the repair left of the night's authoring budget; its titles are authored by the night's own author."""
    import abstractruntime.identity.night_voice as night_voice
    import abstractruntime.identity.prelude as prelude

    home_dir = _home(tmp_path, legacy=True)
    order: List[str] = []
    calls = _install(monkeypatch, order,
                     dream={"created": True, "dream_record_id": "ex:dream-1", "signals": ["novel"]})
    monkeypatch.setattr(prelude, "render_summon_prelude", lambda *a, **k: {"text": ""})

    def voice(*a, **k):
        order.append("night_voice")
        return {"narrated": False}

    monkeypatch.setattr(night_voice, "run_night_voice", voice)
    _asleep(home_dir)
    cap = _Capability()
    lines: List[str] = []
    result = _night(home_dir, lines, cap, authoring_cap=40, narrator_llm_factory=lambda: object())
    assert order == ["sleep_pass", "repair", "embed_missing", "communities_pass", "night_voice"], order
    assert calls["embed_missing"] == [{"scopes": SCOPES, "owner_id": EID, "record_ids": None}]
    spent = result["authoring"]["considered"]
    assert spent >= 1 and result["authoring"]["authored"] >= 1, result["authoring"]
    assert calls["communities_pass"] == [{"scopes": SCOPES, "owner_id": EID, "budget": 40 - spent}]
    receipt = result["communities"]
    # (the repair re-embeds what it re-authors; embed_missing embeds the rest - together, every record)
    assert receipt["embedding"]["embedded"] >= 1 and receipt["embedding"]["failed"] == 0, receipt
    assert receipt["unvectored_count"] == 0, "the pass ran before the embedding"
    assert receipt["version_seq"] is not None and receipt["communities"] >= 1, receipt
    assert receipt["authored"] >= 1 and cap.calls.count("community") == receipt["authored"], (receipt, cap.calls)
    assert len(_read(home_dir)["communities"]) == receipt["communities"]


def test_the_topics_receipt_its_line_and_the_maintenance_note(tmp_path: Path, monkeypatch) -> None:
    """The receipt rides the night's dict under `communities` (memory's receipt + embedding counts + budget); one
    own_time.log line says what formed; `communities_version` (the version's seq) joins tonight's
    completed-maintenance detail, its completion time unchanged."""
    home_dir = _home(tmp_path)
    order: List[str] = []
    _install(monkeypatch, order)
    _asleep(home_dir)
    lines: List[str] = []
    result = _night(home_dir, lines)
    receipt = result["communities"]
    v = receipt["version_seq"]
    assert v is not None and receipt["cancelled"] is None and "stopped" not in receipt, receipt
    assert receipt["budget"] == 64 - result["authoring"]["considered"]
    assert set(receipt["embedding"]) == {"embedded", "skipped", "failed"}
    (line,) = _topic_lines(lines)
    assert line.startswith(f"(topics: version {v} - {receipt['communities']} topics: {receipt['new']} new, "), line
    assert f"titles {receipt['authored']} authored" in line and "0 without a vector, 0 unplaced)" in line, line
    row = _read(home_dir)["maintenance"]
    assert row["detail"] == {"include_identity": True, "communities_version": v}, row
    assert row["completed_at"] == "2026-11-10T03:00:00+00:00"


def test_a_title_the_pass_could_not_author_has_its_failure_line(tmp_path: Path, monkeypatch) -> None:
    """The summaries model fails for the topics: memory forms them pending (a task each) and the night writes one
    authoring-failure line per topic from the receipt's `authoring.failures`."""
    home_dir = _home(tmp_path)
    _install(monkeypatch, [])
    _asleep(home_dir)
    lines: List[str] = []
    result = _night(home_dir, lines, _Capability(raise_on_community=True))
    receipt = result["communities"]
    assert receipt["failed"] == receipt["communities"] >= 1 and receipt["pending"] == receipt["communities"], receipt
    failure_lines = [ln for ln in lines if ln.startswith("#AUTHORING_FAILURE") and '"memory.communities_pass"' in ln]
    assert len(failure_lines) == receipt["failed"], lines
    assert all("TimeoutError" in ln for ln in failure_lines)


def test_an_unchanged_night_writes_no_version_and_no_note(tmp_path: Path, monkeypatch) -> None:
    """The second sleep over the same memories: memory writes nothing (`skipped: unchanged`), the line says so, and
    that night's maintenance row carries no version (none formed)."""
    home_dir = _home(tmp_path)
    _install(monkeypatch, [])
    _asleep(home_dir)
    first = _night(home_dir, [])["communities"]
    assert first["version_seq"] is not None
    _asleep(home_dir)
    lines: List[str] = []
    second = _night(home_dir, lines)["communities"]
    assert second["version_seq"] is None and second["skipped"].startswith("unchanged"), second
    (line,) = _topic_lines(lines)
    assert line.startswith("(topics: no new version - unchanged"), line
    assert _read(home_dir)["maintenance"]["detail"] == {"include_identity": True}


def test_no_maintenance_note_when_tonight_completed_no_maintenance(tmp_path: Path, monkeypatch) -> None:
    """A night whose sleep_pass ended early completed no maintenance: an older row is never re-stamped."""
    import abstractmemory

    home_dir = _home(tmp_path)
    _install(monkeypatch, [], record_maintenance=False)

    def ended_early(system, **kw):
        system.tasks.record_maintenance(kw["scopes"], kind="night", completed_at="2026-11-01T03:00:00+00:00", seq=1,
                                        detail={"include_identity": True})  # last week's completed night
        return {"maintenance": {}, "dream": {}, "cancelled_after": "consolidation"}

    monkeypatch.setattr(abstractmemory, "sleep_pass", ended_early)
    _asleep(home_dir)
    result = _night(home_dir, [])
    assert result["communities"]["version_seq"] is not None
    assert _read(home_dir)["maintenance"]["detail"] == {"include_identity": True}


# ------------------------------------------------------------------------------------------------- the Stop


def _clean(lines: List[str]) -> None:
    assert not _topic_lines(lines), lines
    assert lines.count("(the operator's Stop - the night ends here; nothing more is written)") == 1, lines
    assert not [ln for ln in lines if "#FALLBACK" in ln or "fail" in ln.lower()], lines


def test_a_stop_while_the_pass_authors_leaves_no_version_and_no_late_model_call(tmp_path: Path,
                                                                                monkeypatch) -> None:
    """The operator's Stop during the pass's first title: the stage's fenced author refuses every later call (no
    model call after the Stop), memory's check before its one transaction sees the Stop and writes nothing, the
    receipt says so, and the night ends with its one Stop line - no topics line, no note, no failure line."""
    home_dir = _home(tmp_path)
    _install(monkeypatch, [])
    _asleep(home_dir)

    def stop_on_first(n: int) -> None:
        if n == 1:
            _stop_here(home_dir)

    cap = _Capability(on_community=stop_on_first)
    lines: List[str] = []
    result = _night(home_dir, lines, cap, narrator_llm_factory=lambda: (_ for _ in ()).throw(AssertionError("voice")))
    assert cap.calls.count("community") == 1, f"a model call ran after the Stop: {cap.calls}"
    assert result["stopped_by"] == "kill_switch", result
    receipt = result["communities"]
    assert receipt["stopped"] == "the operator's Stop" and receipt["cancelled"] and receipt["version_seq"] is None
    _clean(lines)
    held = _read(home_dir)
    assert held["communities"] == [], "a community record landed after the Stop"
    assert "communities_version" not in held["maintenance"]["detail"]


def test_the_commit_guard_refuses_the_topics_transaction_after_the_stop(tmp_path: Path, monkeypatch) -> None:
    """A Stop between memory's check and its COMMIT: the home's commit guard (the loop's fence) refuses the one
    transaction; nothing lands; the stage takes the refusal as the Stop, never a failure."""
    import abstractmemory.communities as mc

    home_dir = _home(tmp_path)
    _install(monkeypatch, [])
    real_write = mc._write_version

    def stopped_at_the_write(*a, **k):
        _stop_here(home_dir)
        return real_write(*a, **k)

    monkeypatch.setattr(mc, "_write_version", stopped_at_the_write)
    _asleep(home_dir)
    lines: List[str] = []
    result = _night(home_dir, lines)
    assert result["stopped_by"] == "kill_switch"
    receipt = result["communities"]
    assert receipt["stopped"].startswith("the operator's Stop (") and receipt["version_seq"] is None, receipt
    _clean(lines)
    assert _read(home_dir)["communities"] == []


def test_a_stop_during_the_embedding_ends_the_stage_before_the_pass(tmp_path: Path, monkeypatch) -> None:
    import abstractmemory.communities as mc

    home_dir = _home(tmp_path)
    order: List[str] = []
    _install(monkeypatch, order)
    spied = mc.embed_missing

    def stopped_while_embedding(system, **kw):
        _stop_here(home_dir)
        return spied(system, **kw)

    monkeypatch.setattr(mc, "embed_missing", stopped_while_embedding)
    _asleep(home_dir)
    lines: List[str] = []
    result = _night(home_dir, lines)
    assert "communities_pass" not in order
    assert result["communities"]["stopped"] == "the operator's Stop" and result["stopped_by"] == "kill_switch"
    _clean(lines)


# ------------------------------------------------------------------------------------------------- the edges


def test_a_night_ended_before_the_topics_step_skips_it(tmp_path: Path, monkeypatch) -> None:
    """A wake (or a visitor, the brake) during the pass ends the night at its next stage: no topics step."""
    import abstractmemory

    from abstractruntime.identity.life import write_entity_state

    home_dir = _home(tmp_path)
    order: List[str] = []
    calls = _install(monkeypatch, order)

    def woken(system, **kw):
        write_entity_state(home_dir, "awake", reason="operator wake", written_by="operator",
                           fence=LifecycleFence.admit(home_dir))
        return {"maintenance": {}, "dream": {}, "cancelled_after": "consolidation"}

    monkeypatch.setattr(abstractmemory, "sleep_pass", woken)
    _asleep(home_dir)
    result = _night(home_dir, [])
    assert result["communities"] == {"skipped": "the night ended before the topics step"}
    assert calls == {"embed_missing": [], "communities_pass": []}


def test_a_memory_without_the_verbs_is_said_loudly_and_the_night_stands(tmp_path: Path, monkeypatch) -> None:
    """No second implementation in the runtime: a memory without `embed_missing` / `communities_pass` is one loud
    #MISSING_VERB line and a receipt that names the verbs; the night's other work stands."""
    import abstractmemory.communities as mc

    home_dir = _home(tmp_path)
    order: List[str] = []
    _install(monkeypatch, order)
    monkeypatch.delattr(mc, "communities_pass")
    _asleep(home_dir)
    lines: List[str] = []
    result = _night(home_dir, lines)
    assert result["communities"] == {"missing_verb": "abstractmemory.embed_missing / communities_pass"}
    assert [ln for ln in lines if ln.startswith("#MISSING_VERB abstractmemory.embed_missing / communities_pass")]
    assert "stopped_by" not in result and order == ["sleep_pass", "repair"]


def test_a_pass_that_raises_is_said_and_the_night_stands(tmp_path: Path, monkeypatch) -> None:
    import abstractmemory.communities as mc

    home_dir = _home(tmp_path)
    _install(monkeypatch, [])

    def broken(system, **kw):
        raise RuntimeError("the graph read failed")

    monkeypatch.setattr(mc, "communities_pass", broken)
    _asleep(home_dir)
    lines: List[str] = []
    result = _night(home_dir, lines)
    assert result["communities"]["error"] == "RuntimeError: the graph read failed"
    assert "#FALLBACK the topics step did not run (RuntimeError: the graph read failed)" in lines
    assert "communities_version" not in _read(home_dir)["maintenance"]["detail"]


def test_a_receipt_the_runtime_cannot_read_is_said_and_writes_nothing_more(tmp_path: Path, monkeypatch) -> None:
    """Version skew between the runtime and memory (a receipt without a field this runtime reads): one loud line
    naming the receipt's keys, no maintenance note, no partial line - and the night's other work stands (the stage
    never raises). The one scripted receipt here: memory's real one, a field renamed."""
    import abstractmemory.communities as mc

    home_dir = _home(tmp_path)
    _install(monkeypatch, [])
    spied = mc.communities_pass

    def skewed(system, **kw):
        out = spied(system, **kw)
        out["version"] = out.pop("version_seq")
        return out

    monkeypatch.setattr(mc, "communities_pass", skewed)
    _asleep(home_dir)
    lines: List[str] = []
    result = _night(home_dir, lines)
    receipt = result["communities"]
    assert receipt["error"] == "the receipt could not be read (KeyError: 'version_seq')", receipt
    loud = [ln for ln in lines if ln.startswith("#FALLBACK the topics step ran but its receipt could not be read")]
    assert len(loud) == 1 and "'version'" in loud[0], lines
    assert not _topic_lines(lines), lines
    assert "communities_version" not in _read(home_dir)["maintenance"]["detail"]
    assert "stopped_by" not in result


def test_what_could_not_be_embedded_is_a_night_note(tmp_path: Path, monkeypatch) -> None:
    """A night with no embedding model: memory skips every vector-less record (`no_embedder`); the night says so in
    one note beside its line - never a failure line."""
    import abstractmemory

    home_dir = _home(tmp_path)
    _install(monkeypatch, [])
    _asleep(home_dir)
    lines: List[str] = []
    from abstractruntime.identity.life import build_consolidator

    cap = _Capability()
    result = build_consolidator(home_dir, out=lines.append, fence=LifecycleFence.admit(home_dir),
                                llm_for_capability=lambda name: cap)()
    receipt = result["communities"]
    assert receipt["embedding"]["embedded"] == 0 and receipt["embedding"]["skipped"] >= len(RECORDS), receipt
    notes = [ln for ln in _topic_lines(lines) if "were not embedded this sleep" in ln]
    assert notes == [f"(topics: {receipt['embedding']['skipped']} memories were not embedded this sleep - "
                     f"{receipt['embedding']['skipped']} no_embedder; the pass places them by their neighbours)"], lines
    assert not [ln for ln in lines if ln.startswith("#AUTHORING_FAILURE")], lines
    assert abstractmemory  # the import is the module under test's
