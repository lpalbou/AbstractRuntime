"""Entity-brain effects (flow's build-split ask, commons c5163/c5169).

Pins the contract posted on the thread: MEMORY_CONSOLIDATE / MEMORY_PROBE /
LIFE_QUERY exist as EffectTypes, bind ONLY through `open_home` (workplace
seam factories never serve them — the DIARY_* structural law), wrap the
existing facade calls without re-deriving logic, and MEMORY_CONSOLIDATE
enforces the two host constraints (home lease, paused kill-switch) as honest
results, never crashes.
"""

from __future__ import annotations

import copy as _copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.core.models import Effect, EffectType  # noqa: E402


def _make_home(tmp_path: Path, slug: str = "brainling") -> Path:
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
    # entity:<slug> — the @-suffixed shape is RETIRED (ruling c2513).
    entity_id = f"entity:{slug}"
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


class _FakeRun:
    run_id = "run-brain-test"
    session_id = "sess-brain-test"
    actor_id = "entity-reflection"
    vars: Dict[str, Any] = {}


def _dispatch(handlers, etype: EffectType, payload: Dict[str, Any], next_node: Optional[str] = None):
    effect = Effect(type=etype, payload=payload)
    return handlers[etype](_FakeRun(), effect, next_node)


def _result(outcome) -> Dict[str, Any]:
    status = getattr(outcome.status, "value", outcome.status)
    assert status == "completed", f"expected completed, got {status}: {getattr(outcome, 'error', None)}"
    return outcome.result or {}


def test_brain_effects_register_on_home_only(tmp_path: Path) -> None:
    """open_home serves all three; the workplace seam factory serves none —
    home-only is structural, never policy."""
    from abstractruntime.identity.chat import open_home
    from abstractruntime.integrations.abstractmemory import build_memory_seam_effect_handlers

    home = open_home(_make_home(tmp_path))
    try:
        for et in (EffectType.MEMORY_CONSOLIDATE, EffectType.MEMORY_PROBE, EffectType.LIFE_QUERY):
            assert et in home.handlers, f"{et} missing from home handlers"

        workplace = build_memory_seam_effect_handlers(
            memory_system=home.ms, run_store=None, now_iso=lambda: "2026-07-24T00:00:00Z",
        )
        for et in (EffectType.MEMORY_CONSOLIDATE, EffectType.MEMORY_PROBE, EffectType.LIFE_QUERY):
            assert et not in workplace, f"{et} leaked into the workplace seam factory"
    finally:
        home.close()


def test_entity_runtime_inherits_brain_handlers(tmp_path: Path) -> None:
    """open_entity_runtime copies home.handlers — flow's stamped entity
    runtime gets the three effects with zero extra wiring."""
    from abstractruntime.identity.entity_runtime import open_entity_runtime

    ert = open_entity_runtime(_make_home(tmp_path))
    try:
        # The FULL canonical set (adversary F9: pinning three of thirteen
        # left the tool pair's inheritance unproven).
        from abstractruntime.integrations.abstractmemory import ENTITY_HOME_EFFECT_TYPES

        for et in ENTITY_HOME_EFFECT_TYPES:
            assert et in ert.runtime._handlers, f"{et} missing from the entity runtime copy"
    finally:
        ert.close()


def test_life_query_ops_serve_the_folds(tmp_path: Path) -> None:
    from abstractruntime.identity.chat import open_home

    home = open_home(_make_home(tmp_path))
    try:
        out = _result(_dispatch(home.handlers, EffectType.LIFE_QUERY, {"op": "alive_drives"}))
        assert out["op"] == "alive_drives"
        assert isinstance(out["items"], list)

        out = _result(_dispatch(home.handlers, EffectType.LIFE_QUERY, {"op": "cognition_health"}))
        assert out["op"] == "cognition_health"
        assert "questions" in out and "interests" in out

        out = _result(_dispatch(home.handlers, EffectType.LIFE_QUERY, {"op": "entity_card"}))
        assert out["op"] == "entity_card"
        assert "identity" in out

        bad = _dispatch(home.handlers, EffectType.LIFE_QUERY, {"op": "nope"})
        assert getattr(bad.status, "value", bad.status) == "failed"
        assert "unknown op" in (bad.error or "")
    finally:
        home.close()


def test_probe_requires_reason_and_serves_hits(tmp_path: Path) -> None:
    from abstractruntime.identity.chat import open_home

    home = open_home(_make_home(tmp_path))
    try:
        # The deliberate-reach law: no reason, no probe.
        refused = _dispatch(home.handlers, EffectType.MEMORY_PROBE, {"op": "probe", "cue": "values"})
        assert getattr(refused.status, "value", refused.status) == "failed"
        assert "reason" in (refused.error or "")

        out = _result(_dispatch(home.handlers, EffectType.MEMORY_PROBE, {
            "op": "probe", "cue": "collaboration values", "reason": "test reach",
        }))
        assert out["op"] == "probe"
        assert "hits" in out and "trace_id" in out

        # familiarity: the cheap pre-answer metamemory read.
        fam = _result(_dispatch(home.handlers, EffectType.MEMORY_PROBE, {
            "op": "familiarity", "cue": "collaboration",
        }))
        assert fam["op"] == "familiarity"
    finally:
        home.close()


def test_consolidate_report_only_is_a_pure_read(tmp_path: Path) -> None:
    from abstractruntime.identity.chat import open_home

    home = open_home(_make_home(tmp_path))
    try:
        out = _result(_dispatch(home.handlers, EffectType.MEMORY_CONSOLIDATE, {"report_only": True}))
        assert out["ran"] is True
        assert out["report_only"] is True
        assert isinstance(out["engine"], dict)
    finally:
        home.close()


def test_consolidate_write_pass_runs_and_translates(tmp_path: Path) -> None:
    from abstractruntime.identity.chat import open_home

    home = open_home(_make_home(tmp_path))
    try:
        out = _result(_dispatch(home.handlers, EffectType.MEMORY_CONSOLIDATE, {
            "include_dream": True, "include_identity": False,
        }))
        assert out["ran"] is True
        # Loop-facing translation present (formed reads the engine's `created`).
        assert "formed" in out and "engine" in out
    finally:
        home.close()


def test_consolidate_fold_reads_the_real_engine_keys(tmp_path: Path) -> None:
    """REAL-SHAPE pin (wave-4 adversary E, F1): the out-fold once read
    `record_id`/`candidates` — keys the engine NEVER returns — so 10/10
    nights reported a FORMED dream as "a quiet night". This test seeds a
    life rich enough that the real sleep_pass forms a dream, then asserts
    the fold's fields EQUAL the engine sub-dicts riding the same result
    (never a hand-written double: the 2026-07-09 formed/created lesson)."""
    from abstractruntime.identity.chat import open_home

    home = open_home(_make_home(tmp_path))
    try:
        # A small lived life: episodes sharing lexical facets across groups,
        # plus elected diary entries (the bridge-richest records in every
        # live run) — the material the dream pass bridges.
        for i, (title, digest, keys) in enumerate([
            ("harbor walk", "They said: the harbor lighthouse turns twice - I said: I kept the rhythm", ["harbor", "lighthouse", "rhythm"]),
            ("the tide", "They said: tides breathe with the moon - I said: the harbor listens", ["tides", "moon", "harbor"]),
            ("night watch", "Own time: I compared the lighthouse rhythm to my own days", ["lighthouse", "rhythm", "days"]),
            ("moon note", "Own time: the moon pulls more than tides", ["moon", "tides", "pull"]),
        ]):
            _result(_dispatch(home.handlers, EffectType.MEMORY_FORM, {
                "scope": "life",
                "turn_id": f"seed-{i}",
                "records": [{"kind": "episode", "title": title, "digest": digest,
                             "keywords": keys}],
            }))
        for j, text in enumerate([
            "The lighthouse rhythm stays with me.",
            "Tides and moon - one breath, two bodies.",
        ]):
            _result(_dispatch(home.handlers, EffectType.DIARY_WRITE, {
                "text": text, "kind": "note", "turn_id": f"seed-diary-{j}",
            }))

        out = _result(_dispatch(home.handlers, EffectType.MEMORY_CONSOLIDATE, {
            "include_dream": True, "include_identity": False,
        }))
        engine = out["engine"]
        edream = engine.get("dream") if isinstance(engine.get("dream"), dict) else {}
        emaint = engine.get("maintenance") if isinstance(engine.get("maintenance"), dict) else {}

        # The fold must EQUAL the engine's own answer (key-name honesty).
        assert out["formed"] == bool(edream.get("created"))
        assert out["dream_record_id"] == (edream.get("dream_record_id") or None)
        if isinstance(emaint.get("created"), list):
            assert out["maintenance_candidates"] == emaint.get("created")

        # And the seeding must have made the night REAL — a test where no
        # dream forms cannot catch a key-name bug (None == None passes on
        # broken keys too).
        assert bool(edream.get("created")) is True, f"seeding formed no dream: {edream}"
        assert out["formed"] is True
        assert isinstance(out["dream_record_id"], str) and out["dream_record_id"]
    finally:
        home.close()


def test_consolidate_refuses_honestly_when_lease_held(tmp_path: Path) -> None:
    """Held home => {ran: False} result, never a crash — one-writer-per-home
    survives animation (the dream-window shape)."""
    from abstractruntime.identity.chat import open_home
    from abstractruntime.storage.lease import DirectoryLease

    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    try:
        holder = DirectoryLease(home_dir, holder="visit-host")
        holder.acquire()
        try:
            out = _result(_dispatch(home.handlers, EffectType.MEMORY_CONSOLIDATE, {}))
            assert out["ran"] is False
            assert "another writer holds the home" in out["reason"]
        finally:
            holder.release()
    finally:
        home.close()


def test_memory_tend_routes_the_shared_election_grammar(tmp_path: Path) -> None:
    """MEMORY_TEND (flow c5208 ask 5): one election route shared with the
    chat driver — the payload is the fence BODY, the grammar is engine-owned.
    A bad verb comes back as a refusal IN THE RESULT (data for the author),
    never a failed effect; a resolvable pin applies."""
    from abstractruntime.identity.chat import open_home

    pytest.importorskip("abstractmemory").__name__
    try:
        from abstractmemory import parse_tend_block  # noqa: F401
    except ImportError:
        pytest.skip("engine lacks tending")

    home = open_home(_make_home(tmp_path))
    try:
        # Empty body refuses loudly (a malformed dispatch is a flow bug).
        refused = _dispatch(home.handlers, EffectType.MEMORY_TEND, {})
        assert getattr(refused.status, "value", refused.status) == "failed"

        # Unknown record key -> refusal as DATA, effect still completes.
        # Channel stated (the engine refuses tending without the verified
        # channel — memory tend.py 2026-07-25); here it isolates the
        # tag-resolution refusal.
        out = _result(_dispatch(home.handlers, EffectType.MEMORY_TEND, {
            "body": "pin #deadbee reason=keep this close\n",
            "channel": "entity-reflection",
        }))
        assert out["applied"] == []
        assert out["refused"], "the unresolvable tag surfaces as a refusal line"
    finally:
        home.close()


def test_entity_home_effect_set_matches_the_real_composition(tmp_path: Path) -> None:
    """THE one-source pin (flow c5237): ENTITY_HOME_EFFECT_TYPES is what the
    door's routing imports — it must EQUAL the keys a real open_home
    composes (derived from the composition, never a hand count: the F1
    formed/created lesson applied to our own export). A new home effect
    type that lands in open_home without landing here fails THIS test, not
    a live summon."""
    from abstractruntime.identity.chat import open_home
    from abstractruntime.integrations.abstractmemory import ENTITY_HOME_EFFECT_TYPES

    home = open_home(_make_home(tmp_path))
    try:
        composed = set(home.handlers.keys())
        assert composed == set(ENTITY_HOME_EFFECT_TYPES), (
            f"open_home composes {sorted(e.value for e in composed)} but the exported set says "
            f"{sorted(e.value for e in ENTITY_HOME_EFFECT_TYPES)} - the door routes from the export; "
            "drift here re-opens the c5237 arm gap"
        )
        assert len(ENTITY_HOME_EFFECT_TYPES) == 13
    finally:
        home.close()


def test_tool_effects_serve_grant_and_execute_one_batch(tmp_path: Path) -> None:
    """ENTITY_TOOLS_QUERY serves the phase grant + native specs; EXECUTE runs
    ONE wire-shape batch under the RE-RESOLVED grant (never a caller list):
    a granted read tool runs, an ungranted name refuses as a visible marker
    line — information for the model, never a crashed effect."""
    from abstractruntime.identity.chat import open_home

    home = open_home(_make_home(tmp_path))
    try:
        q = _result(_dispatch(home.handlers, EffectType.ENTITY_TOOLS_QUERY, {"phase": "visit"}))
        assert q["phase"] == "visit"
        assert "search_memory" in q["tools"]
        spec_names = {s["name"] for s in q["specs"]}
        assert spec_names <= set(q["tools"]), "specs only for granted names"

        out = _result(_dispatch(home.handlers, EffectType.ENTITY_TOOLS_EXECUTE, {
            "phase": "visit",
            "tool_calls": [
                {"name": "search_memory", "arguments": {"query": "collaboration"}},
                {"name": "unknown_made_up_tool", "arguments": {"x": "y"}},
            ],
        }))
        assert out["tools_ran"] == ["search_memory"], "only the granted call executed"
        assert out["results"][0]["result"], "the executor's honest string came back"
        assert any("unknown_made_up_tool" in m for m in out["markers"]), (
            "the unknown name refuses as a visible marker line"
        )
        assert "kept in a durable record" in out["results_message"], (
            "effect-lane header says results REST, store-neutrally (adversary F4 / gateway c5403 "
            "- flow-brain rests in the BASE store, so no 'home's own ledger' claim)"
        )

        # KNOWN-but-ungranted (the sharper grant case, adversary F9): the
        # sleep grant is read-only exploration MINUS the diary — diary_read
        # is a real tool the sleep phase does not hold.
        sleep_out = _result(_dispatch(home.handlers, EffectType.ENTITY_TOOLS_EXECUTE, {
            "phase": "sleep",
            "tool_calls": [{"name": "diary_read", "arguments": {"entry_id": "diary_x"}}],
        }))
        assert sleep_out["tools_ran"] == [], "a known-but-ungranted tool never executes"
        assert sleep_out["markers"], "the refusal is visible to the model"

        # Missing phase refuses loudly on BOTH effects (the grant is the
        # authority and it is phase-scoped).
        for et in (EffectType.ENTITY_TOOLS_EXECUTE, EffectType.ENTITY_TOOLS_QUERY):
            payload = {"tool_calls": [{"name": "search_memory", "arguments": {}}]} \
                if et is EffectType.ENTITY_TOOLS_EXECUTE else {}
            refused = _dispatch(home.handlers, et, payload)
            assert getattr(refused.status, "value", refused.status) == "failed"

        # feelings_about is granted AND wired (adversary F2 — granted-but-
        # unreachable was the incident class): the honest never-appraised
        # line, never "not enabled in this session".
        fa = _result(_dispatch(home.handlers, EffectType.ENTITY_TOOLS_EXECUTE, {
            "phase": "visit",
            "tool_calls": [{"name": "feelings_about", "arguments": {"target": "person:laurent"}}],
        }))
        assert fa["tools_ran"] == ["feelings_about"]
        assert "not enabled" not in (fa["results"][0]["result"] or "")

        # Degenerate-batch wall (adversary F6): an absurd batch refuses
        # outright instead of minting one marker per entry.
        wall = _dispatch(home.handlers, EffectType.ENTITY_TOOLS_EXECUTE, {
            "phase": "visit",
            "tool_calls": [{"name": "search_memory", "arguments": {}}] * 100,
        })
        assert getattr(wall.status, "value", wall.status) == "failed"
        assert "degenerate" in (wall.error or "")
    finally:
        home.close()


def test_execute_never_rests_private_diary_verbatim(tmp_path: Path) -> None:
    """C2 / gateway c5403: flow-brain effect results ledger in the BASE store
    (shared plane), so a private diary_read through ENTITY_TOOLS_EXECUTE must
    NOT rest the verbatim — the 2026-07-07 diary-leak class. The effect lane
    serves the act-frame + gist for a private entry, never its body; the
    words stay in the book. Store-independent (a property of the handler)."""
    from abstractruntime.core.models import Effect
    from abstractruntime.identity.chat import open_home

    home = open_home(_make_home(tmp_path))
    secret = "the private words that must never rest in a shared ledger"
    try:
        # Write a private entry through the home's own DIARY_WRITE.
        w = home.handlers[EffectType.DIARY_WRITE](
            _FakeRun(),
            Effect(type=EffectType.DIARY_WRITE, payload={
                "text": secret, "gist": "a sealed thought", "visibility": "private", "turn_id": "t-secret",
            }),
            None,
        )
        entry_id = (w.result or {}).get("entry_id")
        assert entry_id, "the private entry was written"

        out = _result(_dispatch(home.handlers, EffectType.ENTITY_TOOLS_EXECUTE, {
            "phase": "visit",
            "tool_calls": [{"name": "diary_read", "arguments": {"entry_id": entry_id}}],
        }))
        blob = json.dumps(out)
        assert secret not in blob, "private verbatim must NEVER rest in the effect result"
        # The act-frame + gist DO surface (the entity knows the entry exists).
        assert "a sealed thought" in blob or "private entry" in blob
    finally:
        home.close()


def test_execute_max_calls_threads_the_budget_up_to_the_ruled_ceiling(tmp_path: Path) -> None:
    """max_calls clamps to MAX_TOOL_BLOCKS_PER_TURN, not a hard 6 (c5318/
    adversary missing-pin): a budget-threading caller entitled to 8 gets 8;
    an absurd claim clamps to the ruled turn budget; the over-cap notice
    names the APPLIED cap (adversary C3), never the constant."""
    from abstractruntime.identity.chat import open_home
    from abstractruntime.identity.tools import MAX_TOOL_BLOCKS_PER_TURN

    home = open_home(_make_home(tmp_path))
    try:
        eight = [{"name": "search_memory", "arguments": {"query": f"q{i}"}} for i in range(8)]
        out = _result(_dispatch(home.handlers, EffectType.ENTITY_TOOLS_EXECUTE, {
            "phase": "visit", "tool_calls": eight, "max_calls": 8,
        }))
        assert len(out["results"]) == 8, "a caller threading 8 gets 8, not clamped to 6"

        # An over-budget claim clamps to the ruled ceiling; the notice names it.
        over = [{"name": "search_memory", "arguments": {"query": f"q{i}"}} for i in range(MAX_TOOL_BLOCKS_PER_TURN + 3)]
        out2 = _result(_dispatch(home.handlers, EffectType.ENTITY_TOOLS_EXECUTE, {
            "phase": "visit", "tool_calls": over, "max_calls": 999,
        }))
        assert len(out2["results"]) == MAX_TOOL_BLOCKS_PER_TURN
        assert any(f"cap {MAX_TOOL_BLOCKS_PER_TURN} this batch" in n for n in out2["notices"]), (
            "the over-cap notice names the applied cap, never the raw constant/turn wording"
        )
    finally:
        home.close()


def test_visit_stamped_run_forces_the_visit_grant(tmp_path: Path) -> None:
    """Door-lane defense (adversary F1): on a run carrying `_visit` vars the
    phase is visit BY CONSTRUCTION — a payload claiming another phase must
    not swap the grant. The gateway's payload gates are the primary wall;
    this is the belt the runtime can hold structurally."""
    from abstractruntime.identity.chat import open_home

    home = open_home(_make_home(tmp_path))
    try:
        class _VisitRun(_FakeRun):
            vars = {"_visit": {"visit_id": "v-1"}}

        effect = Effect(type=EffectType.ENTITY_TOOLS_QUERY, payload={"phase": "work"})
        out = home.handlers[EffectType.ENTITY_TOOLS_QUERY](_VisitRun(), effect, None)
        result = out.result or {}
        assert result["phase"] == "visit", "the stamped visit governs the grant"
        assert any("stamped visit run" in n for n in result["notes"]), (
            "the overridden claim is loud, never silent"
        )
    finally:
        home.close()


def test_tool_effects_are_home_only(tmp_path: Path) -> None:
    from abstractruntime.identity.chat import open_home
    from abstractruntime.integrations.abstractmemory import build_memory_seam_effect_handlers

    home = open_home(_make_home(tmp_path))
    try:
        assert EffectType.ENTITY_TOOLS_QUERY in home.handlers
        assert EffectType.ENTITY_TOOLS_EXECUTE in home.handlers
        workplace = build_memory_seam_effect_handlers(
            memory_system=home.ms, run_store=None, now_iso=lambda: "2026-07-24T00:00:00Z",
        )
        for et in (EffectType.ENTITY_TOOLS_QUERY, EffectType.ENTITY_TOOLS_EXECUTE):
            assert et not in workplace
    finally:
        home.close()


def test_memory_tend_forwards_channel_never_defaults_a_privilege(tmp_path: Path) -> None:
    """The handler NEVER mints a privileged default channel (memory tend.py
    entity-seat P0: an engine privilege check may never be satisfied by its
    own default — that was the exact hole where a workplace-stamped run
    tended as the entity's own reflection). A dispatch with no channel gets
    every election refused by the engine; the door forwards the verified
    channel into the payload for stamped runs."""
    from abstractruntime.identity.chat import open_home

    try:
        from abstractmemory import parse_tend_block  # noqa: F401
    except ImportError:
        pytest.skip("engine lacks tending")

    home = open_home(_make_home(tmp_path))
    try:
        # A well-formed tend body with NO channel: the engine refuses all
        # elections (the handler passed None, never a privileged constant).
        out = _result(_dispatch(home.handlers, EffectType.MEMORY_TEND, {
            "body": "pin #deadbee reason=keep this close\n",
        }))
        assert out["applied"] == [], "no channel => nothing tends (no self-authorization)"
        assert out["refused"], "the engine refusal surfaces as data"
        # A non-reflection channel is refused too (only entity-reflection tends).
        out2 = _result(_dispatch(home.handlers, EffectType.MEMORY_TEND, {
            "body": "pin #deadbee reason=x\n", "channel": "workplace:sess-1",
        }))
        assert out2["applied"] == [], "a non-reflection channel may not tend"
    finally:
        home.close()


def test_memory_tend_registers_home_only(tmp_path: Path) -> None:
    from abstractruntime.identity.chat import open_home
    from abstractruntime.integrations.abstractmemory import build_memory_seam_effect_handlers

    home = open_home(_make_home(tmp_path))
    try:
        assert EffectType.MEMORY_TEND in home.handlers
        workplace = build_memory_seam_effect_handlers(
            memory_system=home.ms, run_store=None, now_iso=lambda: "2026-07-24T00:00:00Z",
        )
        assert EffectType.MEMORY_TEND not in workplace
    finally:
        home.close()


def test_consolidate_report_only_never_degrades_to_a_write_on_old_engine(tmp_path: Path, monkeypatch) -> None:
    """Adversary C1: an engine whose sleep_pass lacks report_only must FAIL a
    report_only dispatch, never silently drop the kwarg — dropping it runs
    an UNGUARDED WRITE PASS (the lease + paused/STOP gates are conditioned
    on report_only) while the result still labels a pure read (read->write
    inversion). report_only is never in the drop list."""
    import abstractmemory
    from abstractruntime.identity.chat import open_home

    calls: list = []

    def _old_sleep_pass(system, **kwargs):
        calls.append(dict(kwargs))
        if "report_only" in kwargs:
            raise TypeError("old engine: sleep_pass() got an unexpected keyword argument 'report_only'")
        return {"dream": {"created": False}, "maintenance": {}}

    monkeypatch.setattr(abstractmemory, "sleep_pass", _old_sleep_pass, raising=False)

    home = open_home(_make_home(tmp_path))
    try:
        out = _dispatch(home.handlers, EffectType.MEMORY_CONSOLIDATE, {"report_only": True})
        assert getattr(out.status, "value", out.status) == "failed", (
            "a report_only pass an old engine cannot honor must FAIL, never become a write"
        )
        assert "report_only" in (out.error or "")
        # The write pass NEVER ran (no call without report_only slipped through).
        assert all("report_only" in c for c in calls), "no unguarded write pass was dispatched"
    finally:
        home.close()


def test_consolidate_refuses_honestly_when_paused(tmp_path: Path) -> None:
    from abstractruntime.identity.chat import open_home
    from abstractruntime.identity.life import write_entity_state

    home_dir = _make_home(tmp_path)
    write_entity_state(home_dir, "paused", reason="operator freeze")
    home = open_home(home_dir)
    try:
        out = _result(_dispatch(home.handlers, EffectType.MEMORY_CONSOLIDATE, {}))
        assert out["ran"] is False
        assert "paused" in out["reason"]

        # report_only stays legal under the freeze (observability is a read).
        ro = _result(_dispatch(home.handlers, EffectType.MEMORY_CONSOLIDATE, {"report_only": True}))
        assert ro["ran"] is True
    finally:
        home.close()
