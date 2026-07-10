"""Per-entity runtime rooted in the home (plan item 8, runtime R2 half).

Pins the composition contract: the run store is `runtime_<slug>.sqlite3`
INSIDE the home (slug = directory name, the registry key); pending runs and
durable waits TRAVEL when the home directory is copied; effects executed
through the entity's Runtime land in the home's own stores (memory graph,
diary book); two homes have two fully isolated run stores; host handlers
may extend but never shadow the home's own.
"""

from __future__ import annotations

import copy as _copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.core.models import Effect, EffectType, RunStatus, StepPlan  # noqa: E402
from abstractruntime.core.spec import WorkflowSpec  # noqa: E402
from abstractruntime.identity.entity_runtime import (  # noqa: E402
    EntityRuntime,
    entity_run_store_path,
    open_entity_runtime,
)


def _make_home(tmp_path: Path, slug: str = "runling") -> Path:
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


def _visit_like_workflow(entity_id: str) -> WorkflowSpec:
    """A minimal visit-shaped run: form a memory, write a diary entry, then
    park on a durable WAIT_EVENT for the visitor's next message."""

    def form_node(run: Any, ctx: Any) -> StepPlan:
        return StepPlan(
            node_id="FORM",
            effect=Effect(
                type=EffectType.MEMORY_FORM,
                payload={
                    "records": [{
                        "kind": "episode",
                        "title": "the first visit moment",
                        "digest": "a visitor said hello and the entity replied",
                        "keywords": ["visit", "hello"],
                    }],
                    "scope": "life",
                    "owner_id": entity_id,
                    "turn_id": "t-0001",
                },
                result_key="_temp.formed",
            ),
            next_node="DIARY",
        )

    def diary_node(run: Any, ctx: Any) -> StepPlan:
        return StepPlan(
            node_id="DIARY",
            effect=Effect(
                type=EffectType.DIARY_WRITE,
                payload={
                    "text": "A visitor came; I want to remember the hello.",
                    "turn_id": "t-0001",
                    "kind": "note",
                },
                result_key="_temp.diary",
            ),
            next_node="WAIT",
        )

    def wait_node(run: Any, ctx: Any) -> StepPlan:
        return StepPlan(
            node_id="WAIT",
            effect=Effect(
                type=EffectType.WAIT_EVENT,
                payload={"wait_key": "visitor_input", "resume_to_node": "DONE"},
                result_key="_temp.visitor_message",
            ),
            next_node="DONE",
        )

    def done_node(run: Any, ctx: Any) -> StepPlan:
        return StepPlan(node_id="DONE", complete_output={"ok": True})

    return WorkflowSpec(
        workflow_id="wf_visit_like",
        entry_node="FORM",
        nodes={"FORM": form_node, "DIARY": diary_node, "WAIT": wait_node, "DONE": done_node},
    )


def test_run_store_lives_inside_the_home_named_by_slug(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path, slug="runling")
    ert = open_entity_runtime(home_dir)
    try:
        assert ert.store_path == home_dir / "runtime_runling.sqlite3"
        wf = _visit_like_workflow(ert.entity_id)
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-1")
        state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=10)
        assert state.status == RunStatus.WAITING
        assert state.waiting.wait_key == "visitor_input"
        assert ert.store_path.exists()  # the run store is REAL, inside the home
        # The run + its durable wait are readable back from the same store.
        again = ert.runtime.get_state(run_id)
        assert again.status == RunStatus.WAITING
    finally:
        ert.close()


def test_effects_land_in_the_homes_own_stores(tmp_path: Path) -> None:
    """The Runtime is bound to the HOME's handlers: MEMORY_FORM writes the
    home graph; DIARY_WRITE writes the home book — through Runtime.tick,
    not through a driver."""
    from abstractmemory import TripleQuery

    home_dir = _make_home(tmp_path, slug="graphling")
    ert = open_entity_runtime(home_dir)
    try:
        wf = _visit_like_workflow(ert.entity_id)
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-1")
        ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=10)

        life = ert.home.ms.query(TripleQuery(scope="life", owner_id=ert.entity_id, limit=0))
        # The digest is the OBJECT of the dcterms:abstract triple (the
        # canonical record shape); kind/title ride attributes.
        assert any("visitor said hello" in str(getattr(a, "object", "")) for a in life)
        assert any(
            isinstance(a.attributes, dict) and a.attributes.get("record_kind") == "episode"
            for a in life
        )

        entries = ert.home.diary.list_entries()
        assert any("remember the hello" in (e.get("text") or "") for e in entries)
    finally:
        ert.close()


def test_pending_waits_travel_when_the_home_is_copied(tmp_path: Path) -> None:
    """Item 8's payoff line: copying the home moves pending runs, waits,
    and commitments too. A parked visit resumes in the COPY."""
    origin = _make_home(tmp_path, slug="mover")
    ert = open_entity_runtime(origin)
    wf = _visit_like_workflow(ert.entity_id)
    run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-1")
    state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=10)
    assert state.status == RunStatus.WAITING
    ert.close()  # checkpointed: the directory is copy-clean

    # The move: the directory IS the life. Slug stays the same (a rename is
    # a different, refused case — gateway GW-B owns that door).
    new_root = tmp_path / "elsewhere" / "entities"
    new_root.mkdir(parents=True)
    moved = new_root / "mover"
    shutil.copytree(origin, moved)

    ert2 = open_entity_runtime(moved)
    try:
        parked = ert2.runtime.get_state(run_id)
        assert parked.status == RunStatus.WAITING
        assert parked.waiting.wait_key == "visitor_input"
        # The visitor's next message resumes the SAME run at the new door.
        resumed = ert2.runtime.resume(
            workflow=wf, run_id=run_id, wait_key="visitor_input",
            payload={"text": "hello again"}, max_steps=10,
        )
        assert resumed.status == RunStatus.COMPLETED
    finally:
        ert2.close()


def test_two_homes_two_isolated_run_stores(tmp_path: Path) -> None:
    a = open_entity_runtime(_make_home(tmp_path, slug="alpha"))
    b = open_entity_runtime(_make_home(tmp_path, slug="beta"))
    try:
        wf = _visit_like_workflow(a.entity_id)
        run_id = a.runtime.start(workflow=wf, vars={}, session_id="visit-a")
        a.runtime.tick(workflow=wf, run_id=run_id, max_steps=10)
        assert a.runtime.get_state(run_id).status == RunStatus.WAITING
        with pytest.raises(KeyError):
            b.runtime.get_state(run_id)  # B's store never saw A's run
        assert not (b.home.home_dir / "runtime_alpha.sqlite3").exists()
    finally:
        a.close()
        b.close()


def test_moved_home_refuses_before_minting_a_stray_store(tmp_path: Path) -> None:
    """GW-B's refusal, home-direct (gateway extension ask): a copied home
    under a DIFFERENT directory name refuses at open — no
    runtime_<straydir>.sqlite3 ever mints beside the true one."""
    origin = _make_home(tmp_path, slug="truename")
    stray = tmp_path / "entities" / "wrongname"
    shutil.copytree(origin, stray)
    with pytest.raises(ValueError, match="moved-home collision"):
        open_entity_runtime(stray)
    assert not (stray / "runtime_wrongname.sqlite3").exists()
    # The true home still opens.
    ert = open_entity_runtime(origin)
    ert.close()


def test_extra_handlers_extend_but_never_shadow(tmp_path: Path) -> None:
    from abstractruntime.core.runtime import EffectOutcome

    home_dir = _make_home(tmp_path, slug="hostling")

    def fake_llm(run: Any, effect: Effect, default_next_node: Any) -> Any:
        return EffectOutcome.completed({"content": "ok"})

    # Extending with a host LLM handler is the intended composition.
    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: fake_llm})
    ert.close()
    # Shadowing a HOME handler is wiring drift and refuses loudly.
    with pytest.raises(ValueError, match="shadow"):
        open_entity_runtime(home_dir, extra_handlers={EffectType.MEMORY_FORM: fake_llm})
