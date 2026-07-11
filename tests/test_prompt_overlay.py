"""Operator prompt overlay — the editable layer of the entity system prompt.

Maintainer ask (2026-07-11): a workspace tab where the system prompt can be
REWRITTEN. Ownership rules under test:

- missing file  = built-in defaults, byte-identical composition;
- `conversation` / `visit` REPLACE their built-in paragraphs;
- `own_time` replaces OWN_TIME_CONTRACT on own-time sessions (life factory);
- `operator` appends LAST, attributed ("from your operator") — words in the
  head must never pretend to be the entity's own;
- the identity prelude and the tools contract are NOT overlay-editable:
  identity evolves by the entity's own acts, tools text derives from the
  actual grant;
- a malformed file degrades loudly (#FALLBACK note), never blocks a summon;
- write-side: unknown keys refuse; all-empty overlay deletes the file.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractruntime.identity.chat import (
    CONTRACT_PARAGRAPH,
    VISIT_OWN_TIME_PARAGRAPH,
    ChatSession,
    compose_system_base,
    open_home,
)
from abstractruntime.identity.prompt_overlay import (
    OVERLAY_FILENAME,
    read_prompt_overlay,
    write_prompt_overlay,
)

pytest.importorskip("abstractmemory")

from abstractmemory import (  # noqa: E402
    DEFAULT_SPARK_TEMPLATE,
    MemorySystem,
    SQLiteJournal,
    SQLiteTripleStore,
    engram,
    lint_spark,
)

ENTITY_SLUG = "castor"
ENTITY_ID = "entity:castor@home-test"


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _ScriptedLLM:
    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def generate(self, *, messages, system_prompt):
        self.calls.append({"messages": list(messages), "system_prompt": system_prompt})
        return _FakeResponse(self.replies.pop(0))


def _create_home(home_dir: Path) -> None:
    import yaml

    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Castor"
    spark["spark"] = 1
    assert lint_spark(spark) == []

    home_dir.mkdir(parents=True, exist_ok=True)
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": ENTITY_ID}), encoding="utf-8")

    store = SQLiteTripleStore(home_dir / "memory.sqlite3")
    journal = SQLiteJournal(home_dir / "memory.sqlite3")
    ms = MemorySystem(store=store, journal=journal)
    result = engram(ms, spark, owner_id=ENTITY_ID)
    assert result.created is True
    store.close()
    journal.close()


@pytest.fixture()
def castor_home(tmp_path: Path) -> Path:
    home = tmp_path / "entities" / ENTITY_SLUG
    _create_home(home)
    return home


def _session(home_dir: Path, **over: Any) -> ChatSession:
    home = open_home(home_dir)
    kwargs: Dict[str, Any] = dict(
        participants=["person:albou"],
        session_id="s1",
        context_window=32768,
        out=lambda s: None,
    )
    kwargs.update(over)
    return ChatSession(home, _ScriptedLLM([]), **kwargs)


# ---------------------------------------------------------------- file layer


class TestOverlayFile:
    def test_missing_file_reads_empty(self, tmp_path):
        assert read_prompt_overlay(tmp_path) == {}

    def test_write_then_read_round_trips(self, tmp_path):
        write_prompt_overlay(tmp_path, {"conversation": "Talk plainly.", "operator": "Always answer in French."})
        got = read_prompt_overlay(tmp_path)
        assert got == {"conversation": "Talk plainly.", "operator": "Always answer in French."}

    def test_unknown_key_refuses_loudly(self, tmp_path):
        with pytest.raises(ValueError) as e:
            write_prompt_overlay(tmp_path, {"identity": "I am someone else"})
        assert "identity" in str(e.value)

    def test_empty_values_drop_and_all_empty_deletes_the_file(self, tmp_path):
        write_prompt_overlay(tmp_path, {"conversation": "X"})
        assert (tmp_path / OVERLAY_FILENAME).exists()
        write_prompt_overlay(tmp_path, {"conversation": "  "})
        assert not (tmp_path / OVERLAY_FILENAME).exists()
        assert read_prompt_overlay(tmp_path) == {}

    def test_malformed_file_reads_as_labeled_error(self, tmp_path):
        (tmp_path / OVERLAY_FILENAME).write_text("{not yaml: [", encoding="utf-8")
        got = read_prompt_overlay(tmp_path)
        assert got.get("#error") == "unreadable"

    def test_hand_edited_unknown_keys_are_ignored_on_read(self, tmp_path):
        import yaml

        (tmp_path / OVERLAY_FILENAME).write_text(
            yaml.safe_dump({"conversation": "Y", "typo_key": "Z"}), encoding="utf-8"
        )
        assert read_prompt_overlay(tmp_path) == {"conversation": "Y"}

    def test_oversized_layer_refuses_loudly(self, tmp_path):
        """A pasted-in 10MB layer would starve recall of the context window
        with no budget accounting anywhere — refuse at the write boundary."""
        from abstractruntime.identity.prompt_overlay import MAX_LAYER_CHARS

        with pytest.raises(ValueError) as e:
            write_prompt_overlay(tmp_path, {"operator": "x" * (MAX_LAYER_CHARS + 1)})
        assert "cap" in str(e.value)
        assert not (tmp_path / OVERLAY_FILENAME).exists()


# ---------------------------------------------------------- composition unit


class TestComposeSystemBase:
    def test_defaults_without_overlay(self):
        base = compose_system_base("<prelude/>", phase="visit", overlay={})
        assert CONTRACT_PARAGRAPH in base
        assert VISIT_OWN_TIME_PARAGRAPH in base
        assert "STANDING INSTRUCTIONS" not in base

    def test_overlay_replaces_layers_and_appends_operator_last(self):
        base = compose_system_base(
            "<prelude/>",
            phase="visit",
            overlay={"conversation": "CONV-REWRITE", "visit": "VISIT-REWRITE", "operator": "OP-NOTE"},
        )
        assert "CONV-REWRITE" in base and CONTRACT_PARAGRAPH not in base
        assert "VISIT-REWRITE" in base and VISIT_OWN_TIME_PARAGRAPH not in base
        assert base.endswith("STANDING INSTRUCTIONS FROM YOUR OPERATOR:\nOP-NOTE")

    def test_own_time_phase_skips_visit_paragraph(self):
        base = compose_system_base("<prelude/>", phase="own_time", overlay={"visit": "VISIT-REWRITE"})
        assert "VISIT-REWRITE" not in base
        assert VISIT_OWN_TIME_PARAGRAPH not in base

    def test_own_time_text_lands_before_operator_block(self):
        """Operator-last must hold in EVERY phase (adversary finding: the
        old life-factory append buried the attributed block mid-head)."""
        base = compose_system_base(
            "<prelude/>", phase="own_time",
            overlay={"operator": "OP-NOTE"},
            own_time_text="OWN-TIME-TEXT",
        )
        assert base.index("OWN-TIME-TEXT") < base.index("STANDING INSTRUCTIONS")
        assert base.endswith("STANDING INSTRUCTIONS FROM YOUR OPERATOR:\nOP-NOTE")

    def test_default_prompt_texts_names_every_overlay_key(self):
        from abstractruntime.identity.chat import default_prompt_texts
        from abstractruntime.identity.life import OWN_TIME_CONTRACT
        from abstractruntime.identity.prompt_overlay import OVERLAY_KEYS

        texts = default_prompt_texts()
        assert set(texts) == set(OVERLAY_KEYS)
        assert texts["own_time"] == OWN_TIME_CONTRACT


# ------------------------------------------------------------- session wiring


class TestSessionComposition:
    def test_no_overlay_is_byte_identical_default(self, castor_home):
        s = _session(castor_home)
        assert CONTRACT_PARAGRAPH in s.system_base
        assert VISIT_OWN_TIME_PARAGRAPH in s.system_base
        s.home.close()

    def test_overlay_rewrites_conversation_and_visit(self, castor_home):
        write_prompt_overlay(
            castor_home,
            {"conversation": "Speak in your own words; memories may follow.", "visit": "Your life continues after this visit."},
        )
        s = _session(castor_home)
        assert "Speak in your own words" in s.system_base
        assert CONTRACT_PARAGRAPH not in s.system_base
        assert "Your life continues after this visit." in s.system_base
        assert VISIT_OWN_TIME_PARAGRAPH not in s.system_base
        s.home.close()

    def test_operator_notes_append_last_and_attributed(self, castor_home):
        write_prompt_overlay(castor_home, {"operator": "Prefer metric units."})
        s = _session(castor_home)
        assert s.system_base.endswith("STANDING INSTRUCTIONS FROM YOUR OPERATOR:\nPrefer metric units.")
        s.home.close()

    def test_identity_prelude_survives_any_overlay(self, castor_home):
        write_prompt_overlay(castor_home, {"conversation": "X", "visit": "Y", "operator": "Z"})
        s = _session(castor_home)
        assert "You are Castor." in s.system_base  # the prelude is not the overlay's to touch
        s.home.close()

    def test_malformed_overlay_degrades_loudly_never_blocks(self, castor_home):
        (castor_home / OVERLAY_FILENAME).write_text("{not yaml: [", encoding="utf-8")
        notes: List[str] = []
        s = _session(castor_home, out=notes.append)
        assert CONTRACT_PARAGRAPH in s.system_base  # defaults used
        assert any("#FALLBACK" in n and "system_prompt.yaml" in n for n in notes)
        s.home.close()

    def test_overlay_reaches_the_durable_visit_arm(self, castor_home):
        """The third composition point (visit_workflow.open_node) reads the
        same file — a drift here would give the drawer a different head
        than the CLI for the same entity."""
        from abstractruntime.core.models import EffectType
        from abstractruntime.identity.entity_runtime import open_entity_runtime
        from abstractruntime.identity.visit_workflow import build_visit_workflow

        write_prompt_overlay(
            castor_home,
            {"conversation": "DURABLE-CONV-REWRITE", "operator": "DURABLE-OP-NOTE"},
        )

        class _NeverCalledLLM:
            def __call__(self, run, effect, dnn=None):  # pragma: no cover
                raise AssertionError("OPEN must not reach the LLM")

        ert = open_entity_runtime(castor_home, extra_handlers={EffectType.LLM_CALL: _NeverCalledLLM()})
        try:
            wf = build_visit_workflow(ert.home, participants=["person:albou"], idle_seconds=3600)
            run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-overlay")
            state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=10)
            base = str(state.vars["_visit"]["system_base"])
            assert "DURABLE-CONV-REWRITE" in base
            assert CONTRACT_PARAGRAPH not in base
            assert VISIT_OWN_TIME_PARAGRAPH in base  # untouched layer keeps its default
            assert base.endswith("STANDING INSTRUCTIONS FROM YOUR OPERATOR:\nDURABLE-OP-NOTE")
        finally:
            ert.close()

    def test_own_time_overlay_reaches_own_time_sessions(self, castor_home, monkeypatch):
        """The life factory swaps OWN_TIME_CONTRACT for the overlay's text."""
        from abstractruntime.identity.life import OWN_TIME_CONTRACT, build_session_factory

        write_prompt_overlay(castor_home, {"own_time": "This hour is yours alone."})

        import abstractcore

        class _LLMStub:
            def generate(self, **kwargs):
                return _FakeResponse("ok")

        monkeypatch.setattr(abstractcore, "create_llm", lambda *a, **k: _LLMStub())
        factory = build_session_factory(
            castor_home, provider="lmstudio", model="test", base_url="http://x",
            embedding_model=None, embedding_base_url="http://x",
            context_window=32768, out=lambda s: None,
        )
        session = factory()
        assert "This hour is yours alone." in session.system_base
        assert OWN_TIME_CONTRACT not in session.system_base
        session.home.close()

    def test_own_time_operator_block_stays_last_after_own_time(self, castor_home, monkeypatch):
        """The re-compose fix: own-time contract BEFORE the operator block
        (the old append put the attributed block mid-head)."""
        from abstractruntime.identity.life import OWN_TIME_CONTRACT, build_session_factory

        write_prompt_overlay(castor_home, {"operator": "Wrap up by dawn."})

        import abstractcore

        class _LLMStub:
            def generate(self, **kwargs):
                return _FakeResponse("ok")

        monkeypatch.setattr(abstractcore, "create_llm", lambda *a, **k: _LLMStub())
        factory = build_session_factory(
            castor_home, provider="lmstudio", model="test", base_url="http://x",
            embedding_model=None, embedding_base_url="http://x",
            context_window=32768, out=lambda s: None,
        )
        session = factory()
        assert OWN_TIME_CONTRACT in session.system_base
        assert session.system_base.index(OWN_TIME_CONTRACT) < session.system_base.index("STANDING INSTRUCTIONS")
        assert session.system_base.endswith("STANDING INSTRUCTIONS FROM YOUR OPERATOR:\nWrap up by dawn.")
        session.home.close()
