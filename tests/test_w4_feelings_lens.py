"""W4-render (laurent's decision 2, wave-4): feelings as the lens.

The AUTO half: a FEELINGS block of dated first-person lines renders per
turn for targets the moment touches (memory's stimulus_feelings read),
deduped against the prelude's STANDING section. The ELECT half: the
feelings_about tool walks the why behind ONE target on the entity's
reach. value_refs: a feel election may ground in a value (touches=...)
and the appraisal event carries the resolved ref.
"""
from __future__ import annotations

import copy
import json as _json
import tempfile
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractmemory import (
    DEFAULT_SPARK_TEMPLATE,
    MemorySystem,
    SQLiteJournal,
    SQLiteTripleStore,
    engram,
    lint_spark,
)

from abstractruntime.identity.chat import ChatSession, _feelings_block, open_home


def _mk_home(tmp: Path, slug: str, name: str):
    home_dir = tmp / "entities" / slug
    home_dir.mkdir(parents=True)
    entity_id = f"entity:{slug}@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = name
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(_json.dumps({"entity_id": entity_id}), encoding="utf-8")
    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir, entity_id


class _LLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, *, messages, system_prompt):
        class _R:
            pass

        r = _R()
        r.content = self.replies.pop(0) if self.replies else "…"
        return r


def test_feelings_block_renders_first_person_dated_lines() -> None:
    rows = [
        {"target": "person:laurent", "net": 7.0, "positive_count": 8,
         "negative_count": 1, "standing": "bond", "last_felt": "2026-07-18",
         "matched_via": "participant"},
        {"target": "concept:rush", "net": -3.0, "positive_count": 0,
         "negative_count": 3, "standing": "none", "last_felt": "2026-07-15",
         "matched_via": "cue"},
    ]
    block = _feelings_block(rows)
    assert block.startswith("FEELINGS (")
    assert "I feel deeply warm toward person:laurent (+7 over 9 marks, last 2026-07-18)" in block
    assert "BOND stands" in block
    assert "I feel wary of concept:rush" in block
    assert "feelings_about" in block, "the why-tool is named (reach, not recitation)"
    assert "never decide" not in block, "counterweight lives in the CONTRACT now, not per turn"
    assert _feelings_block([]) == ""


def test_turn_carries_the_lens_and_the_why_walk_answers() -> None:
    tmp = Path(tempfile.mkdtemp())
    home_dir, entity_id = _mk_home(tmp, "lens", "Lens")
    home = open_home(home_dir)

    # A standing feeling toward the visitor, deposited before the visit.
    home.ms.appraise(
        "person:shore", sign=1, magnitude=3.0, reason="kept me honest",
        scope="self", owner_id=entity_id, event_id="ev-warm-1",
    )
    seen_prompts = []

    class _SpyLLM(_LLM):
        def generate(self, *, messages, system_prompt):
            seen_prompts.append(system_prompt)
            return super().generate(messages=messages, system_prompt=system_prompt)

    s = ChatSession(
        home, _SpyLLM(["Good to see you."]),
        participants=["person:shore"], context_window=20000, out=lambda s: None,
    )
    # Force the lens visible even if the prelude rendered the same target:
    # the dedup must key on the PRELUDE's actual standing_targets.
    prelude_standing = set((s.prelude or {}).get("standing_targets") or [])
    s.turn("hello again")
    joined = "\n---\n".join(seen_prompts)
    if "person:shore" in prelude_standing:
        assert "FEELINGS (" not in joined, "prelude already carries it (dedup)"
    else:
        assert "FEELINGS (" in joined and "person:shore" in joined

    # The why-walk tool truth: reasons + dates.
    out = s._feelings_about("person:shore")
    assert "person:shore: net +3" in out
    assert 'kept me honest' in out
    home.close()


def test_touches_stamps_value_refs_on_the_event() -> None:
    tmp = Path(tempfile.mkdtemp())
    home_dir, entity_id = _mk_home(tmp, "grounder", "Grounder")
    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM(['Moved.\n```feel\ntarget=person:shore feeling=+2 reason="held the line" touches="honesty"\n```\nk.']),
        participants=["person:shore"], context_window=20000, out=lambda s: None,
    )
    _r, rep = s.turn("something honest happened")
    assert s.feelings_applied == 1, rep.notices
    events = home.ms.journal.valence_events(scope="self", owner_id=entity_id, limit=0)
    ours = [e for e in events if e.target_id == "person:shore"]
    assert ours, "the appraisal landed"
    refs = list(ours[-1].value_refs or ())
    assert refs, "touches= stamped value_refs"
    # Either resolved to the engrammed honesty value record or rode as words.
    assert any("honest" in r.lower() or r.startswith("ex:") for r in refs)
    home.close()
