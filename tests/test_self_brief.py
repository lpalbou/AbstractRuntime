"""THE SELF-KNOWLEDGE BRIEF (identity/self_brief.py; operator 2026-11-10: "we have to help the entities know what it
knows, at least at a very high level. then it can deep dive actively (consciously) into its memory ... careful with
those summaries, so that it doesn't invalidate the prefix (kv cache prompt)").

The night half runs on memory's REAL topics stage (the hash-embedder home of test_topics_night_stage: embed_missing and
communities_pass really form and title the topics; only `sleep_pass` and the summaries model are stand-ins), with real
co-recall events in the journal so memory's `facts.paths` serve trails of thought. The prefix half drives the real
composers: ChatSession (hosted chat and own time), the life factory's re-compose, and the durable visit's OPEN node.
Each test goes red without the piece it names."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_stop_fence import _stop_here  # noqa: E402
from test_topics_night_stage import EID, SCOPES, _asleep, _Capability, _home, _install, _night  # noqa: E402

from abstractruntime.identity.lifecycle_fence import LifecycleFence  # noqa: E402
from abstractruntime.identity.self_brief import (  # noqa: E402
    SELF_BRIEF_FILENAME,
    SELF_BRIEF_HEADING,
    SELF_BRIEF_MAX_TOKENS,
    read_self_brief,
    write_self_brief,
)

pytest.importorskip("numpy")  # memory's pass needs abstractmemory[fast]


def _faithful(prompt: str) -> str:
    """A summaries model that does what the prompt asks: one line per topic (its title and its summary's first
    words), the trails as given, what changed."""
    lines: List[str] = []
    for m in re.finditer(r"^\d+\. (.+?) - \d+ memories, .*?\. (.+)$", prompt, re.M):
        lines.append(f"{m.group(1)}: {' '.join(m.group(2).split()[:6])}")
    trails = [m.group(1) for m in re.finditer(r"^- (.+? \(\d+ times(?: each)?\)), in the topic ", prompt, re.M)]
    if trails:
        lines.append("Trails of thought: " + "; ".join(trails))
    changed = re.search(r"^What changed since the last sleep: (.+)$", prompt, re.M)
    lines.append("Since the last sleep: " + (changed.group(1) if changed else "nothing"))
    return "\n".join(lines)


class _BriefCapability(_Capability):
    """The summaries capability of test_topics_night_stage, which also answers the brief's call (its response model
    has a `brief` field): `brief(prompt)` writes it; `on_brief()` runs as the call is made; `fail` is raised."""

    def __init__(self, *, brief: Callable[[str], str] = _faithful, on_brief: Optional[Callable[[], None]] = None,
                 fail: Optional[BaseException] = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.briefs: List[Dict[str, str]] = []
        self.brief, self.on_brief, self.fail = brief, on_brief, fail

    def generate(self, *, prompt: str, system_prompt: str, response_model: Any) -> Any:
        if "brief" not in response_model.model_fields:
            return super().generate(prompt=prompt, system_prompt=system_prompt, response_model=response_model)
        self.briefs.append({"system": system_prompt, "prompt": prompt})
        if self.on_brief is not None:
            self.on_brief()
        if self.fail is not None:
            raise self.fail
        return response_model(brief=self.brief(prompt))


def _co_recall(home_dir: Path, pairs: List[tuple]) -> List[tuple]:
    """Real co-recall events (memory's `co_selected` journal kind), between memories named by title, n times each:
    what memory's `facts.paths` count as a trail of thought. Returns the pairs as (graph id, graph id, n)."""
    from abstractmemory.journal import MemoryEvent

    from abstractruntime.identity.chat import open_home
    from abstractruntime.identity.memory_reader import HomeMemoryReader

    home = open_home(home_dir, fence=None)
    try:
        reader = HomeMemoryReader(home, ladder=[list(s) for s in SCOPES])
        digests = {str(a.attributes.get("title")): a for a in reader.digest_assertions_all()}
        events = [MemoryEvent(kind="co_selected", scope="life", owner_id=EID,
                              pair_ids=(str(digests[a].assertion_id), str(digests[b].assertion_id)))
                  for a, b, n in pairs for _ in range(n)]
        home.ms.journal.append_events(events)
        return [(str(digests[a].subject), str(digests[b].subject), n) for a, b, n in pairs]
    finally:
        home.close()


def _titles_now(home_dir: Path) -> Dict[str, str]:
    """graph id -> the memory's title after the night (its repair re-authors the test records' titles)."""
    from abstractruntime.identity.chat import open_home
    from abstractruntime.identity.memory_reader import HomeMemoryReader

    home = open_home(home_dir, fence=None)
    try:
        reader = HomeMemoryReader(home, ladder=[list(s) for s in SCOPES])
        return {str(a.subject): str(a.attributes.get("title")) for a in reader.digest_assertions_all()}
    finally:
        home.close()


def _brief_lines(lines: List[str]) -> List[str]:
    return [ln for ln in lines if ln.startswith("(self brief: ")]


def _previous(home_dir: Path) -> bytes:
    write_self_brief(home_dir, "The brief of an earlier sleep.", version_seq=1, tokens=8,
                     fence=LifecycleFence.admit(home_dir))
    return (home_dir / SELF_BRIEF_FILENAME).read_bytes()


# ------------------------------------------------------------------------------------------------- the night


def test_the_night_writes_the_brief_with_the_topics_and_the_top_trails(tmp_path: Path, monkeypatch) -> None:
    """At the end of the topics stage, ONE call of the summaries author (briefed with the entity's name and sentence
    case) over tonight's topics (each title and summary, by size) and trails of thought (memory's facts.paths, the
    older memory first, most often first); the brief lands in <home>/self_brief.md under a header {version_seq,
    written_at, tokens}, at most 400 tokens by the runtime's budget estimator, and never in the memory graph."""
    from abstractruntime.identity.digest import token_estimate

    home_dir = _home(tmp_path)
    pairs = _co_recall(home_dir, [("Checking the dam spillway", "The dam inspection report", 3),
                                  ("Planting the garden sensors", "Watering schedule", 2)])
    order: List[str] = []
    _install(monkeypatch, order)
    _asleep(home_dir)
    cap = _BriefCapability()
    lines: List[str] = []
    result = _night(home_dir, lines, cap)
    receipt = result["communities"]
    assert receipt["version_seq"] is not None and receipt["communities"] >= 1, receipt
    brief = result["self_brief"]
    assert brief["written"] is True and brief["topics"] == receipt["communities"] and brief["trails"] == 2, brief
    assert len(cap.briefs) == 1, "the brief is one call"
    call = cap.briefs[0]
    assert "Liveling is the entity whose memories these are" in call["system"]
    assert "sentence case" in call["system"]
    from abstractmemory import TripleQuery

    from abstractruntime.identity.chat import open_home
    from abstractruntime.identity.topics import entity_communities

    home = open_home(home_dir, fence=None)
    try:  # the served topics (the read the step makes), and every record kind in his graph
        payload = entity_communities(home.ms, scopes=SCOPES, owner_id=EID)
        graph_kinds = {str((a.attributes or {}).get("record_kind")) for scope, owner in SCOPES
                       for a in home.store.query(TripleQuery(scope=scope, owner_id=owner, limit=0))}
    finally:
        home.close()
    titles = [str(c["title"]) for c in payload["communities"]]
    assert len(titles) == receipt["communities"] and all(t.startswith("Authored title") for t in titles), titles
    for title in titles:
        assert title in call["prompt"], (title, call["prompt"])
    now = _titles_now(home_dir)
    trails = [(re.escape(now[a]), re.escape(now[b]), n) for a, b, n in pairs]
    for a, b, n in trails:  # the older memory first: either may be older here (formed in the same instant)
        assert re.search(rf"^- ({a} → {b}|{b} → {a}) \({n} times\), in the topic ", call["prompt"], re.M), \
            call["prompt"]
    assert call["prompt"].index(f"({trails[0][2]} times)") < call["prompt"].index(f"({trails[1][2]} times)")
    stored = read_self_brief(home_dir)
    assert stored is not None and stored["version_seq"] == receipt["version_seq"] == payload["version_seq"]
    assert stored["tokens"] == token_estimate(stored["text"]) == brief["tokens"] <= SELF_BRIEF_MAX_TOKENS
    for title in titles:
        assert title in stored["text"], stored["text"]
    for a, b, n in trails:
        assert re.search(rf"({a} → {b}|{b} → {a}) \({n} times\)", stored["text"]), stored["text"]
    raw = (home_dir / SELF_BRIEF_FILENAME).read_text(encoding="utf-8")
    assert raw.startswith("---\nversion_seq: ") and "\nwritten_at: " in raw and "\ntokens: " in raw, raw
    assert _brief_lines(lines) == [f"(self brief: written - {brief['tokens']} tokens over {len(titles)} topics "
                                   "and 2 trails of thought)"], lines
    assert "community" in graph_kinds and "self_brief" not in graph_kinds, graph_kinds  # a projection, not a memory


def test_a_sleep_whose_topics_did_not_change_keeps_the_brief_and_makes_no_call(tmp_path: Path,
                                                                                monkeypatch) -> None:
    """The prefix invalidates only when the topics change: the second sleep over the same memories (memory writes no
    new version) keeps the brief written for the current version, byte for byte, with no model call."""
    home_dir = _home(tmp_path)
    _install(monkeypatch, [])
    _asleep(home_dir)
    cap = _BriefCapability()
    assert _night(home_dir, [], cap)["self_brief"]["written"] is True
    before = (home_dir / SELF_BRIEF_FILENAME).read_bytes()
    _asleep(home_dir)
    lines: List[str] = []
    second = _night(home_dir, lines, cap)
    assert second["communities"]["version_seq"] is None
    assert second["self_brief"] == {"kept": "the topics did not change since it was written",
                                    "version_seq": read_self_brief(home_dir)["version_seq"]}
    assert len(cap.briefs) == 1, "a model call for unchanged topics"
    assert (home_dir / SELF_BRIEF_FILENAME).read_bytes() == before
    assert _brief_lines(lines) == ["(self brief: kept - the topics did not change since it was written)"]


@pytest.mark.parametrize("case", ["unreachable", "client_error", "over_the_cap", "no_brief"])
def test_a_failed_author_keeps_the_previous_brief_and_writes_a_failure_line(tmp_path: Path, monkeypatch,
                                                                            case: str) -> None:
    """No update on any failure (the previous brief stays, byte for byte) + one authoring-failure line
    (template runtime.self_brief, the reason, what failed) + one night line. A brief over the cap is refused, never
    cut."""
    home_dir = _home(tmp_path)
    _install(monkeypatch, [])
    before = _previous(home_dir)
    _asleep(home_dir)
    cap = {
        "unreachable": lambda: _BriefCapability(fail=TimeoutError("the summaries model did not answer")),
        "client_error": lambda: _BriefCapability(fail=RuntimeError("the endpoint said 500")),
        "over_the_cap": lambda: _BriefCapability(brief=lambda _p: "A long brief. " * 200),
        "no_brief": lambda: _BriefCapability(brief=lambda _p: "   "),
    }[case]()
    lines: List[str] = []
    result = _night(home_dir, lines, cap)
    assert result["communities"]["version_seq"] is not None
    reason = {"unreachable": "unavailable", "client_error": "client_error", "over_the_cap": "malformed",
              "no_brief": "malformed"}[case]
    assert result["self_brief"]["failed"] == reason and result["self_brief"]["kept_previous"] is True, result
    assert (home_dir / SELF_BRIEF_FILENAME).read_bytes() == before, "a failed brief replaced the previous one"
    failures = [ln for ln in lines if ln.startswith("#AUTHORING_FAILURE") and '"runtime.self_brief"' in ln]
    assert len(failures) == 1, lines
    assert f'"reason": "{reason}"' in failures[0] and '"kind": "self_brief"' in failures[0]
    if case == "over_the_cap":
        assert f"over the {SELF_BRIEF_MAX_TOKENS}-token cap" in failures[0]
        assert "A long brief." in failures[0], "the refused reply is kept on the failure line"
    (line,) = _brief_lines(lines)
    assert line.startswith(f"(self brief: not rewritten - {reason} (") and line.endswith("the previous brief stays)")


@pytest.mark.parametrize("where", ["during_the_call", "at_the_write"])
def test_a_stop_during_the_brief_writes_nothing(tmp_path: Path, monkeypatch, where: str) -> None:
    """The operator's Stop while the brief's author call runs (the reply returns after it: abandoned) or between the
    reply and the write (the write section's fence check refuses it): the previous brief stays byte for byte, no
    failure line, no brief line, and the night ends with its one Stop line."""
    import abstractruntime.identity.self_brief as sb

    home_dir = _home(tmp_path)
    _install(monkeypatch, [])
    before = _previous(home_dir)
    _asleep(home_dir)
    if where == "during_the_call":
        cap = _BriefCapability(on_brief=lambda: _stop_here(home_dir))
    else:
        cap = _BriefCapability()
        real = sb.write_self_brief

        def stopped_then_written(*a, **k):
            _stop_here(home_dir)
            return real(*a, **k)

        monkeypatch.setattr(sb, "write_self_brief", stopped_then_written)
    lines: List[str] = []
    result = _night(home_dir, lines, cap, narrator_llm_factory=lambda: (_ for _ in ()).throw(AssertionError("voice")))
    assert len(cap.briefs) == 1
    assert result["self_brief"]["stopped"].startswith("the operator's Stop"), result["self_brief"]
    assert result["stopped_by"] == "kill_switch"
    assert (home_dir / SELF_BRIEF_FILENAME).read_bytes() == before, "the brief was written after the Stop"
    assert not _brief_lines(lines), lines
    assert lines.count("(the operator's Stop - the night ends here; nothing more is written)") == 1, lines
    assert not [ln for ln in lines if "#AUTHORING_FAILURE" in ln or "#FALLBACK" in ln], lines


def test_a_home_without_topics_gets_no_brief(tmp_path: Path, monkeypatch) -> None:
    """Nothing to brief (memory formed no topics: its receipt for an empty pass): no call, no file, nothing said."""
    import abstractmemory.communities as mc

    home_dir = _home(tmp_path)
    _install(monkeypatch, [])

    def empty_pass(system, **kw):
        receipt = mc._receipt()
        receipt["skipped"] = "empty: no memory to group"
        return receipt

    monkeypatch.setattr(mc, "communities_pass", empty_pass)
    _asleep(home_dir)
    cap = _BriefCapability()
    lines: List[str] = []
    result = _night(home_dir, lines, cap)
    assert result["self_brief"] == {"skipped": "no topics yet"}, result["self_brief"]
    assert not cap.briefs and not (home_dir / SELF_BRIEF_FILENAME).exists() and not _brief_lines(lines)


# ------------------------------------------------------------------------------------------------- the prefix

BRIEF = "Dams and gauges: how the spillway is checked.\nTrails of thought: Checking the dam → The report (3 times)."


def test_compose_places_the_brief_after_the_identity_block_and_none_without_one() -> None:
    from abstractruntime.identity.chat import CONTRACT_PARAGRAPH, compose_system_base

    prelude = "<identity_prelude>You are Castor.</identity_prelude>"
    with_brief = compose_system_base(prelude, phase="visit", overlay={"operator": "OP"}, self_brief=BRIEF)
    assert with_brief.startswith(prelude + "\n\n" + SELF_BRIEF_HEADING + "\n" + BRIEF + "\n\n" + CONTRACT_PARAGRAPH)
    assert with_brief.endswith("STANDING INSTRUCTIONS FROM YOUR OPERATOR:\nOP")  # operator-last holds
    for none in (None, "", "  \n"):
        base = compose_system_base(prelude, phase="visit", overlay={"operator": "OP"}, self_brief=none)
        assert SELF_BRIEF_HEADING not in base
        assert base == compose_system_base(prelude, phase="visit", overlay={"operator": "OP"})


def _castor(tmp_path: Path) -> Path:
    from test_prompt_overlay import ENTITY_SLUG, _create_home

    home_dir = tmp_path / "entities" / ENTITY_SLUG
    _create_home(home_dir)
    return home_dir


def test_the_chat_prefix_holds_the_brief_byte_identical_across_two_turns(tmp_path: Path) -> None:
    """Hosted chat (ChatSession): the section sits after the identity block and before MEMORIES, and the prefix the
    provider caches - everything up to the per-turn regions - is byte-identical on two consecutive turns without a
    sleep."""
    from test_prompt_overlay import _FakeResponse

    from abstractruntime.identity.chat import ChatSession, open_home

    home_dir = _castor(tmp_path)
    write_self_brief(home_dir, BRIEF, version_seq=7, tokens=30, fence=LifecycleFence.admit(home_dir))

    class _LLM:
        def __init__(self) -> None:
            self.systems: List[str] = []

        def generate(self, *, messages, system_prompt):
            self.systems.append(system_prompt)
            return _FakeResponse("The dam spillway gauges drift in winter; I checked them with you.")

    llm = _LLM()
    session = ChatSession(open_home(home_dir, fence=None), llm, participants=["person:albou"], session_id="s1",
                          context_window=32768, out=lambda s: None, fence=None)
    try:
        session.turn("Tell me about the dam spillway gauges and the winter.")
        session.turn("And the dam spillway gauges again - what drifts in winter?")
    finally:
        session.home.close()
    first, second = session.reports[0].system_prompt, session.reports[1].system_prompt
    section = SELF_BRIEF_HEADING + "\n" + BRIEF
    for prompt in (first, second):
        assert prompt.index("</identity_prelude>") < prompt.index(section) < prompt.index("The block above is your")
    assert "MEMORIES (" in second, "the second turn recalls the first (the test needs a per-turn region)"
    assert second.index(section) < second.index("MEMORIES (")
    base = session.system_base
    assert section in base and first.startswith(base) and second.startswith(base)
    assert first[:len(base)] == second[:len(base)], "the cached prefix moved between two turns"
    assert llm.systems and all(s.startswith(base) for s in llm.systems)


def test_no_section_on_a_home_without_a_brief_and_an_unreadable_one_is_said(tmp_path: Path) -> None:
    from test_prompt_overlay import _session

    home_dir = _castor(tmp_path)
    session = _session(home_dir)
    try:
        assert SELF_BRIEF_HEADING not in session.system_base and session.self_brief == ""
    finally:
        session.home.close()
    (home_dir / SELF_BRIEF_FILENAME).write_text("no header here\n", encoding="utf-8")
    notes: List[str] = []
    session = _session(home_dir, out=notes.append)
    try:
        assert SELF_BRIEF_HEADING not in session.system_base
        assert any(n.startswith("#FALLBACK the self brief is unreadable") for n in notes), notes
    finally:
        session.home.close()


def test_the_durable_visit_and_own_time_heads_carry_the_brief(tmp_path: Path, monkeypatch) -> None:
    """The visit workflow's OPEN (the head lives in the run state: every turn and resume shares it) and the life
    factory's own-time re-compose put the same section in the same place."""
    import abstractcore

    from test_prompt_overlay import _FakeResponse

    from abstractruntime.core.models import EffectType
    from abstractruntime.identity.chat import CONTRACT_PARAGRAPH
    from abstractruntime.identity.entity_runtime import open_entity_runtime
    from abstractruntime.identity.life import build_session_factory
    from abstractruntime.identity.visit_workflow import build_visit_workflow

    home_dir = _castor(tmp_path)
    write_self_brief(home_dir, BRIEF, version_seq=7, tokens=30, fence=LifecycleFence.admit(home_dir))
    section = SELF_BRIEF_HEADING + "\n" + BRIEF

    class _NeverCalledLLM:
        def __call__(self, run, effect, dnn=None):  # pragma: no cover
            raise AssertionError("OPEN must not reach the LLM")

    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: _NeverCalledLLM()},
                              fence=LifecycleFence.admit(home_dir))
    try:
        wf = build_visit_workflow(ert.home, participants=["person:albou"], idle_seconds=3600)
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-brief")
        state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=10)
        head = str(state.vars["_visit"]["system_base"])
    finally:
        ert.close()
    assert head.index("</identity_prelude>") < head.index(section) < head.index(CONTRACT_PARAGRAPH)

    class _LLMStub:
        def generate(self, **kwargs):
            return _FakeResponse("ok")

    monkeypatch.setattr(abstractcore, "create_llm", lambda *a, **k: _LLMStub())
    factory = build_session_factory(home_dir, provider="lmstudio", model="test", base_url="http://x",
                                    embedding_model=None, embedding_base_url="http://x", context_window=32768,
                                    out=lambda s: None, fence=LifecycleFence.admit(home_dir))
    session = factory()
    try:
        base = session.system_base
        assert base.index("</identity_prelude>") < base.index(section) < base.index(CONTRACT_PARAGRAPH)
    finally:
        session.home.close()


def test_the_topics_tool_teaches_the_brief() -> None:
    from abstractruntime.identity import tools as T

    assert "(your brief lists them)" in T.TOOL_DESCRIPTORS["topics"].description
    flat = " ".join(T.TOOLS_CONTRACT_PARAGRAPH.split())
    assert "Your at-a-glance brief (\"What I know about, at a glance\", written in your sleep) lists your topics and " \
           "trails of thought; use topics to dig in." in flat
