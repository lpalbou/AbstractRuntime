"""The identity-pass spine, runtime slice (cti#399 design, adopted cti#400;
build gate ruled ALREADY SATISFIED by framework c4779).

Waking elects, sleep enacts, the registrar never authors:
- ```realize blocks parse into PROPOSALS (evidence= mandatory, refusals
  loud, titled held-for-sleep markers) — never an identity write;
- the apply path forms kind=realization into SELF scope with derived_from
  edges to the RESOLVED evidence, and degrades loudly on engine skew
  (memory's kind-vocabulary half lands in the same wave);
- include_identity threads into the sleep composition exactly like
  include_dream: cycle windows NEVER, nightly sleeps OFFER, TypeError
  skew ladder labeled — the graph-is-the-queue holds proposals across
  engine upgrades, nothing is lost.
"""

from typing import Any, Dict, List

from abstractruntime.identity.chat import ChatSession
from abstractruntime.identity.life import LifeLoop
from abstractruntime.identity.reflection import (
    MAX_REALIZATIONS_PER_SESSION,
    parse_realize_blocks,
)


class TestParseRealizeBlocks:
    def test_valid_block_parses_and_marks_held_for_sleep(self):
        reply = (
            "I see it now.\n"
            "```realize\n"
            "I trust slow answers more than fast ones\n"
            "evidence=#tolstoy #slow-reading\n"
            "touches=value:patience\n"
            "```\n"
            "That is what the week showed."
        )
        marked, realizations, notices = parse_realize_blocks(reply)
        assert len(realizations) == 1
        r = realizations[0]
        assert r.text == "I trust slow answers more than fast ones"
        assert r.evidence == ["#tolstoy", "#slow-reading"]
        assert r.touches == "value:patience"
        assert "held for sleep" in marked
        assert "```realize" not in marked
        assert notices == []

    def test_missing_evidence_is_refused_with_teaching_marker(self):
        reply = "```realize\nI am more careful than I thought\n```"
        marked, realizations, notices = parse_realize_blocks(reply)
        assert realizations == []
        assert "add evidence=" in marked
        assert any("evidence= is mandatory" in n for n in notices)

    def test_empty_body_is_skipped(self):
        reply = "```realize\nevidence=#a\n```"
        marked, realizations, notices = parse_realize_blocks(reply)
        assert realizations == []
        assert any("no words" in n for n in notices)

    def test_prose_evidence_tokens_drop_with_one_collective_notice(self):
        """Adversary F4: 'evidence=#a is what my diary said' must not explode
        into per-word full-home scans — prose words drop, one notice."""
        reply = "```realize\nslowness is a value\nevidence=#a is what my diary said\n```"
        marked, realizations, notices = parse_realize_blocks(reply)
        assert len(realizations) == 1
        assert realizations[0].evidence == ["#a"]
        assert sum("dropped" in n for n in notices) == 1

    def test_evidence_token_cap(self):
        toks = " ".join(f"#t{i}" for i in range(12))
        reply = f"```realize\ntoo many\nevidence={toks}\n```"
        _marked, realizations, notices = parse_realize_blocks(reply)
        from abstractruntime.identity.reflection import MAX_EVIDENCE_TOKENS

        assert len(realizations[0].evidence) == MAX_EVIDENCE_TOKENS
        assert any("dropped" in n for n in notices)

    def test_session_cap_refuses_loudly(self):
        block = "```realize\nrealization {i}\nevidence=#e{i}\n```"
        reply = "\n".join(block.format(i=i) for i in range(MAX_REALIZATIONS_PER_SESSION + 1))
        marked, realizations, notices = parse_realize_blocks(reply)
        assert len(realizations) == MAX_REALIZATIONS_PER_SESSION
        assert any("cap" in n for n in notices)
        assert "refused" in marked


class _Reader:
    """Reader stub: knows two tags + one real graph id; everything else
    unresolved (digest_assertion None = does not exist in the home ladder)."""

    def find_tag_in_home(self, bare: str):
        known = {"tolstoy": "ex:tolstoy-1", "slow-reading": "ex:slow-2"}
        return (known.get(bare, ""), None)

    def digest_assertion(self, graph_id: str):
        return object() if graph_id in ("ex:tolstoy-1", "ex:slow-2") else None


class _Home:
    entity_id = "entity:test"


class _ApplyStub:
    """Duck-typed ChatSession carrying exactly what the apply path touches."""

    _resolve_evidence_token = ChatSession._resolve_evidence_token
    _apply_realizations = ChatSession._apply_realizations

    def __init__(self, effect_result=None, effect_raises=None):
        self._memory_reader = _Reader()
        self.home = _Home()
        self.session_id = "sess-1"
        self.phase = "personal"
        self.lines: List[str] = []
        self.effect_calls: List[Any] = []
        self._effect_result = effect_result or {"record_ids": ["ex:real-1"], "warnings": []}
        self._effect_raises = effect_raises

    def out(self, line: str) -> None:
        self.lines.append(line)

    def _effect(self, effect_type, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.effect_calls.append((effect_type, payload))
        if self._effect_raises is not None:
            raise self._effect_raises
        return dict(self._effect_result)


def _realization(text="I trust slow answers", evidence=None, touches="value:patience"):
    from abstractruntime.identity.reflection import RealizeElection

    return RealizeElection(
        text=text, evidence=list(evidence or ["#tolstoy", "#slow-reading"]), touches=touches
    )


class TestApplyRealizations:
    def test_forms_inert_proposal_with_derived_from_edges(self):
        stub = _ApplyStub()
        notices: List[str] = []
        stub._apply_realizations([_realization()], turn_id="t1", notices=notices)

        assert len(stub.effect_calls) == 1
        _etype, payload = stub.effect_calls[0]
        rec = payload["records"][0]
        assert rec["kind"] == "realization"
        assert payload["scope"] == "self"
        assert rec["edges"] == [["derived_from", "ex:tolstoy-1"], ["derived_from", "ex:slow-2"]]
        assert rec["attributes"]["touches"] == "value:patience"
        assert rec["provenance"]["actor"] == "entity-reflection"
        assert payload["turn_id"] == "t1-realize-0"
        assert any("held for sleep" in ln for ln in stub.lines)

    def test_unresolved_evidence_refuses_formation(self):
        stub = _ApplyStub()
        notices: List[str] = []
        stub._apply_realizations(
            [_realization(evidence=["#unknown-tag"])], turn_id="t1", notices=notices
        )
        assert stub.effect_calls == []
        assert any("no evidence resolved" in n for n in notices)
        assert any("unresolved" in n for n in notices)

    def test_engine_skew_degrades_loudly_without_killing_the_turn(self):
        stub = _ApplyStub(effect_raises=RuntimeError("Unknown record kind 'realization'"))
        notices: List[str] = []
        stub._apply_realizations([_realization()], turn_id="t1", notices=notices)
        assert any("not yet held durably" in n for n in notices)
        assert any("memory's build" in n for n in notices)

    def test_fabricated_graph_ids_do_not_resolve(self):
        """Adversary F1: colon-shaped tokens must EXIST in the home ladder —
        a fabricated ex:fake-999 (or a stray value:honesty on the evidence
        line) must never mint a derived_from edge to nothing."""
        stub = _ApplyStub()
        notices: List[str] = []
        stub._apply_realizations(
            [_realization(evidence=["ex:fake-999", "value:honesty"])],
            turn_id="t1", notices=notices,
        )
        assert stub.effect_calls == []
        assert any("no evidence resolved" in n for n in notices)

    def test_existing_graph_id_passes_validation(self):
        stub = _ApplyStub()
        notices: List[str] = []
        stub._apply_realizations(
            [_realization(evidence=["ex:tolstoy-1"])], turn_id="t1", notices=notices
        )
        assert len(stub.effect_calls) == 1
        rec = stub.effect_calls[0][1]["records"][0]
        assert rec["edges"] == [["derived_from", "ex:tolstoy-1"]]

    def test_salvage_stamps_the_ended_sessions_identity(self):
        """Adversary F3 (r-rt-3): a salvaged look-back stamps the ENDED
        session's id/phase, never the salvaging session's."""
        stub = _ApplyStub()
        notices: List[str] = []
        stub._apply_realizations(
            [_realization()], turn_id="t1", notices=notices,
            session_id="ended-sess", phase="personal-ended",
        )
        attrs = stub.effect_calls[0][1]["records"][0]["attributes"]
        assert attrs["session_id"] == "ended-sess"
        assert attrs["phase"] == "personal-ended"


class _LoopStub:
    """Duck-typed LifeLoop for the sleep-window composition ladder."""

    _sleep_window = LifeLoop._sleep_window
    state_home = None

    def __init__(self, hook):
        self.on_sleep = hook
        self.lines: List[str] = []

    def out(self, line: str) -> None:
        self.lines.append(line)

    def _clear_dreaming_badge(self, reason: str) -> None:
        pass


class TestIncludeIdentityThreading:
    def test_nightly_default_offers_identity(self):
        seen: Dict[str, Any] = {}

        def hook(*, include_dream=True, include_identity=True):
            seen.update(include_dream=include_dream, include_identity=include_identity)
            return {"formed": False}

        loop = _LoopStub(hook)
        loop._sleep_window("night rest")
        assert seen == {"include_dream": True, "include_identity": True}

    def test_cycle_window_never_touches_the_self(self):
        seen: Dict[str, Any] = {}

        def hook(*, include_dream=True, include_identity=True):
            seen.update(include_dream=include_dream, include_identity=include_identity)
            return {"formed": False}

        loop = _LoopStub(hook)
        loop._sleep_window("cycle maintenance", include_dream=False, include_identity=False)
        assert seen == {"include_dream": False, "include_identity": False}

    def test_skew_ladder_falls_back_and_labels(self):
        seen: Dict[str, Any] = {}

        def older_hook(*, include_dream=True):  # no include_identity: pre-spine engine
            seen.update(include_dream=include_dream)
            return {"formed": False}

        loop = _LoopStub(older_hook)
        loop._sleep_window("night rest")
        assert seen == {"include_dream": True}
        assert any("no include_identity" in ln for ln in loop.lines)

    def test_oldest_hook_gets_the_plain_call_with_identity_label(self):
        """Adversary F5: the plain rung loses both flags — the identity loss
        must be labeled there too, never silent."""
        calls: List[str] = []

        def oldest_hook():
            calls.append("ran")
            return None

        loop = _LoopStub(oldest_hook)
        loop._sleep_window("night rest")
        assert calls == ["ran"]
        assert any("takes no flags" in ln for ln in loop.lines)
