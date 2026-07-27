"""Obedience tests for the effective phase graph (build order c4837).

The order's four pins, verbatim: an added edge fires with its instruction in
the cue; a removed edge never writes; a redirect lands the substituted
target; constitutional edges are immune EVEN ON A HAND-EDITED FILE (the
loop must not obey what the door should have refused — detached loops read
the file, so the gateway validator alone cannot be the wall).

Plus the interpreter's own floors: the sleep floor never wedges (totality),
unknown causes refuse naming the legal list, and the phase_changed marker
(spelled with this build after living 'reserved-unspelled' since the v12
marker contract) lands machine-readably without diluting life_sleep_stats.
"""

import json
from pathlib import Path

from abstractruntime.identity.life import (
    append_phase_changed,
    consult_gate_landing,
    life_sleep_stats,
    read_day_gate,
)
from abstractruntime.identity.phase_graph import (
    Landing,
    instruction_cue,
    load_effective_graph,
    vendored_spec_path,
)


def _spec_with_ops(edge_ops, edit_seq=8):
    base = json.loads(vendored_spec_path().read_text(encoding="utf-8"))
    base["graph"] = {"edge_ops": edge_ops}
    base["_operator"] = {"edit_seq": edit_seq}
    return base


def _write_spec(tmp_path: Path, doc) -> Path:
    p = tmp_path / "entity_phases.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _arm_home(tmp_path: Path, *, grant=True, work_order=False) -> Path:
    home = tmp_path / "entities" / "testee"
    home.mkdir(parents=True, exist_ok=True)
    if grant:
        (home / "phases.yaml").write_text(
            "personal:\n  mode: until_revoked\n", encoding="utf-8"
        )
    if work_order:
        (home / "work_order.md").write_text("build the thing\n", encoding="utf-8")
    return home


def _operator_copy(home: Path, doc) -> None:
    cfg = home.parent.parent / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "entity_phases.json").write_text(json.dumps(doc), encoding="utf-8")


class TestInterpreter:
    def test_added_edge_carries_instruction_and_provenance(self, tmp_path):
        doc = _spec_with_ops([
            {"op": "add", "from": "sleep", "to": "personal", "cause": "cadence_need_check",
             "instruction": "ease back in - review your open questions first"},
        ])
        graph, warnings = load_effective_graph(spec_path=_write_spec(tmp_path, doc))
        landing = graph.legal_to("sleep", "personal", "cadence_need_check")
        assert landing is not None
        assert landing.instruction.startswith("ease back in")
        assert landing.provenance == "blueprint edit #8"
        cue = instruction_cue(landing)
        assert "standing instruction" in cue and "blueprint edit #8" in cue

    def test_removed_edge_is_gone(self, tmp_path):
        doc = _spec_with_ops([
            {"op": "remove", "edge": "sleep->work#cadence_need_check"},
        ])
        graph, _ = load_effective_graph(spec_path=_write_spec(tmp_path, doc))
        assert graph.legal_to("sleep", "work", "cadence_need_check") is None
        # The sibling landing is untouched.
        assert graph.legal_to("sleep", "personal", "cadence_need_check") is not None

    def test_redirect_substitutes_target(self, tmp_path):
        doc = _spec_with_ops([
            {"op": "redirect", "edge": "sleep->work#cadence_need_check", "to": "personal",
             "instruction": "the desk is clear - take the evening"},
        ])
        graph, _ = load_effective_graph(spec_path=_write_spec(tmp_path, doc))
        landing = graph.legal_to("sleep", "work", "cadence_need_check")
        assert landing is not None
        assert landing.to == "personal"
        assert "take the evening" in landing.instruction

    def test_reserved_cause_ops_refuse_until_status_flips(self, tmp_path):
        """The artifact's own engraved rule (v19+): RESERVED blocks
        new/redirected edges until the cause's status flips to shipped.
        Pinned against a SELF-CONTAINED doc so artifact-version churn never
        rewrites this pin (v20 flipped task_complete/no_task to shipped on
        runtime's evaluator receipt — the flip this rule exists to gate)."""
        doc = {
            "phases": {"work": {}, "personal": {}, "sleep": {}, "visit": {}},
            "transition_causes": ["day_end", "cadence_need_check"],
            "cause_evaluators": {"day_end": {"status": "reserved"}},
            "transitions": [
                {"from": "work", "to": "sleep", "cause": "day_end"},
                {"from": "sleep", "to": "work", "cause": "cadence_need_check"},
                {"from": "personal", "to": "sleep", "cause": "day_end"},
            ],
            "graph": {"edge_ops": [
                {"op": "redirect", "edge": "work->sleep#day_end", "to": "personal"},
                {"op": "add", "from": "sleep", "to": "personal", "cause": "day_end",
                 "instruction": "note"},
            ]},
            "_operator": {"edit_seq": 5},
        }
        graph, warnings = load_effective_graph(spec_path=_write_spec(tmp_path, doc))
        landing = graph.legal_to("work", "sleep", "day_end")
        assert landing is not None and landing.to == "sleep"  # redirect refused
        assert graph.legal_to("sleep", "personal", "day_end") is None  # add refused
        assert sum("RESERVED" in w for w in warnings) == 2

    def test_redirect_to_unknown_phase_refused(self, tmp_path):
        doc = _spec_with_ops([
            {"op": "redirect", "edge": "sleep->work#cadence_need_check", "to": "mars"},
        ])
        graph, warnings = load_effective_graph(spec_path=_write_spec(tmp_path, doc))
        landing = graph.legal_to("sleep", "work", "cadence_need_check")
        assert landing is not None and landing.to == "work"  # untouched
        assert any("unknown phase" in w for w in warnings)

    def test_add_onto_dial_edge_refused(self, tmp_path):
        """v19 policy=dial (personal->sleep#personal_cycle): the dial governs;
        instruction/bound attachment through an add op refuses (adversary-2
        F3: the first cut let prose ride a dial edge into the wake cue)."""
        doc = _spec_with_ops([
            {"op": "add", "from": "personal", "to": "sleep", "cause": "personal_cycle",
             "instruction": "INJECTED ONTO DIAL EDGE", "bound_h": 5.0},
        ])
        graph, warnings = load_effective_graph(spec_path=_write_spec(tmp_path, doc))
        landing = graph.legal_to("personal", "sleep", "personal_cycle")
        assert landing is not None
        assert landing.instruction == ""  # nothing attached
        assert landing.bound_h is None
        assert any("protected edge" in w for w in warnings)

    def test_constitutional_edges_immune_on_hand_edited_file(self, tmp_path):
        """The order's hard line: a hand-edited file carrying ops the door
        should have refused is DISOBEYED at the interpreter, loudly."""
        doc = _spec_with_ops([
            {"op": "remove", "edge": "sleep->visit#visit_open"},
            {"op": "remove", "edge": "personal->sleep#self_elected"},
            {"op": "remove", "edge": "personal->sleep#grant_expired"},
            {"op": "redirect", "edge": "sleep->personal#operator", "to": "work"},
        ])
        graph, warnings = load_effective_graph(spec_path=_write_spec(tmp_path, doc))
        assert graph.legal_to("sleep", "visit", "visit_open") is not None
        assert graph.legal_to("personal", "sleep", "self_elected") is not None
        assert graph.legal_to("personal", "sleep", "grant_expired") is not None
        op_landing = graph.legal_to("sleep", "personal", "operator")
        assert op_landing is not None and op_landing.to == "personal"
        assert sum("protected" in w for w in warnings) == 4

    def test_unknown_cause_refused_naming_legal_list(self, tmp_path):
        doc = _spec_with_ops([
            {"op": "add", "from": "personal", "to": "work", "cause": "commitment_due"},
        ])
        graph, warnings = load_effective_graph(spec_path=_write_spec(tmp_path, doc))
        assert graph.legal_to("personal", "work", "commitment_due") is None
        assert any("unknown cause" in w and "legal causes" in w for w in warnings)

    def test_add_landing_in_visit_refused(self, tmp_path):
        doc = _spec_with_ops([
            {"op": "add", "from": "sleep", "to": "visit", "cause": "personal_cycle"},
        ])
        _graph, warnings = load_effective_graph(spec_path=_write_spec(tmp_path, doc))
        assert any("evidence, never elective landings" in w for w in warnings)

    def test_removing_direct_sleep_edges_is_legal_when_sleep_stays_reachable(self, tmp_path):
        """Adversary-2 F2: the floor is REACHABILITY-aware — removing both
        work->sleep edges is a legal edit because work->personal->sleep
        remains (locked personal->sleep edges guarantee it); the floor must
        NOT resurrect door-legal removals."""
        doc = _spec_with_ops([
            {"op": "remove", "edge": "work->sleep#task_complete"},
            {"op": "remove", "edge": "work->sleep#no_task"},
        ])
        graph, warnings = load_effective_graph(spec_path=_write_spec(tmp_path, doc))
        assert graph.legal_to("work", "sleep", "task_complete") is None
        assert graph.legal_to("work", "sleep", "no_task") is None
        assert not any("sleep floor" in w for w in warnings)

    def test_sleep_floor_restores_on_true_totality_break(self, tmp_path):
        """The floor's genuine trigger: a hand-crafted structural doc whose
        only sleep path is removed — restoration UN-REMOVES the structural
        sleep edge (never resets a live redirect's target)."""
        doc = {
            "phases": {"work": {}, "personal": {}, "sleep": {}, "visit": {}},
            "transition_causes": ["day_end", "cadence_need_check"],
            "transitions": [
                {"from": "work", "to": "sleep", "cause": "day_end"},
                {"from": "sleep", "to": "work", "cause": "cadence_need_check"},
            ],
            "graph": {"edge_ops": [
                {"op": "remove", "edge": "work->sleep#day_end"},
            ]},
            "_operator": {"edit_seq": 3},
        }
        graph, warnings = load_effective_graph(spec_path=_write_spec(tmp_path, doc))
        assert graph.legal_to("work", "sleep", "day_end") is not None
        assert any("sleep floor restored" in w for w in warnings)

    def test_floor_never_resets_a_live_redirect(self, tmp_path):
        """Redirect work->sleep => personal while personal->sleep stands:
        totality holds THROUGH the redirect; the floor must not undo it."""
        doc = {
            "phases": {"work": {}, "personal": {}, "sleep": {}, "visit": {}},
            "transition_causes": ["day_end", "grant_expired", "cadence_need_check"],
            "transitions": [
                {"from": "work", "to": "sleep", "cause": "day_end"},
                {"from": "personal", "to": "sleep", "cause": "grant_expired"},
                {"from": "sleep", "to": "work", "cause": "cadence_need_check"},
            ],
            "graph": {"edge_ops": [
                {"op": "redirect", "edge": "work->sleep#day_end", "to": "personal"},
            ]},
            "_operator": {"edit_seq": 4},
        }
        graph, warnings = load_effective_graph(spec_path=_write_spec(tmp_path, doc))
        landing = graph.legal_to("work", "sleep", "day_end")
        assert landing is not None and landing.to == "personal"  # redirect ALIVE
        assert not any("sleep floor" in w for w in warnings)

    def test_sub_tick_bound_refused(self, tmp_path):
        doc = _spec_with_ops([
            {"op": "add", "from": "sleep", "to": "personal", "cause": "cadence_need_check",
             "bound_h": 0.001},
        ])
        graph, warnings = load_effective_graph(spec_path=_write_spec(tmp_path, doc))
        landing = graph.legal_to("sleep", "personal", "cadence_need_check")
        assert landing is not None and landing.bound_h is None
        assert any("finer than a tick" in w for w in warnings)


class TestConsultGateLanding:
    def test_removed_work_leg_skips_to_personal(self, tmp_path):
        """A removed sleep->work#cadence_need_check means a standing order no
        longer sanctions a work day at the need-check — the chain falls to
        personal (grant armed)."""
        home = _arm_home(tmp_path, grant=True, work_order=True)
        _operator_copy(home, _spec_with_ops([
            {"op": "remove", "edge": "sleep->work#cadence_need_check"},
        ]))
        lines = []
        decision = read_day_gate(home)
        assert decision["phase"] == "work"  # the gate still says work
        decision, cue = consult_gate_landing(
            home, "sleep", decision, "cadence_need_check", lines.append
        )
        assert decision["phase"] == "personal"
        assert any("skipped" in ln for ln in lines)

    def test_redirect_lands_substituted_target_with_guard(self, tmp_path):
        home = _arm_home(tmp_path, grant=True, work_order=True)
        _operator_copy(home, _spec_with_ops([
            {"op": "redirect", "edge": "sleep->work#cadence_need_check", "to": "personal",
             "instruction": "start gently"},
        ]))
        lines = []
        decision = read_day_gate(home)
        decision, cue = consult_gate_landing(
            home, "sleep", decision, "cadence_need_check", lines.append
        )
        assert decision["phase"] == "personal"
        assert decision.get("redirected_from") == "work"
        assert "start gently" in cue and "standing instruction" in cue

    def test_redirect_to_personal_without_grant_is_refused(self, tmp_path):
        home = _arm_home(tmp_path, grant=False, work_order=True)
        _operator_copy(home, _spec_with_ops([
            {"op": "redirect", "edge": "sleep->work#cadence_need_check", "to": "personal"},
        ]))
        lines = []
        decision = read_day_gate(home)
        decision, _cue = consult_gate_landing(
            home, "sleep", decision, "cadence_need_check", lines.append
        )
        # Guard travels with the arrow: no grant -> the redirect refuses and
        # the chain falls through (no grant -> sleep).
        assert decision["phase"] == "sleep"
        assert any("guard travels" in ln for ln in lines)

    def test_untouched_graph_passes_decision_through(self, tmp_path):
        home = _arm_home(tmp_path, grant=True, work_order=True)
        lines = []
        decision = read_day_gate(home)
        out_decision, cue = consult_gate_landing(
            home, "sleep", decision, "cadence_need_check", lines.append
        )
        assert out_decision == decision
        assert cue == ""


class TestPhaseChangedMarker:
    def test_marker_lands_and_stats_exclude_it(self, tmp_path):
        home = _arm_home(tmp_path)
        # Two state transitions + one phase marker.
        (home / "state_history.jsonl").write_text(
            json.dumps({"state": "asleep", "written_by": "self"}) + "\n"
            + json.dumps({"state": "awake", "written_by": "self"}) + "\n",
            encoding="utf-8",
        )
        append_phase_changed(
            home, from_phase="sleep", to="personal",
            cause="cadence_need_check", written_by="need-check",
            provenance="blueprint edit #8",
        )
        rows = [
            json.loads(ln)
            for ln in (home / "state_history.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        markers = [r for r in rows if r.get("marker") == "phase_changed"]
        assert len(markers) == 1
        m = markers[0]
        assert m["from"] == "sleep" and m["to"] == "personal"
        assert m["cause"] == "cadence_need_check"
        assert m["provenance"] == "blueprint edit #8"
        assert m["at"]
        stats = life_sleep_stats(home)
        # The marker row never dilutes the state-transition denominator.
        assert stats["transitions"] == 2
        assert stats["sleeps"] == 1 and stats["wakes"] == 1


class TestMarkerSingleConstruction:
    def test_marker_dict_constructed_exactly_once_in_life(self):
        """Adversary-2 F8: the amended module pins exempt life.py wholesale,
        so a second raw writer INSIDE life.py would be invisible — this pin
        asserts the marker dict construction appears exactly once (the
        append_phase_changed helper)."""
        import abstractruntime.identity.life as life_mod
        from pathlib import Path

        src = Path(life_mod.__file__).read_text(encoding="utf-8")
        assert src.count('"marker": "phase_changed"') == 1


class TestFullLoopObedience:
    def test_removed_need_check_work_edge_prevents_the_work_day(self, tmp_path, monkeypatch):
        """The adversary's P0, pinned end-to-end: a removed
        sleep->work#cadence_need_check must govern the day that OPENS, not
        just the wake's words — the wake carries its cause to the top
        boundary, whose consult re-derives the skip, and the day opens
        PERSONAL despite the standing work order."""
        import sys
        from datetime import datetime, timedelta, timezone

        sys.path.insert(0, str(Path(__file__).parent))
        from test_entity_life_loop import _ScriptedLLM, _factory_for, _make_home

        from abstractruntime.identity.life import LifeLoop, write_entity_state
        from abstractruntime.identity.phase_spec import PHASE_SPEC_ENV

        home_dir = _make_home(tmp_path)
        (home_dir / "phases.yaml").write_text(
            "personal:\n  mode: until_revoked\n", encoding="utf-8"
        )
        (home_dir / "work_order.md").write_text("build the thing\n", encoding="utf-8")
        # The operator's edit: need-checks never sanction work days.
        spec = {
            "tunables": {},
            "graph": {"edge_ops": [
                {"op": "remove", "edge": "sleep->work#cadence_need_check"},
            ]},
            "_operator": {"edit_seq": 9},
        }
        p = tmp_path / "spec.json"
        p.write_text(json.dumps(spec), encoding="utf-8")
        monkeypatch.setenv(PHASE_SPEC_ENV, str(p))
        # Asleep on a gate-written landing whose deadline is already past:
        # the need-check fires on the loop's first idle poll.
        write_entity_state(
            home_dir, "asleep", reason="settled desk", written_by="day-gate",
            wake_at=(datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(),
        )

        llm = _ScriptedLLM(["One.", "R1.", "Two.", "R2."])
        loop = LifeLoop(
            _factory_for(home_dir, llm),
            tick_seconds=1, ticks_per_day=1, max_ticks=1,
            stop_file=home_dir / "STOP", state_home=home_dir,
            rest_minutes=0.001,
            sleep_fn=lambda s: None, out=lambda s: None,
        )
        loop.run()

        rows = [
            json.loads(ln)
            for ln in (home_dir / "state_history.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        day_markers = [
            r for r in rows
            if r.get("marker") == "phase_changed" and r.get("to") in ("work", "personal")
        ]
        assert day_markers, "a day opened and was marked"
        first = day_markers[0]
        # THE PIN: despite the standing work order, the first day that
        # opened is PERSONAL — the removed edge governed the boundary.
        assert first["to"] == "personal", f"work day opened over a removed edge: {first}"
        assert first["from"] == "sleep"
        assert first["cause"] == "cadence_need_check"


class TestGateSkipPhases:
    def test_skip_work_falls_to_personal(self, tmp_path):
        home = _arm_home(tmp_path, grant=True, work_order=True)
        decision = read_day_gate(home, skip_phases=frozenset({"work"}))
        assert decision["phase"] == "personal"

    def test_skip_personal_lands_sleep_honestly_named(self, tmp_path):
        home = _arm_home(tmp_path, grant=True, work_order=False)
        decision = read_day_gate(home, skip_phases=frozenset({"personal"}))
        assert decision["phase"] == "sleep"
        assert decision["cause"] == "edge_removed"
