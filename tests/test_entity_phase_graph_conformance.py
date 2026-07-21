"""THE ONE STATE GRAPH — executor conformance (laurent dm#79 via entity,
2026-07-20: "gateway MUST serve your state graph... there is only one state
graph per entity and it MUST be shared across you guys").

abstractentity owns the graph (spec/entity_phases.json); runtime VENDORS it
(identity/spec/entity_phases.vendored.json, byte-copy at sync time) and pins
its executor vocabulary against THE ARTIFACT, never against re-derived
prose — the diary_type two-copies drift class is what direct import kills
(the graph's own consumption_contract, half 1).

Bump protocol (entity's point 5): entity bumps + announces; runtime
re-vendors same-day. The sibling-drift pin below makes a missed re-vendor
loud on any machine that carries both repos.
"""
from __future__ import annotations

import json
from pathlib import Path

VENDORED = (
    Path(__file__).parent.parent
    / "src" / "abstractruntime" / "identity" / "spec" / "entity_phases.vendored.json"
)
UPSTREAM = (
    Path(__file__).parent.parent.parent
    / "abstractentity" / "spec" / "entity_phases.json"
)


def _graph() -> dict:
    return json.loads(VENDORED.read_text(encoding="utf-8"))


def test_vendored_artifact_matches_upstream_when_present() -> None:
    """The re-vendor pin: on a machine carrying both repos, a spec bump
    without a same-day re-vendor fails HERE (the announced-bump protocol
    made structural). CI without the sibling repo skips honestly."""
    import pytest

    if not UPSTREAM.exists():
        pytest.skip("abstractentity sibling repo not present")
    assert VENDORED.read_bytes() == UPSTREAM.read_bytes(), (
        "vendored graph drifted from abstractentity/spec/entity_phases.json - "
        "re-vendor (byte-copy) per the bump protocol"
    )


def test_phase_vocabulary_is_the_graphs() -> None:
    """tool_policy's PHASES == the graph's phase keys, exactly — the
    executor's radio positions are the artifact's, never a local list."""
    from abstractruntime.identity.tool_policy import PHASES

    g = _graph()
    assert set(PHASES) == set(g["phases"].keys()), (PHASES, list(g["phases"]))
    assert g["initial_phase"] in g["phases"]
    assert g["initial_phase"] == "sleep"  # NEWBORN = SLEEP (ruled)


def test_spoken_synonyms_canonicalize_to_their_phase() -> None:
    """Every spoken synonym the graph declares maps to its phase through
    canonical_phase (own-time/own time -> personal today)."""
    from abstractruntime.identity.tool_policy import canonical_phase

    g = _graph()
    for phase, spec in g["phases"].items():
        for syn in spec.get("spoken_synonyms", []):
            # canonical_phase accepts underscore/hyphen-free variants; the
            # graph's synonyms are prose spellings — normalize spaces.
            token = syn.replace(" ", "_").replace("-", "_")
            assert canonical_phase(token) == phase, (syn, phase)


def test_state_axis_words_never_enter_the_phase_vocabulary() -> None:
    """AWAKE lives ONLY on the state axis (v4/v6): the state words and the
    phase words are disjoint sets; day_kind values are phase keys."""
    from abstractruntime.identity.life import ENTITY_STATES, LOOP_PHASES
    from abstractruntime.identity.tool_policy import PHASES

    g = _graph()
    phase_words = set(g["phases"].keys())
    assert set(ENTITY_STATES) == {"awake", "asleep", "paused"}, (
        "the STATE axis vocabulary is frozen (engraved history + the "
        "sleep-bound predicate key on these strings)"
    )
    assert not (set(ENTITY_STATES) & phase_words), "state words are never phases"
    # LOOP_PHASES is the liveness plane (day/between/stopped) — also disjoint.
    assert not (set(LOOP_PHASES) & phase_words)
    # day_kind (the loop heartbeat's phase report) speaks graph words only.
    for kind in ("work", "personal"):
        assert kind in phase_words


def test_pressure_floor_mirrors_the_engine_bound() -> None:
    """DRIVES-FORBID-IDLE one-source pin: the graph mirrors
    abstractmemory.DRIVE_PRESSURE_BOUND; consumers import the bound.
    A drifted mirror fails here (the SELF_FRACTION_FLOOR precedent)."""
    import pytest

    g = _graph()
    floor = g.get("pressure_floor") or {}
    assert floor.get("bound_source") == "abstractmemory.DRIVE_PRESSURE_BOUND"
    am = pytest.importorskip("abstractmemory")
    if not hasattr(am, "DRIVE_PRESSURE_BOUND"):
        pytest.skip("installed abstractmemory predates DRIVE_PRESSURE_BOUND")
    assert int(floor.get("threshold")) == int(am.DRIVE_PRESSURE_BOUND)


def test_sleep_bound_matches_the_ruled_hour() -> None:
    """decision:sleep-is-bounded (v5): the executor's SLEEP_BOUND_SECONDS
    is the graph's ~1h ruling."""
    from abstractruntime.identity.life import SLEEP_BOUND_SECONDS

    assert SLEEP_BOUND_SECONDS == 3600


def test_transition_causes_are_a_closed_set_runtime_never_widens() -> None:
    """The phase_changed marker's cause vocabulary is the graph's closed
    set (+ explicitly reserved words that engrave only through a
    vocabulary round). Runtime emits NO phase_changed marker today (the
    ONE named writer is gateway's door half — the marker contract); this
    pin exists so any future runtime writer must draw from the artifact."""
    g = _graph()
    causes = set(g["transition_causes"])
    assert {"operator", "visit_open", "visit_close", "self_elected"} <= causes
    # The reserved slots are placeholders, not usable words.
    for r in g.get("transition_causes_reserved", []):
        assert r.startswith("<") and r.endswith(">")


def test_runtime_writes_no_phase_markers_sibling_free() -> None:
    """Marker contract, enforced sibling-free (adversary P1-1/P1-2: the
    old pin lived in a module skipped without the abstractentity repo —
    CI never ran it). ONE kind phase_changed, ONE named writer (gateway's
    door half): no runtime identity module may write it."""
    from pathlib import Path

    import abstractruntime.identity as identity_pkg

    root = Path(identity_pkg.__file__).parent
    offenders = [
        p.name for p in root.glob("*.py")
        if "phase_changed" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"runtime must never write phase markers: {offenders}"


def test_salvage_marker_phase_canonicalizes_never_engraves_raw() -> None:
    """Adversary P1-3: the pending-reflection marker is durable JSON a
    pre-rename build may carry — a legacy/cased phase word must
    canonicalize at the read boundary (unknown -> no claim), never engrave
    raw into attributes.phase on the append-only store."""
    from abstractruntime.identity.tool_policy import canonical_phase

    # The exact boundary rule the fix implements:
    assert canonical_phase("own_time") == "personal"
    assert canonical_phase("PERSONAL".lower()) == "personal"
    import pytest

    with pytest.raises(Exception):
        canonical_phase("awake")  # a state word never becomes a phase claim


def test_day_kind_writer_accepts_graph_words_only() -> None:
    """Adversary P2-5: the loop_status day_kind writer validates against
    the graph vocabulary — a drifted caller cannot serve a non-graph word."""
    import tempfile
    from pathlib import Path

    from abstractruntime.identity.life import read_loop_status, write_loop_status

    tmp = Path(tempfile.mkdtemp())
    write_loop_status(tmp, "day", day_kind="WORK")
    assert read_loop_status(tmp).get("day_kind") == "work"
    write_loop_status(tmp, "day", day_kind="awake")
    assert read_loop_status(tmp).get("day_kind") is None, "non-graph word dropped"


def test_mode_words_frozen_set() -> None:
    """Adversary P1-4, CLOSED at v7: the artifact's axes_note now carries
    THE MODE AXIS law (state_mode = visiting|dreaming|resting with their
    derivation roles — visiting decides the visit phase; dreaming
    decorates asleep only; resting = loop-alive-between-days). The
    runtime writers' words stay pinned here against that documented set
    so a typo'd mode can never silently break visit-yield detection."""
    # v7 documents the words in axes_note prose; assert the artifact
    # carries all three so a future bump dropping one fails here.
    note = str(_graph().get("axes_note") or "")
    for word in ("visiting", "dreaming", "resting"):
        assert word in note, f"v7+ artifact lost the mode word {word!r}"
    import re
    from pathlib import Path

    import abstractruntime.identity as identity_pkg

    # TWO mode axes exist: the STATE-file mode (visiting/dreaming/resting,
    # phase-load-bearing) and the GRANT activation mode (disabled/timer/
    # until_revoked, phases.yaml). This pin covers the STATE axis; grant
    # words are excluded by name (a third axis minting under the same
    # keyword would land in `written` and fail here - intended).
    known_state = {"visiting", "dreaming", "resting"}
    grant_axis = {"disabled", "timer", "until_revoked"}
    root = Path(identity_pkg.__file__).parent
    written = set()
    for p in root.glob("*.py"):
        for m in re.finditer(r'mode\s*=\s*"([a-z_]+)"', p.read_text(encoding="utf-8")):
            written.add(m.group(1))
    assert written - grant_axis <= known_state, (
        f"unpinned STATE-mode word(s) written: {written - grant_axis - known_state}"
    )


def test_pressure_floor_comparator_pinned() -> None:
    """Adversary P2-7: the artifact's own comment warns hardcoded bounds
    drift into > vs >= — pin the comparator beside the threshold so the
    top-gate consult (when ruled) implements the artifact's arithmetic."""
    g = _graph()
    assert (g.get("pressure_floor") or {}).get("comparator") == ">"
