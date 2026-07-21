"""GRAPH-AS-CONTROL pins (laurent 15:06, c1505 ask 1b): the canonical phase
artifact (abstractentity/spec/entity_phases.json — ONE source, laurent 13:54)
must be what runtime's phase enforcement derives from or is verified
EQUIVALENT to — never parallel logic that happens to agree.

Runtime is an EXECUTOR, so per the artifact's own consumption contract
(half 1) these tests import the artifact directly and assert the lane's
constants and mechanics against it. Cross-repo file: skipped loudly when the
sibling repo is absent (CI checkout of runtime alone), exactly like the
emergence-experiment cross-repo contract. A drift between this file's pins
and the artifact is a FINDING for the artifact thread, never something to
re-baseline silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "abstractentity" / "spec" / "entity_phases.json"
)

pytestmark = pytest.mark.skipif(
    not _ARTIFACT.exists(),
    reason=f"canonical phase artifact not checked out beside this repo ({_ARTIFACT})",
)


def _graph() -> dict:
    return json.loads(_ARTIFACT.read_text(encoding="utf-8"))


def test_phase_keys_are_the_artifacts_keys() -> None:
    """PHASES (tool_policy) == the artifact's phase nodes — the enum the
    whole lane's grants and stamps run on is the graph's, not a twin."""
    from abstractruntime.identity.tool_policy import PHASES

    assert set(PHASES) == set(_graph()["phases"].keys())


def test_spoken_synonyms_match_the_alias_table() -> None:
    """own-time is a SPOKEN synonym of personal (artifact) and the code's
    alias table maps the at-rest spelling own_time -> personal — same fact,
    both directions; no alias may exist without an artifact synonym and no
    synonym without a legacy mapping (visit/work/sleep declare none)."""
    from abstractruntime.identity.tool_policy import LEGACY_PHASE_ALIASES

    graph = _graph()
    synonyms = {
        phase: set(node.get("spoken_synonyms") or [])
        for phase, node in graph["phases"].items()
    }
    # v8 added the at-rest spelling own_time to the artifact's synonym
    # list itself (one list carries all three spellings now).
    assert synonyms["personal"] == {"own-time", "own time", "own_time"}
    assert synonyms["visit"] == set() and synonyms["work"] == set() and synonyms["sleep"] == set()
    # The code's own_time legacy spelling maps to the phase that OWNS the
    # spoken synonym; resident/tasked are pre-ruling history, mapped to
    # ruled keys that exist in the graph.
    assert LEGACY_PHASE_ALIASES["own_time"] == "personal"
    assert set(LEGACY_PHASE_ALIASES.values()) <= set(graph["phases"].keys())


def test_grant_end_causes_come_from_the_artifacts_closed_set() -> None:
    """personal_grant_end_cause emits ONLY words from the artifact's
    transition_causes, and the artifact carries a personal->sleep
    transition for each — the ruled landing my loop enacts."""
    from abstractruntime.identity.life import personal_grant_end_cause

    graph = _graph()
    causes = set(graph["transition_causes"])
    expired = personal_grant_end_cause(
        {"mode": "timer", "expires_at": "2020-01-01T00:00:00+00:00"}
    )
    revoked = personal_grant_end_cause({"mode": "disabled"})
    assert expired == "grant_expired" and expired in causes
    assert revoked == "grant_revoked" and revoked in causes
    landings = {
        (t["from"], t["to"], t["cause"]) for t in graph["transitions"]
    }
    assert ("personal", "sleep", "grant_expired") in landings
    assert ("personal", "sleep", "grant_revoked") in landings


def test_personal_gating_matches_the_artifacts_invariant() -> None:
    """The artifact's grant-gating invariant (off-by-default, armed grant
    within expiry, process-existence is never permission) is EXACTLY what
    read_personal_grant/personal_grant_refusal enforce."""
    from abstractruntime.identity.life import (
        PERSONAL_GRANT_MODES,
        personal_grant_refusal,
    )

    graph = _graph()
    gating = str(graph["phases"]["personal"].get("gating") or "")
    assert "disabled" in gating and "expiry" in gating
    assert any("OFF BY DEFAULT" in inv for inv in graph["invariants"])
    # Off-by-default: the absent/disabled bucket refuses.
    assert personal_grant_refusal({"mode": "disabled"}) is not None
    # Within-expiry: a live grant passes, a lapsed one refuses.
    assert personal_grant_refusal({"mode": "until_revoked"}) is None
    assert personal_grant_refusal(
        {"mode": "timer", "expires_at": "2999-01-01T00:00:00+00:00"}
    ) is None
    assert personal_grant_refusal(
        {"mode": "timer", "expires_at": "2020-01-01T00:00:00+00:00"}
    ) is not None
    assert "disabled" in PERSONAL_GRANT_MODES  # the off-by-default word itself


def test_restore_previous_rides_the_standing_grant() -> None:
    """Artifact: visit->personal restore happens THROUGH the standing grant
    (grant gone -> sleep instead). Runtime's mechanic: the loop re-checks
    the grant at every day-open, so re-entry after a visit IS a grant-gated
    day-open — verified equivalent by the transitions' semantics text and
    the gate's existence."""
    import inspect

    from abstractruntime.identity import life as life_mod

    graph = _graph()
    restore = next(
        t for t in graph["transitions"]
        if t["from"] == "visit" and t["to"] == "personal" and t["cause"] == "visit_close"
    )
    assert "STANDING GRANT" in restore["semantics"].upper()
    # The gate the re-entry rides: run() consults personal_grant_refusal
    # before any summon (source-level pin — the control point exists on the
    # execution path, not in a parallel module).
    source = inspect.getsource(life_mod.LifeLoop.run)
    assert "personal_grant_refusal" in source
    assert "read_personal_grant" in source


def test_newborn_shape_is_the_artifacts_answer() -> None:
    """NEWBORN=SLEEP: the artifact rules state=asleep at birth with the
    phase derivation folding asleep->sleep, and KEEPS every door open via
    auto-wake. Runtime's missing-state-file default (awake) is therefore a
    GATEWAY-BIRTH-WRITE dependency — pinned here so the contract's runtime
    half (asleep handled correctly from birth: gate idles, doors wake) is
    the verified-equivalent behavior, not an accident."""
    from abstractruntime.identity.life import read_entity_state

    graph = _graph()
    assert graph["initial_phase"] == "sleep"
    assert any("NEWBORN = SLEEP" in inv for inv in graph["invariants"])
    # Runtime half: an asleep state file reads back asleep (the loop's gate
    # idles on it; visit doors auto-wake) — no runtime surface fights the
    # birth write.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        (home / "state").write_text(
            '{"state": "asleep", "reason": "born"}\n', encoding="utf-8"
        )
        assert read_entity_state(home)["state"] == "asleep"


def test_runtime_writes_no_phase_markers() -> None:
    """Artifact marker contract: ONE kind phase_changed, ONE named writer
    (the gateway's door half). Runtime must not write phase markers —
    source-level pin over the identity lane."""
    import abstractruntime.identity as identity_pkg

    root = Path(identity_pkg.__file__).parent
    offenders = [
        p.name
        for p in root.glob("*.py")
        if "phase_changed" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"runtime must never write phase markers: {offenders}"
