"""Who revoked own time (write_personal_grant(..., revoked_by=); the coordinator's ask 2026-11-10 for the gateway's
revoke doors): a revocation records its revoker - "kill_switch" (the operator's Stop) or "operator" - beside
{mode: disabled}; the next grant clears it; naming a revoker on an arming write, or an unknown one, refuses."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from test_entity_life_loop import _make_home  # noqa: E402

from abstractruntime.identity.life import (  # noqa: E402
    hard_stop_loop,
    read_personal_grant,
    write_personal_grant,
)
from abstractruntime.identity.lifecycle_fence import KILL_SWITCH, LifecycleFence  # noqa: E402


def _bucket(home_dir: Path) -> dict:
    return yaml.safe_load((home_dir / "phases.yaml").read_text(encoding="utf-8"))["personal"]


def _arm(home_dir: Path) -> None:
    write_personal_grant(home_dir, mode="until_revoked", granted_by="person:laurent",
                         fence=LifecycleFence.admit(home_dir))


@pytest.mark.parametrize("revoker, fence", [("kill_switch", "sentinel"), ("operator", "admitted")])
def test_a_revocation_records_who_revoked_it(tmp_path: Path, revoker: str, fence: str) -> None:
    home_dir = _make_home(tmp_path)
    _arm(home_dir)
    used = KILL_SWITCH if fence == "sentinel" else LifecycleFence.admit(home_dir)
    grant = write_personal_grant(home_dir, mode="disabled", revoked_by=revoker, fence=used)
    assert _bucket(home_dir) == {"mode": "disabled", "revoked_by": revoker}
    assert grant["mode"] == "disabled" and grant["revoked_by"] == revoker
    assert read_personal_grant(home_dir)["revoked_by"] == revoker


def test_a_revocation_without_a_revoker_writes_disabled_alone(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    _arm(home_dir)
    grant = write_personal_grant(home_dir, mode="disabled", fence=LifecycleFence.admit(home_dir))
    assert _bucket(home_dir) == {"mode": "disabled"}
    assert "revoked_by" not in grant


def test_the_next_grant_clears_the_revoker(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    write_personal_grant(home_dir, mode="disabled", revoked_by="operator", fence=LifecycleFence.admit(home_dir))
    _arm(home_dir)
    assert "revoked_by" not in _bucket(home_dir)
    assert "revoked_by" not in read_personal_grant(home_dir)


@pytest.mark.parametrize("mode, revoker, words", [
    ("until_revoked", "operator", "an arming write (mode until_revoked) revokes nothing"),
    ("disabled", "visitor", "own time is revoked by kill_switch or operator"),
])
def test_a_wrong_revoker_refuses_and_writes_nothing(tmp_path: Path, mode: str, revoker: str, words: str) -> None:
    home_dir = _make_home(tmp_path)
    _arm(home_dir)
    before = _bucket(home_dir)
    with pytest.raises(ValueError) as refused:
        write_personal_grant(home_dir, mode=mode, granted_by="person:laurent", revoked_by=revoker,
                             fence=LifecycleFence.admit(home_dir))
    assert words in str(refused.value)
    assert _bucket(home_dir) == before


def test_the_runtimes_own_stop_records_the_kill_switch(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    _arm(home_dir)
    hard_stop_loop(home_dir, reason="operator stop")
    assert _bucket(home_dir) == {"mode": "disabled", "revoked_by": "kill_switch"}
