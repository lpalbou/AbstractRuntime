"""Mind-substrate resolution for the home-direct CLIs (no code default).

Maintainer rulings 2026-07-09 (04:26 no-fallback; 06:32 one substrate per
entity): the chain is explicit flags > <home>/substrate.yaml > operator env
> LOUD REFUSAL. These pin the runtime half (identity.substrate); the gateway
implements the same chain HTTP-side (entity_chat.resolve_substrate).
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from abstractruntime.identity.substrate import (  # noqa: E402
    SUBSTRATE_ENV_MODEL,
    SUBSTRATE_ENV_PROVIDER,
    SubstrateUnset,
    read_home_substrate,
    resolve_home_substrate,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SUBSTRATE_ENV_PROVIDER, raising=False)
    monkeypatch.delenv(SUBSTRATE_ENV_MODEL, raising=False)


def test_unset_everywhere_refuses_loudly_and_names_every_fix(tmp_path: Path) -> None:
    with pytest.raises(SubstrateUnset) as exc:
        resolve_home_substrate(None, None, home_dir=tmp_path)
    msg = str(exc.value)
    assert "--provider" in msg and "--model" in msg
    assert "substrate.yaml" in msg
    assert SUBSTRATE_ENV_PROVIDER in msg and SUBSTRATE_ENV_MODEL in msg


def test_explicit_flags_win_over_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "substrate.yaml").write_text(
        yaml.safe_dump({"provider": "stored-p", "model": "stored-m"}), encoding="utf-8"
    )
    monkeypatch.setenv(SUBSTRATE_ENV_PROVIDER, "env-p")
    monkeypatch.setenv(SUBSTRATE_ENV_MODEL, "env-m")
    assert resolve_home_substrate("flag-p", "flag-m", home_dir=tmp_path) == ("flag-p", "flag-m")


def test_home_substrate_yaml_fills_when_flags_are_unset(tmp_path: Path) -> None:
    (tmp_path / "substrate.yaml").write_text(
        yaml.safe_dump({"provider": "endpoint:ovh-provider", "model": "gpt-oss-120b"}),
        encoding="utf-8",
    )
    assert resolve_home_substrate(None, None, home_dir=tmp_path) == (
        "endpoint:ovh-provider", "gpt-oss-120b",
    )


def test_operator_env_is_the_last_rung_before_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SUBSTRATE_ENV_PROVIDER, "lmstudio")
    monkeypatch.setenv(SUBSTRATE_ENV_MODEL, "qwen3.5-4b")
    assert resolve_home_substrate(None, None, home_dir=tmp_path) == ("lmstudio", "qwen3.5-4b")


def test_fields_fill_independently_across_rungs(tmp_path: Path) -> None:
    """Gateway-approved semantics: a partial request fills the missing field
    from the next rung; refusal only when no complete pair exists."""
    (tmp_path / "substrate.yaml").write_text(
        yaml.safe_dump({"provider": "stored-p", "model": "stored-m"}), encoding="utf-8"
    )
    assert resolve_home_substrate("flag-p", None, home_dir=tmp_path) == ("flag-p", "stored-m")
    assert resolve_home_substrate(None, "flag-m", home_dir=tmp_path) == ("stored-p", "flag-m")


def test_half_or_malformed_substrate_file_reads_as_unset(tmp_path: Path) -> None:
    # Half a choice is no choice (both-or-nothing, the gateway's contract).
    (tmp_path / "substrate.yaml").write_text(
        yaml.safe_dump({"provider": "only-p"}), encoding="utf-8"
    )
    assert read_home_substrate(tmp_path) == {}
    # Malformed YAML reads as unset; the resolve site refuses loudly.
    (tmp_path / "substrate.yaml").write_text("provider: [unclosed", encoding="utf-8")
    assert read_home_substrate(tmp_path) == {}
    with pytest.raises(SubstrateUnset):
        resolve_home_substrate(None, None, home_dir=tmp_path)


def test_thinking_field_rides_the_reader_when_present(tmp_path: Path) -> None:
    """Reasoning plan R4 (gateway's build is gated on this): the reader
    returns the optional `thinking` field so the gateway's writer can store
    it without the field being silently dropped on read. Spelled `thinking`
    at rest (the plan's one-spelling decision)."""
    (tmp_path / "substrate.yaml").write_text(
        yaml.safe_dump({"provider": "p1", "model": "m1", "thinking": "high"}),
        encoding="utf-8",
    )
    assert read_home_substrate(tmp_path) == {"provider": "p1", "model": "m1", "thinking": "high"}


def test_thinking_is_optional_and_never_a_half_choice(tmp_path: Path) -> None:
    # Absent thinking: pair still reads (unchanged pre-field behavior).
    (tmp_path / "substrate.yaml").write_text(
        yaml.safe_dump({"provider": "p1", "model": "m1"}), encoding="utf-8"
    )
    assert read_home_substrate(tmp_path) == {"provider": "p1", "model": "m1"}
    # Blank thinking is dropped, not returned as an empty string.
    (tmp_path / "substrate.yaml").write_text(
        yaml.safe_dump({"provider": "p1", "model": "m1", "thinking": "  "}),
        encoding="utf-8",
    )
    assert "thinking" not in read_home_substrate(tmp_path)
    # A thinking knob WITHOUT a chosen mind is meaningless: pair-unset
    # reads as fully unset, the field never leaks out alone.
    (tmp_path / "substrate.yaml").write_text(
        yaml.safe_dump({"thinking": "high"}), encoding="utf-8"
    )
    assert read_home_substrate(tmp_path) == {}


def test_resolve_tuple_shape_is_unchanged_for_existing_callers(tmp_path: Path) -> None:
    """The resolver still returns (provider, model) — widening it is the
    coordinated implementation wave, not this unblock (the gateway and the
    runtime CLI lanes move together there)."""
    (tmp_path / "substrate.yaml").write_text(
        yaml.safe_dump({"provider": "p1", "model": "m1", "thinking": "high"}),
        encoding="utf-8",
    )
    assert resolve_home_substrate(None, None, home_dir=tmp_path) == ("p1", "m1")
