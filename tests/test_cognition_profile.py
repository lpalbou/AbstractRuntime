"""The entity's cognition profile (`<home>/cognition.yaml`, identity/cognition_profile.py).

Pins: every capability defaults to the mind and nothing else (no code default for a model);
a profile choice resolves to itself while dialogue stays the mind; the recall width and the
rebuild policy read their defaults when absent; writes are validated before the file moves
and are atomic; a malformed file or field reads as unset with a warning naming the field.
"""

from __future__ import annotations

import ast
import inspect
import logging
import types
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from abstractruntime.identity import cognition_profile as cp  # noqa: E402
from abstractruntime.identity.substrate import (  # noqa: E402
    SUBSTRATE_ENV_MODEL,
    SUBSTRATE_ENV_PROVIDER,
    SubstrateUnset,
    read_home_substrate,
)

pytestmark = pytest.mark.basic

THEFORGE = {"provider": "endpoint:compat-theforge-nvidia", "model": "qwen/qwen3.8-27b"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SUBSTRATE_ENV_PROVIDER, raising=False)
    monkeypatch.delenv(SUBSTRATE_ENV_MODEL, raising=False)


def _write_mind(home: Path, provider: str = "endpoint:airelay", model: str = "gpt-5.6-terra") -> None:
    (home / "substrate.yaml").write_text(yaml.safe_dump({"provider": provider, "model": model}), encoding="utf-8")


def _assert_default_resolves_to_mind(module: types.ModuleType, home: Path) -> None:
    """No cognition.yaml: every capability is the mind, with source "mind"."""
    for cap in module.CAPABILITIES:
        got = module.resolve_capability(home, cap)
        assert got == {
            "provider": "endpoint:airelay",
            "model": "gpt-5.6-terra",
            "thinking": None,
            "max_input_tokens": 140000,
            "source": "mind",
        }, cap


def test_without_a_profile_every_capability_is_the_mind(tmp_path: Path) -> None:
    _write_mind(tmp_path)
    _assert_default_resolves_to_mind(cp, tmp_path)
    profile = cp.read_cognition_profile(tmp_path)
    assert profile["schema_version"] == 1
    assert {cap: row["choice"] for cap, row in profile["capabilities"].items()} == {
        "summaries": {"inherit": "mind"},
        "understanding": {"inherit": "mind"},
        "extraction": {"inherit": "mind"},
    }
    assert all(row["source"] == "default" for row in profile["capabilities"].values())
    assert profile["recall"] == {"shelf_size": None, "source": "default"}
    assert profile["rebuild"] == {
        "sensitivity": 0.5,
        "cap_per_turn": 3,
        "sources": {"sensitivity": "default", "cap_per_turn": "default"},
    }
    assert cp.recall_shelf_size(tmp_path, 50) == 50
    assert cp.rebuild_policy(tmp_path) == {"sensitivity": 0.5, "cap_per_turn": 3}
    assert profile["lookback"] == {"context": "day", "source": "default"}
    assert profile["warnings"] == []


def test_no_mind_and_no_profile_refuses_loudly(tmp_path: Path) -> None:
    with pytest.raises(SubstrateUnset):
        cp.resolve_capability(tmp_path, "summaries")


def test_the_hosts_resolved_mind_is_the_explicit_rung(tmp_path: Path) -> None:
    """The gateway's mind can be its text route (no substrate.yaml): it passes the mind it resolved."""
    got = cp.resolve_capability(
        tmp_path, "understanding", mind={"provider": "lmstudio", "model": "route-model", "thinking": "low"}
    )
    assert got == {
        "provider": "lmstudio", "model": "route-model", "thinking": "low",
        "max_input_tokens": 140000, "source": "mind",
    }


def test_a_profile_choice_resolves_to_itself_and_dialogue_stays_the_mind(tmp_path: Path) -> None:
    _write_mind(tmp_path)
    stored = cp.write_cognition_profile(
        tmp_path, {"capabilities": {"summaries": {**THEFORGE, "thinking": "LOW", "max_input_tokens": 62464}}}
    )
    assert stored == {
        "schema_version": 1,
        "capabilities": {"summaries": {**THEFORGE, "thinking": "low", "max_input_tokens": 62464}},
    }
    assert cp.resolve_capability(tmp_path, "summaries") == {
        **THEFORGE, "thinking": "low", "max_input_tokens": 62464, "source": "profile",
    }
    assert cp.resolve_capability(tmp_path, "understanding")["source"] == "mind"
    assert cp.resolve_capability(tmp_path, "extraction")["model"] == "gpt-5.6-terra"
    # The mind itself is untouched.
    assert read_home_substrate(tmp_path) == {"provider": "endpoint:airelay", "model": "gpt-5.6-terra"}
    # A custom choice without a limit reads the 140000 ceiling.
    cp.write_cognition_profile(tmp_path, {"capabilities": {"summaries": dict(THEFORGE)}})
    row = cp.read_cognition_profile(tmp_path)["capabilities"]["summaries"]
    assert row == {"choice": {**THEFORGE, "thinking": None, "max_input_tokens": 140000}, "source": "profile"}


def test_round_trip_keeps_only_what_was_chosen(tmp_path: Path) -> None:
    stored = cp.write_cognition_profile(
        tmp_path,
        {
            "schema_version": 1,
            "capabilities": {"summaries": {"inherit": "mind"}, "understanding": dict(THEFORGE), "extraction": None},
            "recall": {"shelf_size": 7},
            "rebuild": {"sensitivity": 1, "cap_per_turn": 0},
        },
    )
    assert stored == {
        "schema_version": 1,
        "capabilities": {"understanding": dict(THEFORGE)},
        "recall": {"shelf_size": 7},
        "rebuild": {"sensitivity": 1.0, "cap_per_turn": 0},
    }
    assert yaml.safe_load((tmp_path / "cognition.yaml").read_text(encoding="utf-8")) == stored
    assert cp.stored_cognition_profile(tmp_path) == stored
    assert cp.recall_shelf_size(tmp_path, 50) == 7
    assert cp.rebuild_policy(tmp_path) == {"sensitivity": 1.0, "cap_per_turn": 0}
    profile = cp.read_cognition_profile(tmp_path)
    assert profile["recall"] == {"shelf_size": 7, "source": "profile"}
    assert profile["rebuild"]["sources"] == {"sensitivity": "profile", "cap_per_turn": "profile"}
    # The temporary file is gone after the rename.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cognition.yaml"]


@pytest.mark.parametrize(
    "profile, field, words",
    [
        ({"recall": {"shelf_size": 0}}, "recall.shelf_size", "Memories per turn must be a whole number from 1 to 400."),
        ({"recall": {"shelf_size": 401}}, "recall.shelf_size", "from 1 to 400"),
        ({"recall": {"shelf_size": "7"}}, "recall.shelf_size", "from 1 to 400"),
        ({"recall": {"shelf_size": True}}, "recall.shelf_size", "from 1 to 400"),
        ({"recall": {"shelf_size": 7.0}}, "recall.shelf_size", "from 1 to 400"),
        ({"rebuild": {"cap_per_turn": 11}}, "rebuild.cap_per_turn", "Rebuilds per turn must be a whole number from 0 to 10."),
        ({"rebuild": {"cap_per_turn": True}}, "rebuild.cap_per_turn", "from 0 to 10"),
        ({"rebuild": {"sensitivity": 1.5}}, "rebuild.sensitivity", "Rebuild sensitivity must be a number from 0 to 1."),
        ({"rebuild": {"sensitivity": "0.5"}}, "rebuild.sensitivity", "from 0 to 1"),
        ({"rebuild": {"sensitivity": True}}, "rebuild.sensitivity", "from 0 to 1"),
        ({"rebuild": {"sensitivity": float("nan")}}, "rebuild.sensitivity", "from 0 to 1"),
        ({"capabilities": {"dialogue": dict(THEFORGE)}}, "capabilities", "no part of thinking called 'dialogue'"),
        ({"capabilities": {"summaries": {"inherit": "gateway"}}}, "capabilities.summaries", "only inherit from the mind"),
        ({"capabilities": {"summaries": {"provider": "lmstudio"}}}, "capabilities.summaries", "both a provider and a model"),
        ({"capabilities": {"summaries": {**THEFORGE, "thinking": "turbo"}}}, "capabilities.summaries.thinking", "Unknown reasoning effort 'turbo'"),
        ({"capabilities": {"summaries": {**THEFORGE, "max_input_tokens": 4095}}}, "capabilities.summaries.max_input_tokens", "from 4096 to 140000"),
        ({"capabilities": {"summaries": {**THEFORGE, "max_input_tokens": 200000}}}, "capabilities.summaries.max_input_tokens", "from 4096 to 140000"),
        ({"capabilities": {"summaries": {**THEFORGE, "temperature": 0.2}}}, "capabilities.summaries", "unknown field 'temperature'"),
        ({"recall": {"shelf_size": 9, "width": 3}}, "recall.width", "unknown field 'width'"),
        ({"memories_rebuilt_per_turn": 1}, "memories_rebuilt_per_turn", "Unknown field"),
        ({"schema_version": 2}, "schema_version", "schema 1"),
        ({"lookback": {"context": "week"}}, "lookback.context", "Look-back context must be day or tick."),
        ({"lookback": {"context": True}}, "lookback.context", "Look-back context must be day or tick."),
        ({"lookback": {"context": "day", "depth": 3}}, "lookback.depth", "unknown field 'depth'"),
    ],
)
def test_a_write_outside_the_schema_is_refused_before_the_file_moves(
    tmp_path: Path, profile: dict, field: str, words: str
) -> None:
    cp.write_cognition_profile(tmp_path, {"recall": {"shelf_size": 12}})
    before = (tmp_path / "cognition.yaml").read_text(encoding="utf-8")
    with pytest.raises(cp.CognitionProfileInvalid) as refused:
        cp.write_cognition_profile(tmp_path, profile)
    assert refused.value.field == field
    assert words in str(refused.value)
    assert (tmp_path / "cognition.yaml").read_text(encoding="utf-8") == before


def test_a_malformed_file_reads_as_unset_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write_mind(tmp_path)
    path = tmp_path / "cognition.yaml"
    for text in ("capabilities: [unclosed\n", "- a list\n", "schema_version: 2\nrecall: {shelf_size: 9}\n"):
        path.write_text(text, encoding="utf-8")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="abstractruntime.identity.cognition_profile"):
            profile = cp.read_cognition_profile(tmp_path)
        assert profile["recall"]["shelf_size"] is None
        assert profile["warnings"] and caplog.records, text
        _assert_default_resolves_to_mind(cp, tmp_path)


def test_a_bad_field_never_hides_a_good_one(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "cognition.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "capabilities": {"summaries": {"provider": "lmstudio"}, "understanding": dict(THEFORGE)},
                "recall": {"shelf_size": 0},
                "rebuild": {"sensitivity": 0.8, "cap_per_turn": "3"},
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="abstractruntime.identity.cognition_profile"):
        profile = cp.read_cognition_profile(tmp_path)
    warned = " ".join(profile["warnings"])
    assert "capabilities.summaries" in warned and "recall.shelf_size" in warned and "rebuild.cap_per_turn" in warned
    assert profile["capabilities"]["summaries"]["choice"] == {"inherit": "mind"}
    assert profile["capabilities"]["understanding"]["source"] == "profile"
    assert profile["recall"]["shelf_size"] is None
    assert profile["rebuild"]["sensitivity"] == 0.8 and profile["rebuild"]["cap_per_turn"] == 3
    assert profile["rebuild"]["sources"] == {"sensitivity": "profile", "cap_per_turn": "default"}
    assert any("recall.shelf_size" in r.getMessage() for r in caplog.records)


def test_a_hand_written_thinking_off_is_not_swallowed(tmp_path: Path) -> None:
    (tmp_path / "cognition.yaml").write_text(
        "capabilities:\n  summaries: {provider: lmstudio, model: m, thinking: off}\n", encoding="utf-8"
    )
    assert cp.resolve_capability(tmp_path, "summaries")["thinking"] == "off"


def test_an_unknown_capability_is_refused_at_resolve(tmp_path: Path) -> None:
    _write_mind(tmp_path)
    with pytest.raises(cp.CognitionProfileInvalid):
        cp.resolve_capability(tmp_path, "dialogue")


def _mutant_without_the_inherit_branch() -> types.ModuleType:
    """cognition_profile with `resolve_capability`'s inherit-the-mind branch deleted."""
    tree = ast.parse(inspect.getsource(cp))
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_capability")
    branches = [n for n in func.body if isinstance(n, ast.If) and "INHERIT_MIND" in ast.unparse(n.test)]
    assert len(branches) == 1, "the inherit branch moved: update this mutant"
    func.body.remove(branches[0])
    mutant = types.ModuleType("abstractruntime.identity._cognition_profile_mutant")
    mutant.__package__ = "abstractruntime.identity"
    exec(compile(tree, cp.__file__, "exec"), mutant.__dict__)  # noqa: S102 - the module's own source
    return mutant


def test_the_default_test_turns_red_without_the_inherit_branch(tmp_path: Path) -> None:
    _write_mind(tmp_path)
    mutant = _mutant_without_the_inherit_branch()
    with pytest.raises(Exception):
        _assert_default_resolves_to_mind(mutant, tmp_path)
    # The same check passes on the real module.
    _assert_default_resolves_to_mind(cp, tmp_path)


def _assert_default_lookback_is_the_day(module: types.ModuleType, home: Path) -> None:
    assert module.lookback_context(home) == "day"
    assert module.read_cognition_profile(home)["lookback"] == {"context": "day", "source": "default"}


def test_lookback_context_reads_the_day_by_default_and_round_trips(tmp_path: Path) -> None:
    _assert_default_lookback_is_the_day(cp, tmp_path)
    stored = cp.write_cognition_profile(tmp_path, {"lookback": {"context": "Tick"}, "recall": {"shelf_size": 8}})
    assert stored == {"schema_version": 1, "recall": {"shelf_size": 8}, "lookback": {"context": "tick"}}
    assert yaml.safe_load((tmp_path / "cognition.yaml").read_text(encoding="utf-8")) == stored
    assert cp.lookback_context(tmp_path) == "tick"
    assert cp.read_cognition_profile(tmp_path)["lookback"] == {"context": "tick", "source": "profile"}
    # null returns it to the default; the stored form drops it.
    stored = cp.write_cognition_profile(tmp_path, {"lookback": {"context": None}, "recall": {"shelf_size": 8}})
    assert "lookback" not in stored
    _assert_default_lookback_is_the_day(cp, tmp_path)


def test_an_invalid_lookback_reads_as_the_day_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "cognition.yaml").write_text(
        "schema_version: 1\nlookback: {context: week}\nrecall: {shelf_size: 9}\n", encoding="utf-8"
    )
    with caplog.at_level(logging.WARNING, logger="abstractruntime.identity.cognition_profile"):
        profile = cp.read_cognition_profile(tmp_path)
    assert profile["lookback"] == {"context": "day", "source": "default"}
    assert profile["recall"]["shelf_size"] == 9
    assert any("lookback.context" in w for w in profile["warnings"])
    assert any("lookback.context" in r.getMessage() for r in caplog.records)
    assert cp.lookback_context(tmp_path) == "day"


def test_the_lookback_default_test_turns_red_when_the_default_moves(tmp_path: Path) -> None:
    tree = ast.parse(inspect.getsource(cp))
    assigns = [
        n for n in tree.body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "DEFAULT_LOOKBACK_CONTEXT" for t in n.targets)
    ]
    assert len(assigns) == 1, "the default moved: update this mutant"
    assigns[0].value = ast.copy_location(ast.Constant("tick"), assigns[0].value)
    mutant = types.ModuleType("abstractruntime.identity._cognition_profile_lookback_mutant")
    mutant.__package__ = "abstractruntime.identity"
    exec(compile(tree, cp.__file__, "exec"), mutant.__dict__)  # noqa: S102 - the module's own source
    with pytest.raises(AssertionError):
        _assert_default_lookback_is_the_day(mutant, tmp_path)
    _assert_default_lookback_is_the_day(cp, tmp_path)


# ---------------------------------------------------------------- sleep thresholds (CONTRACT v3 D2): idle -> sleep


def test_the_sleep_thresholds_default_to_24_hours_and_500_memories_and_round_trip(tmp_path: Path) -> None:
    profile = cp.read_cognition_profile(tmp_path)
    assert profile["sleep"] == {"after_hours": 24, "after_events": 500, "after_fragmented_share": 0.15,
                                "sources": {"after_hours": "default", "after_events": "default",
                                            "after_fragmented_share": "default"}}
    assert cp.sleep_thresholds(tmp_path) == {"after_hours": 24, "after_events": 500, "after_fragmented_share": 0.15}
    stored = cp.write_cognition_profile(tmp_path, {"sleep": {"after_hours": 720, "after_events": 1}})
    assert stored == {"schema_version": 1, "sleep": {"after_hours": 720, "after_events": 1}}
    assert yaml.safe_load((tmp_path / "cognition.yaml").read_text(encoding="utf-8")) == stored
    assert cp.sleep_thresholds(tmp_path) == {"after_hours": 720, "after_events": 1, "after_fragmented_share": 0.15}
    assert cp.read_cognition_profile(tmp_path)["sleep"]["sources"] == {"after_hours": "profile",
                                                                       "after_events": "profile",
                                                                       "after_fragmented_share": "default"}
    # One field chosen, the other its default; null returns it to the default.
    stored = cp.write_cognition_profile(tmp_path, {"sleep": {"after_hours": 6, "after_events": None}})
    assert stored["sleep"] == {"after_hours": 6}
    assert cp.sleep_thresholds(tmp_path) == {"after_hours": 6, "after_events": 500, "after_fragmented_share": 0.15}


def test_the_fragmented_share_threshold_defaults_to_15_percent_and_round_trips(tmp_path: Path) -> None:
    """Operator 2026-11-10: the idle sweep also sleeps him once more than this share of his live memories is
    fragmented (provisional topic placement, no vector, a title or summary pending). 0 < share <= 1."""
    assert cp.DEFAULT_SLEEP_AFTER_FRAGMENTED_SHARE == 0.15
    stored = cp.write_cognition_profile(tmp_path, {"sleep": {"after_fragmented_share": 0.4}})
    assert stored == {"schema_version": 1, "sleep": {"after_fragmented_share": 0.4}}
    assert yaml.safe_load((tmp_path / "cognition.yaml").read_text(encoding="utf-8")) == stored
    profile = cp.read_cognition_profile(tmp_path)
    assert profile["sleep"]["after_fragmented_share"] == 0.4
    assert profile["sleep"]["sources"]["after_fragmented_share"] == "profile"
    assert cp.sleep_thresholds(tmp_path) == {"after_hours": 24, "after_events": 500, "after_fragmented_share": 0.4}
    # The whole share is allowed (an int reads as the number it is); null returns it to the default.
    assert cp.write_cognition_profile(tmp_path, {"sleep": {"after_fragmented_share": 1}})["sleep"] == {
        "after_fragmented_share": 1.0}
    assert cp.write_cognition_profile(tmp_path, {"sleep": {"after_fragmented_share": None}}) == {"schema_version": 1}
    assert cp.sleep_thresholds(tmp_path)["after_fragmented_share"] == 0.15


@pytest.mark.parametrize(
    "value, field, words",
    [
        ({"after_hours": 0}, "sleep.after_hours", "Hours before he sleeps must be a whole number from 1 to 720."),
        ({"after_hours": 721}, "sleep.after_hours", "from 1 to 720"),
        ({"after_hours": 24.0}, "sleep.after_hours", "from 1 to 720"),
        ({"after_hours": True}, "sleep.after_hours", "from 1 to 720"),
        ({"after_events": 0}, "sleep.after_events", "New memories before he sleeps must be a whole number of at least 1."),
        ({"after_events": "500"}, "sleep.after_events", "at least 1"),
        ({"after_minutes": 5}, "sleep.after_minutes", "unknown field 'after_minutes'"),
        ({"after_fragmented_share": 0}, "sleep.after_fragmented_share",
         "The share of fragmented memories before he sleeps must be a number above 0 and at most 1 (0.15 = 15%)."),
        ({"after_fragmented_share": 1.01}, "sleep.after_fragmented_share", "above 0 and at most 1"),
        ({"after_fragmented_share": -0.1}, "sleep.after_fragmented_share", "above 0 and at most 1"),
        ({"after_fragmented_share": float("nan")}, "sleep.after_fragmented_share", "above 0 and at most 1"),
        ({"after_fragmented_share": True}, "sleep.after_fragmented_share", "above 0 and at most 1"),
        ({"after_fragmented_share": "0.15"}, "sleep.after_fragmented_share", "above 0 and at most 1"),
    ],
)
def test_a_sleep_threshold_outside_the_schema_is_refused_with_its_field(tmp_path: Path, value: dict, field: str,
                                                                        words: str) -> None:
    with pytest.raises(cp.CognitionProfileInvalid) as refused:
        cp.write_cognition_profile(tmp_path, {"sleep": value})
    assert refused.value.field == field and words in str(refused.value)
    assert not (tmp_path / "cognition.yaml").exists()


def test_a_bad_sleep_threshold_in_the_file_reads_as_its_default_with_a_warning(tmp_path: Path) -> None:
    (tmp_path / "cognition.yaml").write_text(
        "schema_version: 1\nsleep: {after_hours: 0, after_events: 42}\n", encoding="utf-8"
    )
    profile = cp.read_cognition_profile(tmp_path)
    assert (profile["sleep"]["after_hours"], profile["sleep"]["after_events"]) == (24, 42)
    assert any("sleep.after_hours" in w for w in profile["warnings"])


def test_a_bad_fragmented_share_in_the_file_reads_as_its_default_with_a_warning(tmp_path: Path) -> None:
    (tmp_path / "cognition.yaml").write_text(
        "schema_version: 1\nsleep: {after_fragmented_share: 2, after_events: 42}\n", encoding="utf-8"
    )
    profile = cp.read_cognition_profile(tmp_path)
    assert (profile["sleep"]["after_fragmented_share"], profile["sleep"]["after_events"]) == (0.15, 42)
    assert profile["sleep"]["sources"]["after_fragmented_share"] == "default"
    assert any("sleep.after_fragmented_share" in w for w in profile["warnings"])
