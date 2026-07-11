"""Phase vocabulary: the four ruled keys + the legacy-spelling migration
window (config-object consensus F7/N7 + laurent's human-words ruling c786:
visit/work/personal/sleep, 2026-07-11).

Pins, in the order the consensus argued them:
- The phase SET is runtime's and root-exported (the door imports from the
  root, never a second copy — the diary_type-clamp lesson).
- Legacy spellings ("resident"/"own_time"→personal, "tasked"→work) map
  LOUDLY on BOTH axes: the ARG (a pre-flip caller like gateway's literals
  keeps working) and the FILE section (an operator's pre-rename narrow
  grant SURVIVES — never the silent widen F7 traced).
- Writes normalize legacy keys on disk (files converge without an
  operator act); explicit own_time wins when both spellings are present.
- Unknown phases still raise at the arg (refusing code owns the set) and
  warn as file keys.
- N8: a narrow `work:` file section is CONSULTED (source=="policy-file")
  — the full-set default must never make work "accidentally correct"
  through a permissive fallthrough.

MIGRATION END-STATE (dies before release, the lease-shim policy): when the
aliases are removed, flip test_legacy_arg_resolves_own_time_with_note and
its siblings from expects-mapping to expects-raise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from abstractruntime.identity.tool_policy import (  # noqa: E402
    ALL_TOOL_NAMES,
    LEGACY_PHASE_ALIASES,
    PHASES,
    POLICY_FILENAME,
    SLEEP_DEFAULT_TOOL_NAMES,
    resolve_tool_grant,
    write_policy_file,
)
from abstractruntime.identity.tools import (  # noqa: E402
    TIER1_TOOL_NAMES,
    WORKSPACE_TOOL_NAMES,
)

_FULL = TIER1_TOOL_NAMES + WORKSPACE_TOOL_NAMES


# ------------------------------------------------------------ the set


def test_phases_are_the_four_ruled_keys() -> None:
    assert PHASES == ("visit", "work", "personal", "sleep")
    assert LEGACY_PHASE_ALIASES == {
        "resident": "personal", "own_time": "personal", "tasked": "work",
    }


def test_root_exports_phase_vocabulary_and_resolver() -> None:
    """F7: the door imports from the root — one source, no second copy."""
    import abstractruntime as rt

    assert rt.PHASES == ("visit", "work", "personal", "sleep")
    assert rt.PHASE_VISIT == "visit"
    assert rt.PHASE_WORK == "work"
    assert rt.PHASE_PERSONAL == "personal"
    assert rt.PHASE_SLEEP == "sleep"
    assert rt.PHASES == (rt.PHASE_VISIT, rt.PHASE_WORK, rt.PHASE_PERSONAL, rt.PHASE_SLEEP)
    # The resolver + grant type + file surfaces ride along.
    assert rt.resolve_tool_grant is resolve_tool_grant
    assert rt.write_policy_file is write_policy_file
    assert callable(rt.read_policy_file)
    assert rt.ToolGrant is not None
    assert rt.LEGACY_PHASE_ALIASES == LEGACY_PHASE_ALIASES


def test_every_historical_phase_word_resolves_through_the_canon() -> None:
    """CANON GOVERNANCE (agency c831, runtime-ruled): no word that ever
    appeared in PHASES may leave the vocabulary without an alias — every
    historical phase word resolves through canonical_phase without
    raising, which makes gateway's unknown-verified-phase fallback
    STRUCTURALLY UNREACHABLE for as long as stamps signed under any past
    canon can exist. At the pre-release alias-removal edit, this test is
    revisited DELIBERATELY together with the expects-raise flips (the
    c830 removal checklist: "no phase string ever leaves the canon
    without either an alias or a ruling on the fallback") — never
    silently deleted."""
    from abstractruntime.identity.tool_policy import canonical_phase

    historical = ("visit", "tasked", "own_time", "resident", "work", "personal", "sleep")
    for word in historical:
        resolved = canonical_phase(word)
        assert resolved in PHASES, (word, resolved)


def test_canonical_phase_is_the_stamp_normalizer() -> None:
    """Semantics c700 V5: entity-stamp-v2 must sign the CANONICAL phase
    only — an alias inside the MAC basis is a verify-time chain-break.
    canonical_phase is the one public gate for consumers that persist or
    sign phase strings."""
    import abstractruntime as rt

    assert rt.canonical_phase("visit") == "visit"
    assert rt.canonical_phase("work") == "work"
    assert rt.canonical_phase("personal") == "personal"
    assert rt.canonical_phase("sleep") == "sleep"
    assert rt.canonical_phase("resident") == "personal"  # alias normalized at mint
    assert rt.canonical_phase("own_time") == "personal"
    assert rt.canonical_phase("tasked") == "work"
    with pytest.raises(ValueError, match="unknown phase"):
        rt.canonical_phase("weekend")


# ----------------------------------------------- legacy ARG (caller axis)


def test_legacy_arg_resolves_own_time_with_note(tmp_path: Path) -> None:
    """A pre-flip caller (gateway literals, an old loop) passing
    phase="resident" gets own_time's grant + a #FALLBACK note — never a
    raise, never silence."""
    grant = resolve_tool_grant(tmp_path, "resident")
    assert grant.tools == _FULL  # the personal phase's ruled default (Q1 c684)
    assert any("#FALLBACK" in n and "resident" in n and "personal" in n for n in grant.notes)
    tasked = resolve_tool_grant(tmp_path, "tasked")
    assert any("#FALLBACK" in n and "work" in n for n in tasked.notes)


def test_legacy_arg_reads_the_own_time_file_section(tmp_path: Path) -> None:
    """The spellings are ONE phase: a personal file section narrows a
    resident-arg caller too."""
    write_policy_file(tmp_path, {"personal": ["diary_list"]})
    grant = resolve_tool_grant(tmp_path, "resident")
    assert grant.tools == ("diary_list",)
    assert grant.source == "policy-file"


def test_unknown_phase_arg_still_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown phase"):
        resolve_tool_grant(tmp_path, "weekend")


# ------------------------------------------------ legacy FILE (home axis)


def test_legacy_file_section_survives_the_rename_never_widens(tmp_path: Path) -> None:
    """THE F7 CASE: an operator's narrow pre-rename `resident:` grant must
    keep narrowing own_time — falling to the full-set default would be a
    silent WIDEN of a deliberate restriction."""
    (tmp_path / POLICY_FILENAME).write_text(
        yaml.safe_dump({"own_time": {"tools": ["diary_list", "read_memory"]}}),
        encoding="utf-8",
    )
    grant = resolve_tool_grant(tmp_path, "personal")
    assert grant.tools == ("diary_list", "read_memory")  # narrow SURVIVES
    assert grant.source == "policy-file"
    # The note names the actual file so the operator can fix the spelling.
    assert any("#FALLBACK" in n and str(tmp_path / POLICY_FILENAME) in n for n in grant.notes)


def test_explicit_personal_section_wins_over_legacy_twin(tmp_path: Path) -> None:
    (tmp_path / POLICY_FILENAME).write_text(
        yaml.safe_dump({
            "own_time": {"tools": ["web_search"]},
            "personal": {"tools": ["diary_list"]},
        }),
        encoding="utf-8",
    )
    grant = resolve_tool_grant(tmp_path, "personal")
    assert grant.tools == ("diary_list",)  # the ruled key is the word


def test_unknown_file_key_warns_in_notes(tmp_path: Path) -> None:
    """A typo'd section silently granting nothing is the same class of
    quiet loss — surface it."""
    (tmp_path / POLICY_FILENAME).write_text(
        yaml.safe_dump({"vist": {"tools": ["diary_list"]}}),  # typo'd "visit"
        encoding="utf-8",
    )
    grant = resolve_tool_grant(tmp_path, "visit")
    assert grant.source == "default"
    assert any("unknown phase key" in n and "vist" in n for n in grant.notes)


# --------------------------------------------------- writes converge files


def test_write_normalizes_legacy_key_on_disk(tmp_path: Path) -> None:
    """N7 migration contract (c672): writes land with the ruled keys, so
    home files converge to the new vocabulary without an operator act."""
    with pytest.warns(UserWarning, match="resident"):
        write_policy_file(tmp_path, {"resident": ["diary_list"]})
    on_disk = yaml.safe_load((tmp_path / POLICY_FILENAME).read_text(encoding="utf-8"))
    assert "resident" not in on_disk
    assert on_disk["personal"] == {"tools": ["diary_list"]}
    assert resolve_tool_grant(tmp_path, "personal").tools == ("diary_list",)


def test_any_write_migrates_legacy_keys_at_rest(tmp_path: Path) -> None:
    """A pre-rename file gains the ruled key on the NEXT write — whichever
    phase that write touches."""
    (tmp_path / POLICY_FILENAME).write_text(
        yaml.safe_dump({"tasked": {"tools": ["read_memory"]}}),
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="normalized"):
        write_policy_file(tmp_path, {"visit": ["diary_list"]})
    on_disk = yaml.safe_load((tmp_path / POLICY_FILENAME).read_text(encoding="utf-8"))
    assert "tasked" not in on_disk
    assert on_disk["work"] == {"tools": ["read_memory"]}  # the word survives, respelled
    assert on_disk["visit"] == {"tools": ["diary_list"]}


# ----------------------------------------------------------- N8 (tasked)


def test_work_file_section_is_consulted_never_fallthrough(tmp_path: Path) -> None:
    """N8: work must not be 'accidentally correct'. A narrow work
    section proves the resolver consults the file (source=policy-file);
    equality with the default would mask a permissive fallthrough. Raw
    YAML + a decoy section (adversary find 5): a mirrored write/read key
    bug can't self-mask, and a whichever-section-exists bug picks up the
    decoy's list and fails the equality."""
    (tmp_path / POLICY_FILENAME).write_text(
        yaml.safe_dump({
            "work": {"tools": ["read_file", "list_files"]},
            "visit": {"tools": ["web_search"]},  # decoy
        }),
        encoding="utf-8",
    )
    grant = resolve_tool_grant(tmp_path, "work")
    assert grant.tools == ("read_file", "list_files")
    assert grant.source == "policy-file"
    assert grant.tools != _FULL  # the narrow word held


def test_work_and_sleep_defaults_hold(tmp_path: Path) -> None:
    assert resolve_tool_grant(tmp_path, "work").tools == _FULL
    assert resolve_tool_grant(tmp_path, "sleep").tools == SLEEP_DEFAULT_TOOL_NAMES
    # Sanity: every default name is a real tool.
    assert set(_FULL) <= set(ALL_TOOL_NAMES)


# ------------------------------------------- adversary findings (P1/P2)


def test_one_payload_naming_both_spellings_ruled_key_wins_both_orders(tmp_path: Path) -> None:
    """Adversary find 1 (P1): {"own_time": [...], "resident": [...]} in ONE
    write must land the explicit ruled key's word in EITHER insertion
    order — never last-wins by dict order."""
    with pytest.warns(UserWarning, match="ruled key"):
        write_policy_file(tmp_path, {"personal": ["diary_list"], "own_time": ["web_search"]})
    assert resolve_tool_grant(tmp_path, "personal").tools == ("diary_list",)

    (tmp_path / POLICY_FILENAME).unlink()
    with pytest.warns(UserWarning, match="ruled key"):
        write_policy_file(tmp_path, {"own_time": ["web_search"], "personal": ["diary_list"]})
    assert resolve_tool_grant(tmp_path, "personal").tools == ("diary_list",)


def test_chat_session_phase_attribute_is_canonical() -> None:
    """Adversary find 2 (P1): a session opened with the legacy spelling
    must STORE the canonical phase — `session.phase == PHASE_PERSONAL`
    comparisons (the exact usage the root constants invite) must not miss.
    Pinned at the normalizer level: the same gate ChatSession.__init__
    calls."""
    from abstractruntime.identity.tool_policy import canonical_phase

    assert canonical_phase("resident") == "personal"
    assert canonical_phase("Resident") == "personal"  # find 9: arg axis lowercases
    assert canonical_phase(" VISIT ") == "visit"


def test_malformed_ruled_key_does_not_shadow_intact_legacy_narrow(tmp_path: Path) -> None:
    """Adversary find 4 (P2): a null `own_time:` (half-finished hand edit)
    above an intact narrow `resident:` section must honor the operator's
    last intact word — falling to the FULL default would be the F7 widen
    through a side door."""
    (tmp_path / POLICY_FILENAME).write_text(
        "own_time:\nresident:\n  tools: [diary_list]\n",
        encoding="utf-8",
    )
    grant = resolve_tool_grant(tmp_path, "own_time")
    assert grant.tools == ("diary_list",)  # the intact legacy narrow held
    assert grant.source == "policy-file"
    assert any("not a mapping" in n for n in grant.notes)  # the malformed key is named


def test_write_resident_none_deletes_the_personal_entry(tmp_path: Path) -> None:
    """Adversary find 6 (P2): delete-via-normalization — {"resident": None}
    reverts the personal entry to defaults."""
    write_policy_file(tmp_path, {"personal": ["diary_list"]})
    with pytest.warns(UserWarning):
        write_policy_file(tmp_path, {"resident": None})
    assert resolve_tool_grant(tmp_path, "personal").tools == _FULL
    assert not (tmp_path / POLICY_FILENAME).exists()  # emptied file removed


def test_scalar_tools_value_is_noted_never_silent(tmp_path: Path) -> None:
    """Adversary find 7 (P2): `tools: diary_list` (scalar, not list) used
    to resolve deny-all with no note — the 'loses its hands without anyone
    deciding' class."""
    (tmp_path / POLICY_FILENAME).write_text("visit:\n  tools: diary_list\n", encoding="utf-8")
    grant = resolve_tool_grant(tmp_path, "visit")
    assert grant.tools == ()  # still an explicit-file resolution, not a widen
    assert any("non-list" in n for n in grant.notes)
