"""Per-phase tool policy + workspace mounts (maintainer asks, 2026-07-08).

- tool_policy.yaml is the operator's word on which tools each phase of life
  holds; a missing file gives the RULED defaults (12:37 matrix ruling + Q1
  c684: visit + tasked + own_time = the full set, sleep = read-only
  exploration minus the diary).
- The phase vocabulary is the four c607 keys (visit/tasked/own_time/sleep);
  the legacy "resident" spelling maps to own_time LOUDLY on both axes for
  the migration window (F7/N7 — pins below) and dies before release.
- workspace_mounts.json whitelists extra roots under mounts/<name>/ with an
  honest mode; ro refuses writes; containment applies per root.
- The turn report carries the probe surface: tool_details + files touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.identity.chat import ChatSession, open_home  # noqa: E402
from abstractruntime.identity.tool_policy import (  # noqa: E402
    resolve_tool_grant,
    write_policy_file,
)
from abstractruntime.identity.tools import (  # noqa: E402
    TIER1_TOOL_NAMES,
    WORKSPACE_TOOL_NAMES,
    WorkspaceRoot,
    write_workspace_mounts,
)


class _ScriptedLLM:
    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def generate(self, *, messages, system_prompt):
        self.calls.append({"messages": list(messages), "system_prompt": system_prompt})

        class _R:
            pass

        r = _R()
        r.content = self.replies.pop(0)
        return r


def _make_home(tmp_path: Path) -> Path:
    import copy

    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    home_dir = tmp_path / "entities" / "testling"
    home_dir.mkdir(parents=True)
    entity_id = "entity:testling@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Testling"
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": entity_id}), encoding="utf-8")

    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    result = engram(ms, spark, owner_id=entity_id)
    assert result.created is True
    store.close()
    journal.close()
    return home_dir


# ------------------------------------------------------------- tool policy


def test_missing_policy_file_gives_the_ruled_defaults(tmp_path: Path) -> None:
    """Maintainer rulings (12:37 matrix + Q1 c684): every NEW summoned
    entity defaults to the full set for visit, tasked and own_time (hands
    by default; narrowing is the operator's explicit act — for own_time
    the brake is the enabled flag + durable grant, never handlessness),
    and sleep defaults to read-only exploration WITHOUT the diary — "the
    entity can't act/change the environment while sleeping, but it can
    recall or search information; it won't be in its diary" — and without
    writes. The old per-session enable_workspace gate no longer subtracts
    from the default visit grant."""
    from abstractruntime.identity.tool_policy import SLEEP_DEFAULT_TOOL_NAMES

    grant = resolve_tool_grant(tmp_path, "visit", enable_workspace=False)
    assert grant.tools == TIER1_TOOL_NAMES + WORKSPACE_TOOL_NAMES
    assert grant.source == "default"
    grant_ws = resolve_tool_grant(tmp_path, "visit", enable_workspace=True)
    assert grant_ws.tools == TIER1_TOOL_NAMES + WORKSPACE_TOOL_NAMES
    own_time = resolve_tool_grant(tmp_path, "own_time")
    assert own_time.tools == TIER1_TOOL_NAMES + WORKSPACE_TOOL_NAMES
    tasked = resolve_tool_grant(tmp_path, "tasked")
    assert tasked.tools == TIER1_TOOL_NAMES + WORKSPACE_TOOL_NAMES

    sleep = resolve_tool_grant(tmp_path, "sleep")
    assert sleep.tools == SLEEP_DEFAULT_TOOL_NAMES
    # The rationale, pinned structurally: no writes, no diary, reads yes.
    assert "write_file" not in sleep.tools
    assert "diary_list" not in sleep.tools and "diary_read" not in sleep.tools
    assert {"web_search", "fetch_url", "read_memory", "search_memory", "read_file", "list_files"} == set(sleep.tools)


def test_sleep_default_is_in_canonical_order(tmp_path: Path) -> None:
    """The constant claims canonical ALL_TOOL_NAMES order — pin it, or a
    reorder silently diverges the default tuple from a policy-file
    roundtrip of the same six names (resolve dedups in canonical order)."""
    from abstractruntime.identity.tool_policy import ALL_TOOL_NAMES, SLEEP_DEFAULT_TOOL_NAMES

    assert SLEEP_DEFAULT_TOOL_NAMES == tuple(t for t in ALL_TOOL_NAMES if t in set(SLEEP_DEFAULT_TOOL_NAMES))
    # Roundtrip parity: a file naming the same six (any order) resolves
    # to the identical tuple as the default.
    write_policy_file(tmp_path, {"sleep": list(reversed(SLEEP_DEFAULT_TOOL_NAMES))})
    assert resolve_tool_grant(tmp_path, "sleep").tools == SLEEP_DEFAULT_TOOL_NAMES


def test_named_but_malformed_phase_falls_to_defaults_loudly(tmp_path: Path) -> None:
    """A hand-edited file with `visit:` left null used to fall to defaults
    SILENTLY — under full-set defaults that is a silent WIDEN. Now noted."""
    (tmp_path / "tool_policy.yaml").write_text("visit:\n", encoding="utf-8")
    grant = resolve_tool_grant(tmp_path, "visit")
    assert grant.source == "default"
    assert grant.tools == TIER1_TOOL_NAMES + WORKSPACE_TOOL_NAMES
    assert any("#FALLBACK" in n and "visit" in n for n in grant.notes)


def test_policy_file_is_the_operators_word(tmp_path: Path) -> None:
    write_policy_file(tmp_path, {"visit": ["diary_list", "diary_read"], "sleep": []})
    grant = resolve_tool_grant(tmp_path, "visit", enable_workspace=True)
    assert grant.tools == ("diary_list", "diary_read")  # file wins over the flag
    assert grant.source == "policy-file"
    assert grant.workspace_enabled is False
    # Unnamed phases keep their defaults.
    assert resolve_tool_grant(tmp_path, "own_time").tools == TIER1_TOOL_NAMES + WORKSPACE_TOOL_NAMES


def test_policy_tiers_add_remove_and_unknowns(tmp_path: Path) -> None:
    (tmp_path / "tool_policy.yaml").write_text(
        yaml.safe_dump({
            "own_time": {"tiers": ["tier1"], "add": ["read_file", "made_up_tool"], "remove": ["web_search"]},
        }),
        encoding="utf-8",
    )
    grant = resolve_tool_grant(tmp_path, "own_time")
    assert "web_search" not in grant.tools
    assert "read_file" in grant.tools and "diary_list" in grant.tools
    assert any("made_up_tool" in n for n in grant.notes)  # loud, never silent


def test_write_policy_file_refuses_unknown_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_policy_file(tmp_path, {"visit": ["rm_rf_slash"]})
    with pytest.raises(ValueError):
        write_policy_file(tmp_path, {"weekend": ["diary_list"]})


def test_write_policy_file_merges_per_phase(tmp_path: Path) -> None:
    """Adversary find (2026-07-11): whole-document replace materialized the
    day's resolved defaults as "the operator's word" (every real home froze
    `sleep: []` from pre-ruling matrix saves). Writes now MERGE: only named
    phases change; None deletes a phase's entry (revert to the evolving
    defaults); an emptied file is removed (absence = defaults, honestly)."""
    from abstractruntime.identity.tool_policy import POLICY_FILENAME, SLEEP_DEFAULT_TOOL_NAMES

    # Write visit only — sleep/own_time stay ABSENT (defaults apply).
    write_policy_file(tmp_path, {"visit": ["diary_list"]})
    assert resolve_tool_grant(tmp_path, "visit").tools == ("diary_list",)
    assert resolve_tool_grant(tmp_path, "sleep").tools == SLEEP_DEFAULT_TOOL_NAMES
    assert resolve_tool_grant(tmp_path, "sleep").source == "default"

    # A later write touching own_time keeps visit's entry intact.
    write_policy_file(tmp_path, {"own_time": ["diary_list", "read_memory"]})
    assert resolve_tool_grant(tmp_path, "visit").tools == ("diary_list",)
    assert resolve_tool_grant(tmp_path, "own_time").tools == ("diary_list", "read_memory")

    # None deletes the phase entry — visit reverts to the ruled default.
    write_policy_file(tmp_path, {"visit": None})
    assert resolve_tool_grant(tmp_path, "visit").tools == TIER1_TOOL_NAMES + WORKSPACE_TOOL_NAMES
    assert resolve_tool_grant(tmp_path, "visit").source == "default"

    # Removing the last entry removes the file (absence = all defaults).
    write_policy_file(tmp_path, {"own_time": None})
    assert not (tmp_path / POLICY_FILENAME).exists()


def test_explicit_empty_grant_denies_all_never_falls_open(tmp_path: Path) -> None:
    """Adversary find (2026-07-11): the parsers' `or TIER1` default treated
    the operator's explicit visit:[] ZERO grant as "unspecified" and fell
    open to tier-1 (incl. diary_read of private entries) in the in-process
    driver. None = tier-1 default; () = deny all, honest refusals."""
    from abstractruntime.identity.tools import TIER1_TOOL_NAMES as T1
    from abstractruntime.identity.tools import native_tool_elections, parse_tool_blocks

    # Unit truth: None defaults, () denies.
    _, elections, notices = parse_tool_blocks("```tool name=web_search\nx\n```", ())
    assert elections == [] and any("refused" in n for n in notices)
    _, elections_default, _ = parse_tool_blocks("```tool name=web_search\nx\n```", None)
    assert [e.name for e in elections_default] == ["web_search"]
    n_elections, n_markers, n_notices = native_tool_elections(
        [{"name": "diary_read", "arguments": {"entry": "diary_ab12cd34"}}], ()
    )
    assert n_elections == [] and any("refused" in n for n in n_notices)
    assert T1  # the default set itself stays non-empty (sanity)

    # Session truth: a zero-grant policy file reaches the driver as deny-all.
    home_dir = _make_home(tmp_path)
    write_policy_file(home_dir, {"visit": []})
    home = open_home(home_dir)
    llm = _ScriptedLLM([
        "```tool name=diary_read\ndiary_ab12cd34\n```",
        "I hold no tools in this phase of my life.",
    ])
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        assert session.allowed_tools == ()
        reply, r = session.turn("Read your diary for me.")
        assert r.tools == []  # nothing ran
        assert any("refused" in n for n in r.notices)
        assert "no tools" in reply
    finally:
        home.close()


def test_session_honors_policy_and_states_the_grant(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    write_policy_file(home_dir, {"visit": ["diary_list"]})
    home = open_home(home_dir)
    # Reply 1 is ONLY a (refused) tool block, so the speak-now guard asks
    # for words; reply 2 is the spoken follow-up the person actually hears.
    llm = _ScriptedLLM([
        "```tool name=web_search\nanything\n```",
        "I cannot search the web in this phase of my life.",
    ])
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        assert session.allowed_tools == ("diary_list",)
        assert "this phase of your life grants: diary_list" in session.system_base
        reply, r = session.turn("Search the web for beavers.")
        assert r.tools == []  # web_search refused: not granted this phase
        assert any("refused" in n for n in r.notices)
        assert "I cannot search the web" in reply  # the guard got him to speak
    finally:
        home.close()


# ---------------------------------------------------------------- mounts


def test_mounts_ro_reads_but_refuses_writes(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "notes.txt").write_text("from the operator\n", encoding="utf-8")
    write_workspace_mounts(home_dir, [{"name": "shared", "path": str(shared), "mode": "ro"}])

    ws = WorkspaceRoot(home_dir)
    listing = ws.list_files(".")
    assert "mounts/shared/ (extra workspace, read-only)" in listing
    assert "from the operator" in ws.read_file("mounts/shared/notes.txt")
    with pytest.raises(PermissionError):
        ws.write_file("mounts/shared/hack.txt", "no")
    # Containment holds inside the mount too.
    with pytest.raises(PermissionError):
        ws.read_file("mounts/shared/../../home/secret.txt")
    # Unknown mounts refuse with the honest list.
    with pytest.raises(PermissionError):
        ws.read_file("mounts/ghost/x.txt")


def test_mounts_rw_allows_writes_inside_only(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    lab = tmp_path / "lab"
    lab.mkdir()
    write_workspace_mounts(home_dir, [{"name": "lab", "path": str(lab), "mode": "rw"}])

    ws = WorkspaceRoot(home_dir)
    out = ws.write_file("mounts/lab/exp/plan.md", "step one\n")
    assert "mounts/lab/exp/plan.md" in out
    assert (lab / "exp" / "plan.md").read_text(encoding="utf-8") == "step one\n"
    # The home workspace stays writable as before.
    ws.write_file("own.txt", "mine\n")
    assert (home_dir / "workspace" / "own.txt").exists()


def test_mounts_never_overlap_the_home(tmp_path: Path) -> None:
    """The wall guards itself: policy files live in the home ROOT (outside
    workspace/), so an rw mount of the home would let the entity edit its
    own grants. Inside, equal, and containing paths all refuse."""
    home_dir = tmp_path / "home"
    (home_dir / "sub").mkdir(parents=True)
    for path in (home_dir, home_dir / "sub", tmp_path):  # inside, equal, containing
        with pytest.raises(ValueError, match="overlaps the entity home"):
            write_workspace_mounts(home_dir, [{"name": "x", "path": str(path), "mode": "rw"}])


def test_mounts_file_validation_is_loud(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    with pytest.raises(ValueError):
        write_workspace_mounts(home_dir, [{"name": "a/b", "path": str(tmp_path), "mode": "ro"}])
    with pytest.raises(ValueError):
        write_workspace_mounts(home_dir, [{"name": "x", "path": str(tmp_path / "missing"), "mode": "ro"}])
    with pytest.raises(ValueError):
        write_workspace_mounts(home_dir, [{"name": "x", "path": str(tmp_path), "mode": "yolo"}])


# --------------------------------------------------------- probe surface


def test_turn_report_carries_tool_details_and_files(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM([
        "```tool name=write_file path=notes/day1.md\nfirst note\n```",
        "Saved my note.",
    ])
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000,
            enable_workspace=True, out=lambda s: None,
        )
        _, r = session.turn("Keep a file about today.")
        assert r.tools == ["write_file"]
        # Details carry the argument AND what the lookup returned (the
        # observability ask: results were invisible to the operator).
        assert len(r.tool_details) == 1
        assert r.tool_details[0]["name"] == "write_file"
        assert r.tool_details[0]["arg"] == "notes/day1.md"
        assert "notes/day1.md" in r.tool_details[0]["result"]
        assert r.files == [{"path": "notes/day1.md", "action": "wrote"}]
        # The enriched memories carry the probe fields.
        for m in r.memories:
            assert "digest" in m and "global_count" in m and "activation" in m
    finally:
        home.close()
