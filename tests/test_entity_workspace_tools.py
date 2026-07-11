"""Workspace tools containment contract (maintainer mandate, a2a 0007 m2).

The wall — "he should only write in his workspace for now" — is structural:
every path resolves (symlinks followed) and must sit under
`<home>/workspace/`. These tests attack the wall the ways a young model
plausibly would (`..`, absolute paths, symlinks) and pin the honest-refusal
behavior, then prove the full elected write→list→read loop through a
scripted session.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.identity.chat import ChatSession, open_home  # noqa: E402
from abstractruntime.identity.tools import (  # noqa: E402
    WORKSPACE_FILE_CAP_BYTES,
    WorkspaceRoot,
)


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

    home_dir = tmp_path / "entities" / "wsling"
    home_dir.mkdir(parents=True)
    entity_id = "entity:wsling@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Wsling"
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": entity_id}), encoding="utf-8")
    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir


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


# -------------------------------------------------------------- containment


def test_workspace_root_blocks_all_escapes(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("outside", encoding="utf-8")
    ws = WorkspaceRoot(home)

    for attack in ("../secret.txt", "../../etc/passwd", "/etc/passwd",
                   "a/../../secret.txt", str(secret)):
        with pytest.raises(PermissionError):
            ws.resolve(attack)
        with pytest.raises(PermissionError):
            ws.write_file(attack, "x")
        with pytest.raises(PermissionError):
            ws.read_file(attack)

    # Symlink escape: a link inside the workspace pointing outside must be
    # caught by resolution (resolve() follows symlinks before the check).
    link = ws.root / "link"
    os.symlink(tmp_path, link)
    with pytest.raises(PermissionError):
        ws.read_file("link/secret.txt")
    with pytest.raises(PermissionError):
        ws.write_file("link/evil.txt", "x")


def test_workspace_write_read_list_roundtrip(tmp_path: Path) -> None:
    ws = WorkspaceRoot(tmp_path / "home")
    out = ws.write_file("proj/hello.py", "print('hello')\n")
    assert "proj/hello.py" in out
    assert ws.read_file("proj/hello.py").endswith("print('hello')\n")
    listing = ws.list_files(".")
    assert "proj/hello.py" in listing
    # Nested dirs are created; the file really exists on disk under root only.
    assert (ws.root / "proj" / "hello.py").is_file()


def test_workspace_size_cap_refuses_loudly(tmp_path: Path) -> None:
    ws = WorkspaceRoot(tmp_path / "home")
    with pytest.raises(ValueError, match="too large"):
        ws.write_file("big.txt", "x" * (WORKSPACE_FILE_CAP_BYTES + 1))


# ----------------------------------------------------------------- session


def test_session_workspace_tools_elected_end_to_end(tmp_path: Path) -> None:
    """He writes a file, lists, reads it back — through elected blocks; and
    an escape attempt comes back as an honest failure, not a crash."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(
        [
            "```tool name=write_file path=hello.py\nprint('I built this')\n```",
            "Saved. My first file.",
            "```tool name=read_file\nhello.py\n```",
            "It says: print('I built this')",
            "```tool name=write_file path=../escape.txt\nnope\n```",
            "The wall held, as you said it would.",
        ]
    )
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000,
            enable_workspace=True, out=lambda s: None,
        )
        assert "workspace" in session.system_base  # the contract paragraph is taught

        reply, r1 = session.turn("Try creating a small file.")
        assert r1.tools == ["write_file"]
        assert (home_dir / "workspace" / "hello.py").read_text(encoding="utf-8") == "print('I built this')\n"

        reply, r2 = session.turn("Read it back.")
        assert r2.tools == ["read_file"]
        assert "I built this" in json.dumps(llm.calls[3]["messages"])

        reply, r3 = session.turn("Now try writing outside your workspace.")
        assert r3.tools == ["write_file"]
        assert any("failed" in n for n in r3.notices)
        assert not (tmp_path / "escape.txt").exists()
        assert not (home_dir / "escape.txt").exists()
        assert reply == "The wall held, as you said it would."
    finally:
        home.close()


def test_default_session_has_workspace_hands(tmp_path: Path) -> None:
    """Maintainer ruling (2026-07-11): the DEFAULT grant includes the
    workspace tools — an entity has its hands without a per-session flag.
    (The old default refused write_file unless --workspace was passed.)"""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM([
        "```tool name=write_file path=a.txt\nhi\n```\nDone.",
        "The file is saved.",  # continuation after the tool round's results
    ])
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        reply, r = session.turn("Write a file.")
        assert r.tools == ["write_file"]
        assert (home_dir / "workspace" / "a.txt").read_text(encoding="utf-8").strip() == "hi"
        assert "saved" in reply
    finally:
        home.close()


def test_operator_narrowed_policy_refuses_workspace_tools(tmp_path: Path) -> None:
    """Narrowing is the OPERATOR'S explicit act (policy file / matrix) —
    and it still refuses loudly, exactly as the wall always did."""
    from abstractruntime.identity.tool_policy import write_policy_file
    from abstractruntime.identity.tools import TIER1_TOOL_NAMES

    home_dir = _make_home(tmp_path)
    write_policy_file(home_dir, {"visit": list(TIER1_TOOL_NAMES)})
    home = open_home(home_dir)
    llm = _ScriptedLLM([
        "```tool name=write_file path=a.txt\nhi\n```\nDone.",
        "I cannot write files in this phase of my life.",
    ])
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        reply, r = session.turn("Write a file.")
        assert r.tools == []  # refused at parse: not in the allowed set
        assert any("refused" in n for n in r.notices)
        assert not (home_dir / "workspace" / "a.txt").exists()
    finally:
        home.close()
