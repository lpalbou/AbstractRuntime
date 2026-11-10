"""The runtime's own communities compute is RETIRED (2026-11-10; identity/communities.py): topics are memory's
(`abstractmemory.communities`), formed during the sleep (identity/topics.py) and read with the `topics` tool.

Pins: the last caller fails LOUDLY (never the old shape, never a silent empty answer), as the ImportError the
gateway route already maps to "501 communities unavailable"; no community algorithm or keyword label survives in the
runtime's source."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import abstractruntime
from abstractruntime.identity import communities

RUNTIME_SRC = Path(abstractruntime.__file__).resolve().parent


def test_memory_communities_fails_loudly_naming_what_to_serve() -> None:
    with pytest.raises(communities.CommunitiesRetired) as retired:
        communities.memory_communities(object())
    assert isinstance(retired.value, ImportError)  # the gateway route answers 501 with the sentence, never a 500
    assert "is retired (2026-11-10)" in str(retired.value)
    assert "AbstractMemory's communities() payload (version 2" in str(retired.value)


def test_no_community_algorithm_or_keyword_label_is_left_in_the_runtime() -> None:
    """I9: the runtime computes no communities and imports or renders no keyword label."""
    banned = re.compile(r"_LABEL_STOP|_community_labels|KEYWORD_BRIDGE|louvain_partition|def collect_graph|"
                        r"RESOLUTION_SWEEP|_add_keyword_bridges")
    hits = [f"{p.relative_to(RUNTIME_SRC)}:{n}" for p in RUNTIME_SRC.rglob("*.py")
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if banned.search(line)]
    assert hits == [], hits
    assert [n for n in dir(communities) if not n.startswith("__")] == sorted(
        ["Any", "CommunitiesRetired", "NoReturn", "RETIRED", "annotations", "memory_communities"])
