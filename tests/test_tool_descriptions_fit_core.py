"""Every entity tool description fits AbstractCore's limit (abstractcore/tools/core.py _MAX_TOOL_DESCRIPTION_CHARS = 200).

Regression: 0a8fc67 shipped a 249-character `topics` description; AbstractCore refused the tool definition and the
gateway's mention door answered 500 on that runtime (found by the gateway 0.21.0 gate, 2026-11-10)."""
from abstractcore.tools.core import _MAX_TOOL_DESCRIPTION_CHARS

from abstractruntime.identity.tools import TOOL_DESCRIPTORS


def test_every_entity_tool_description_fits_core_limit():
    too_long = {n: len(d.description) for n, d in TOOL_DESCRIPTORS.items()
                if len(d.description) > _MAX_TOOL_DESCRIPTION_CHARS}
    assert too_long == {}, too_long
