from __future__ import annotations


def test_default_tool_specs_include_skim_web_tools_before_full_web_tools() -> None:
    from abstractruntime.integrations.abstractcore.default_tools import get_default_tools, list_default_tool_specs

    specs = [s for s in list_default_tool_specs() if isinstance(s, dict)]
    names = [str(s.get("name") or "") for s in specs]

    assert "skim_websearch" in names
    assert "skim_url" in names
    assert "web_search" in names
    assert "fetch_url" in names
    assert names.index("skim_websearch") < names.index("web_search")
    assert names.index("skim_url") < names.index("fetch_url")

    tool_names = [getattr(tool, "__name__", "") for tool in get_default_tools()]
    assert "skim_websearch" in tool_names
    assert "skim_url" in tool_names
