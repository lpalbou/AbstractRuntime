"""Host-supplied `disabled_toolsets` at registration (gateway c4877 seam,
camera default-off wave).

Served-disabled must also mean ABSENT FROM RUN REGISTRATION — a workflow
bypassing discovery must not reach a toolset the host's settings registry
turned off. PARAMETER-EXPLICIT by design (dm#10 killed the camera enable
env; dm#177 forbids new behavior envs): the host passing its console-held
configuration IS the app deciding. Default behavior (no param) is
byte-identical — availability predicates stay the only default gate.
"""

from abstractruntime.integrations.abstractcore.default_tools import (
    get_default_tools,
    get_default_toolsets,
    list_default_tool_specs,
)


def test_disabled_toolset_absent_from_registration():
    base = get_default_toolsets()
    assert "web" in base  # always-available set used as the fixture
    filtered = get_default_toolsets(disabled_toolsets={"web"})
    assert "web" not in filtered
    assert "files" in filtered  # subtraction, never a rewrite


def test_disabled_toolset_tools_absent_from_flat_list():
    web_names = set()
    for tool in get_default_toolsets()["web"]["tools"]:
        web_names.add(getattr(tool, "__name__", ""))
    assert web_names
    flat = {getattr(t, "__name__", "") for t in get_default_tools(disabled_toolsets={"web"})}
    assert not (web_names & flat)


def test_disabled_toolset_specs_absent():
    specs = list_default_tool_specs(disabled_toolsets={"web"})
    toolsets = {s.get("toolset") for s in specs}
    assert "web" not in toolsets
    assert "files" in toolsets


def test_no_param_is_byte_identical_default():
    assert set(get_default_toolsets().keys()) == set(
        get_default_toolsets(disabled_toolsets=None).keys()
    )
    assert set(get_default_toolsets().keys()) == set(
        get_default_toolsets(disabled_toolsets=[]).keys()
    )


def test_unknown_ids_are_noop():
    base = set(get_default_toolsets().keys())
    filtered = set(get_default_toolsets(disabled_toolsets={"nonexistent-toolset"}).keys())
    assert base == filtered
