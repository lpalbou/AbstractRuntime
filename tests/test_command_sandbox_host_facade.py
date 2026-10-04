"""Round 12: the command-sandbox host facade re-exports AbstractCore's objects themselves."""

import pytest

from abstractcore.tools import sandbox as core
from abstractruntime.integrations.abstractcore import command_sandbox_host as facade


def test_facade_is_pure_reexport():
    assert facade.configure_host is core.configure_host
    assert facade.host_policy is core.host_policy
    assert facade.host_sandbox_kind is core.host_sandbox_kind
    assert facade.KIND_LABELS is core.KIND_LABELS and facade.KIND_NONE == core.KIND_NONE
    assert facade.reset_host_for_tests is core._reset_host_for_tests
    assert sorted(facade.__all__) == ["KIND_LABELS", "KIND_NONE", "configure_host", "host_policy", "host_sandbox_kind", "reset_host_for_tests"]


def test_facade_configures_the_core_host_policy():
    facade.reset_host_for_tests()
    try:
        facade.configure_host(env={"PATH": "/usr/bin"}, unsandboxed_commands_allowed=False)
        assert core.host_policy()["configured"] is True
        with pytest.raises(RuntimeError):
            facade.configure_host(env={"PATH": "/usr/bin"}, unsandboxed_commands_allowed=True)
    finally:
        facade.reset_host_for_tests()
    assert facade.host_policy()["configured"] is False
